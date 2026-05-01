from __future__ import annotations

import asyncio
import json
import logging
import math
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from csf_recommendation_engine.core.config import get_settings
from csf_recommendation_engine.core.startup import (
    get_model_data,
    preload_heuristics_state,
)
from csf_recommendation_engine.domain.recommendation_engine import generate_ranked_candidates
from csf_recommendation_engine.infra.db.pool import get_db_pool
from csf_recommendation_engine.infra.db.trades import (
    fetch_working_trades,
    fetch_recent_ai_recommendations,
    fetch_recent_cross_block_matches,
    insert_ai_recommendations,
    insert_cross_block_matches,
)

logger = logging.getLogger(__name__)

refresh_run_lock = asyncio.Lock()
ADVISORY_LOCK_ID = 424243


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _RefreshCounters:
    total_working_trades: int = 0
    recommendation_attempts: int = 0
    recommendations_inserted: int = 0
    recommendations_skipped_unknown: int = 0
    recommendations_skipped_duplicate: int = 0
    recommendation_failures: int = 0
    mutual_pairs_evaluated: int = 0
    cross_matches_inserted: int = 0
    cross_matches_skipped_unknown: int = 0
    cross_matches_skipped_duplicate: int = 0
    match_failures: int = 0


@dataclass
class _TradeRequest:
    """Lightweight request-like object for generate_ranked_candidates."""
    client_id: str
    desk: str
    structure: str
    side: str
    venue: str
    instrument_name: str
    quantity: float
    top_k: int = 3


def _quantity_to_lots(quantity) -> int:
    """Convert a trade quantity to an integer lot count."""
    lots = int(math.floor(float(quantity)))
    return lots if lots > 0 else 0


def calculate_match_score(
    *,
    score_a_for_b: float,
    score_b_for_a: float,
) -> float:
    """Compute a normalized mutual match score in [0.0, 1.0].

    Currently uses the geometric mean of the two final scores.
    """
    if score_a_for_b <= 0 or score_b_for_a <= 0:
        return 0.0
    return math.sqrt(score_a_for_b * score_b_for_a)


# ---------------------------------------------------------------------------
# Advisory lock helpers
# ---------------------------------------------------------------------------


async def _acquire_advisory_lock(conn) -> bool:
    return bool(await conn.fetchval("SELECT pg_try_advisory_lock($1)", ADVISORY_LOCK_ID))


async def _release_advisory_lock(conn) -> None:
    await conn.fetchval("SELECT pg_advisory_unlock($1)", ADVISORY_LOCK_ID)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_rec_refresh_pipeline() -> None:
    settings = get_settings()
    if not settings.rec_refresh_enabled:
        logger.info("Rec refresh pipeline disabled via config")
        return

    async with refresh_run_lock:
        pool = get_db_pool()
        async with pool.acquire() as conn:
            acquired = await _acquire_advisory_lock(conn)
            if not acquired:
                logger.info("Rec refresh pipeline skipped because advisory lock is already held")
                return

            try:
                logger.info("Starting rec refresh pipeline")
                counters = _RefreshCounters()

                # -------------------------------------------------------
                # Step 1: Reload heuristics index
                # -------------------------------------------------------
                entity_path = (
                    Path(settings.heuristics_artifact_dir)
                    / settings.heuristics_entity_features_latest_filename
                )
                inst_path = (
                    Path(settings.heuristics_artifact_dir)
                    / settings.heuristics_instrument_features_latest_filename
                )
                await preload_heuristics_state(
                    entity_features_path=entity_path,
                    instrument_features_path=inst_path,
                    require=settings.rerank_require_heuristics,
                )
                logger.info("Heuristics artifacts reloaded into memory")

                # -------------------------------------------------------
                # Step 2: Load qualifying working trades
                # -------------------------------------------------------
                working_trades = await fetch_working_trades(
                    conn, min_quantity=settings.nightly_min_working_quantity
                )
                counters.total_working_trades = len(working_trades)
                logger.info(
                    "Loaded working trades",
                    extra={"total_working_trades": counters.total_working_trades},
                )

                if not working_trades:
                    logger.info("No qualifying working trades found; skipping recommendation and match stages")
                    _emit_summary(counters)
                    return

                # Load shared model data
                try:
                    model_data = await get_model_data()
                except ValueError:
                    logger.error("Champion model is not loaded; cannot run recommendation pass")
                    _emit_summary(counters)
                    return

                # Entity name map for resolving candidates
                entity_name_map: dict[str, str] = model_data.get("client_entity_names", {})

                # -------------------------------------------------------
                # Step 3: Fetch existing records for deduplication
                # -------------------------------------------------------
                existing_recs = await fetch_recent_ai_recommendations(
                    conn, agent_name="Caddie AI Match Engine"
                )
                existing_rec_keys: set[tuple[str, str, str]] = {
                    (str(r["entity_id"]), r["product"], r["description"])
                    for r in existing_recs
                }
                logger.info(
                    "Loaded existing recommendations for deduplication",
                    extra={"existing_rec_count": len(existing_rec_keys)},
                )

                existing_matches = await fetch_recent_cross_block_matches(conn)
                existing_match_keys: set[tuple[str, str, str]] = {
                    (str(r["buyer_entity_id"]), str(r["seller_entity_id"]), r["product_name"])
                    for r in existing_matches
                }
                logger.info(
                    "Loaded existing matches for deduplication",
                    extra={"existing_match_count": len(existing_match_keys)},
                )

                # -------------------------------------------------------
                # Step 4: Recommendation pass
                # -------------------------------------------------------
                # Stores trade_id -> list of top-k candidate dicts for cross-matching
                trade_recommendations: dict[str, list[dict]] = {}
                # Stores trade_id -> trade dict for cross-matching lookup
                trade_lookup: dict[str, dict] = {}

                trades_saved_airec = 0
                for trade in working_trades:
                    trade_id = str(trade["trade_id"])
                    entity_id = str(trade["entity_id"])
                    trade_lookup[trade_id] = trade

                    # Skip trades with missing entity_id
                    if not entity_id:
                        logger.warning("Skipping trade with missing entity_id", extra={"trade_id": trade_id})
                        continue

                    counters.recommendation_attempts += 1
                    try:
                        req = _TradeRequest(
                            client_id=entity_id,
                            desk=str(trade.get("desk_type") or ""),
                            structure=str(trade.get("structure") or ""),
                            side=str(trade["side"]),
                            venue=str(trade.get("venue") or ""),
                            instrument_name=str(trade.get("instrument_name") or ""),
                            quantity=float(trade["quantity"]),
                            top_k=settings.nightly_top_k,
                        )
                        _proxy_str, candidates = generate_ranked_candidates(
                            req=req,
                            model_data=model_data,
                            settings=settings,
                            top_k=settings.nightly_top_k,
                        )
                    except Exception:
                        counters.recommendation_failures += 1
                        logger.exception(
                            "Recommendation generation failed for trade",
                            extra={"trade_id": trade_id},
                        )
                        continue

                    # Resolve entity names and filter unknown
                    valid_candidates: list[dict] = []
                    for cand in candidates:
                        cand_id = str(cand["client_id"])
                        cand_name = entity_name_map.get(cand_id, "Unknown Entity")
                        if cand_name == "Unknown Entity":
                            counters.recommendations_skipped_unknown += 1
                            continue
                        cand["entity_name"] = cand_name
                        valid_candidates.append(cand)

                    trade_recommendations[trade_id] = valid_candidates

                    # Build insert rows for ai_recommendations
                    rec_rows: list[dict] = []
                    for rank, cand in enumerate(valid_candidates[:1], start=1):
                        if counters.recommendations_skipped_duplicate > 3:
                            break
                        # TODO: USER REQUEST - COME BACK TO THIS LATER
                        description = f"{cand.get('entity_name', 'Unknown')} is a potential counterparty for {trade.get('side')} {trade.get('instrument_name')}"
                        rec_key = (entity_id, str(trade.get("instrument_name") or ""), description)

                        if rec_key in existing_rec_keys:
                            counters.recommendations_skipped_duplicate += 1
                            continue

                        rec_rows.append({
                            "entity_id": entity_id,
                            "recommendation_type": "Cross-Block Recommendation",
                            "product": str(trade.get("instrument_name") or ""),
                            "description": description,
                            "details": json.dumps({
                                # "source_trade_id": trade_id,
                                "Desk": str(trade.get("desk_type") or ""),
                                "Structure": str(trade.get("structure") or ""),
                                # "source_side": str(trade["side"]),
                                # "source_venue": str(trade.get("venue") or ""),
                                "Quantity": float(trade["quantity"]),
                                # "candidate_entity_id": str(cand["client_id"]),
                                "Counterparty": cand.get("entity_name", ""),
                                # "lightfm_score": cand.get("lightfm_normalized", 0.0),
                                # "size_score": cand.get("size_score", 0.0),
                                # "recency_score": cand.get("recency_score", 0.0),
                                # "time_score": cand.get("time_score", 0.0),
                                # "final_score": cand.get("final_score", 0.0),
                                # "rank": rank,
                            }),
                            "created_by_ai_agent": "Caddie AI Match Engine",
                        })

                    if rec_rows and trades_saved_airec < 3:
                        try:
                            inserted = await insert_ai_recommendations(conn, rec_rows)
                            counters.recommendations_inserted += inserted
                            trades_saved_airec += 1
                        except Exception:
                            counters.recommendation_failures += 1
                            logger.exception(
                                "Failed to insert recommendations for trade",
                                extra={"trade_id": trade_id},
                            )

                logger.info(
                    "Recommendation pass completed",
                    extra={
                        "attempts": counters.recommendation_attempts,
                        "inserted": counters.recommendations_inserted,
                        "skipped_unknown": counters.recommendations_skipped_unknown,
                        "skipped_duplicate": counters.recommendations_skipped_duplicate,
                        "failures": counters.recommendation_failures,
                    },
                )

                # -------------------------------------------------------
                # Step 5: Cross-block matching pass
                # -------------------------------------------------------
                # Build index: entity_id -> list of trade_ids that recommended it
                entity_to_recommending_trades: dict[str, dict[str, float]] = {}
                for trade_id, candidates in trade_recommendations.items():
                    for cand in candidates:
                        cand_entity = str(cand["client_id"])
                        if cand_entity not in entity_to_recommending_trades:
                            entity_to_recommending_trades[cand_entity] = {}
                        entity_to_recommending_trades[cand_entity][trade_id] = cand.get("final_score", 0.0)

                # Check all pairs for mutual recommendations
                trade_ids = list(trade_recommendations.keys())
                seen_pairs: set[tuple[str, str]] = set()
                match_rows: list[dict] = []

                for i, tid_a in enumerate(trade_ids):
                    trade_a = trade_lookup.get(tid_a)
                    if trade_a is None:
                        continue
                    entity_a = str(trade_a["entity_id"])
                    side_a = str(trade_a["side"]).upper()

                    for j, tid_b in enumerate(trade_ids):
                        if i >= j:
                            continue  # avoid duplicates and self-pairs

                        pair_key = (min(tid_a, tid_b), max(tid_a, tid_b))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        trade_b = trade_lookup.get(tid_b)
                        if trade_b is None:
                            continue
                        entity_b = str(trade_b["entity_id"])
                        side_b = str(trade_b["side"]).upper()

                        counters.mutual_pairs_evaluated += 1

                        try:
                            # Same product context required
                            product_a = str(trade_a.get("instrument_name") or "")
                            product_b = str(trade_b.get("instrument_name") or "")
                            if product_a != product_b:
                                continue

                            # Check mutual recommendation
                            a_recs = trade_recommendations.get(tid_a, [])
                            a_recommends_b = any(
                                str(c["client_id"]) == entity_b for c in a_recs
                            )
                            if not a_recommends_b:
                                continue

                            b_recs = trade_recommendations.get(tid_b, [])
                            b_recommends_a = any(
                                str(c["client_id"]) == entity_a for c in b_recs
                            )
                            if not b_recommends_a:
                                continue

                            # Get the scores for the match calculation
                            score_a_for_b = next(
                                (c.get("final_score", 0.0) for c in a_recs if str(c["client_id"]) == entity_b),
                                0.0,
                            )
                            score_b_for_a = next(
                                (c.get("final_score", 0.0) for c in b_recs if str(c["client_id"]) == entity_a),
                                0.0,
                            )

                            # Compute match score
                            match_score = calculate_match_score(
                                score_a_for_b=score_a_for_b,
                                score_b_for_a=score_b_for_a,
                            )
                            if match_score <= settings.cross_match_threshold:
                                continue

                            # Require BUY/SELL pairing
                            if side_a not in ("BUY", "SELL") or side_b not in ("BUY", "SELL"):
                                continue
                            if side_a == side_b:
                                continue  # need opposite sides

                            # Assign buyer/seller
                            if side_a == "BUY":
                                buyer_trade, seller_trade = trade_a, trade_b
                            else:
                                buyer_trade, seller_trade = trade_b, trade_a

                            buyer_entity_id = str(buyer_trade["entity_id"])
                            seller_entity_id = str(seller_trade["entity_id"])

                            # Data integrity: buyer != seller
                            if buyer_entity_id == seller_entity_id:
                                continue

                            # Unknown entity check
                            buyer_name = entity_name_map.get(buyer_entity_id, "Unknown Entity")
                            seller_name = entity_name_map.get(seller_entity_id, "Unknown Entity")
                            if buyer_name == "Unknown Entity" or seller_name == "Unknown Entity":
                                counters.cross_matches_skipped_unknown += 1
                                continue

                            # Lot conversion
                            buyer_lots = _quantity_to_lots(buyer_trade["quantity"])
                            seller_lots = _quantity_to_lots(seller_trade["quantity"])
                            if buyer_lots <= 0 or seller_lots <= 0:
                                logger.warning(
                                    "Skipping match with non-positive lots",
                                    extra={
                                        "buyer_lots": buyer_lots,
                                        "seller_lots": seller_lots,
                                        "trade_a": tid_a,
                                        "trade_b": tid_b,
                                    },
                                )
                                continue

                            # Deduplication check
                            match_key = (buyer_entity_id, seller_entity_id, product_a)
                            if match_key in existing_match_keys:
                                counters.cross_matches_skipped_duplicate += 1
                                continue

                            # Match percentage
                            match_percentage = max(1, min(100, round(match_score * 100)))

                            # TODO: USER REQUEST - COME BACK TO THIS LATER
                            match_rows.append({
                                "product_name": product_a,
                                "match_percentage": match_percentage,
                                "buyer_entity_id": buyer_entity_id,
                                "buyer_side": "BUY",
                                "buyer_lots": buyer_lots,
                                "seller_entity_id": seller_entity_id,
                                "seller_side": "SELL",
                                "seller_lots": seller_lots,
                                "description": (
                                    f"{product_a}: {buyer_name} (BUY) <-> {seller_name} (SELL) "
                                    f"match={match_percentage}%"
                                ),
                            })
                        except Exception:
                            counters.match_failures += 1
                            logger.exception(
                                "Cross-match evaluation failed for pair",
                                extra={"trade_a": tid_a, "trade_b": tid_b},
                            )

                # Insert all match rows
                if match_rows:
                    try:
                        inserted = await insert_cross_block_matches(conn, match_rows[:3])
                        counters.cross_matches_inserted += inserted
                    except Exception:
                        counters.match_failures += 1
                        logger.exception("Failed to insert cross-block match rows")

                # -------------------------------------------------------
                # Step 6: Emit summary
                # -------------------------------------------------------
                _emit_summary(counters)

            finally:
                with suppress(Exception):
                    await _release_advisory_lock(conn)
                logger.info("Rec refresh pipeline finished")


def _emit_summary(counters: _RefreshCounters) -> None:
    logger.info(
        "Rec refresh pipeline summary",
        extra={
            "total_working_trades": counters.total_working_trades,
            "recommendation_attempts": counters.recommendation_attempts,
            "recommendations_inserted": counters.recommendations_inserted,
            "recommendations_skipped_unknown": counters.recommendations_skipped_unknown,
            "recommendations_skipped_duplicate": counters.recommendations_skipped_duplicate,
            "recommendation_failures": counters.recommendation_failures,
            "mutual_pairs_evaluated": counters.mutual_pairs_evaluated,
            "cross_matches_inserted": counters.cross_matches_inserted,
            "cross_matches_skipped_unknown": counters.cross_matches_skipped_unknown,
            "cross_matches_skipped_duplicate": counters.cross_matches_skipped_duplicate,
            "match_failures": counters.match_failures,
        },
    )
