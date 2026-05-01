from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from csf_recommendation_engine.core.config import Settings
from csf_recommendation_engine.domain.reranking import (
    RerankWeights,
    rerank_candidates_with_scores,
)


def build_proxy_string(*, desk: str, structure: str, side: str) -> str:
    target_side = "SELL" if str(side).upper() == "BUY" else "BUY"
    return f"{str(desk).upper().replace(' ', '-')}_{str(structure).upper().replace(' ', '-')}_{target_side}"


def generate_ranked_candidates(
    *,
    req,
    model_data: dict,
    settings: Settings,
    top_k: int | None = None,
    now_local: datetime | None = None,
) -> tuple[str, list[dict]]:
    live = model_data["live"]
    mats = model_data["mats"]
    proxy_str = build_proxy_string(desk=req.desk, structure=req.structure, side=req.side)

    if proxy_str not in live["proxy_index"]:
        raise ValueError(f"No liquidity archetype for {proxy_str}")

    target_proxy_int = live["proxy_index"][proxy_str]
    n_users = len(live["entity_index"])
    scores = live["model"].predict(
        user_ids=np.arange(n_users),
        item_ids=np.full(n_users, target_proxy_int),
        user_features=mats["user_features_matrix"],
        item_features=mats["item_features_matrix"],
    )

    pool_size = max(int(settings.rerank_candidate_pool_size), int(top_k or req.top_k))
    if pool_size >= n_users:
        candidate_indices = np.arange(n_users)
    else:
        candidate_indices = np.argpartition(scores, -pool_size)[-pool_size:]
        candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

    valid_entity_ids = model_data.get("client_entity_ids")
    candidates: list[dict] = []
    for i in candidate_indices:
        entity_id = model_data["rev_entity"][int(i)]
        if entity_id == req.client_id:
            continue
        if valid_entity_ids is not None and entity_id not in valid_entity_ids:
            continue
        candidates.append({"client_id": entity_id, "affinity": float(scores[int(i)])})

    heuristics = model_data.get("heuristics")
    if settings.rerank_enabled and heuristics is not None:
        if now_local is None:
            now_local = datetime.now(ZoneInfo(settings.app_timezone))
        weights = RerankWeights(
            lightfm=float(settings.rerank_weight_lightfm),
            time_affinity=float(settings.rerank_weight_time_affinity),
            recency=float(settings.rerank_weight_recency),
            size_fit=float(settings.rerank_weight_size_fit),
        )
        return proxy_str, rerank_candidates_with_scores(
            candidates=candidates,
            venue=req.venue,
            instrument_name=req.instrument_name,
            quantity=float(req.quantity),
            now_local=now_local,
            heuristics=heuristics,
            weights=weights,
            recency_halflife_days=float(settings.rerank_recency_halflife_days),
            apply_venue_filter=True,
        )[: int(top_k or req.top_k)]

    return proxy_str, [
        {
            **candidate,
            "lightfm_normalized": candidate["affinity"],
            "size_score": 0.0,
            "recency_score": 0.0,
            "time_score": 0.0,
            "final_score": candidate["affinity"],
        }
        for candidate in candidates[: int(top_k or req.top_k)]
    ]
