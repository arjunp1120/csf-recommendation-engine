"""Voice-inquiry match endpoint (demo, local-disk persistence).

Pipeline (each step is recorded in ``pipeline_trace`` on the response AND
logged via ``logger.info`` with structured ``extra={...}`` so you can see
exactly what is loaded, when, what is selected/filtered, and what each
DAF agent saw and returned):

  01 receive_request               — payload + top_k
  02 lookup_intelligence_service   — service presence + configured agent ids
  03 parse_ioi_tags                — agent prompt, raw response, parsed tags
  04 build_recommend_request       — RecommendRequest built from tags
  05 load_model_data               — model artifact summary
  06 generate_ranked_candidates    — proxy string + ranked top-K with all scores
  07 resolve_entity_names          — db-backed entity_id -> name resolution
  08 load_heuristics               — heuristics index summary
  09 load_dossier_cache            — cached dossier hit/miss map
  10 load_all_inquiries            — full inquiry file load + per-candidate counts
  11 enrich_candidate_<i>          — per-candidate dossier build/hit + inquiry list
  12 match_voice_inquiry           — matcher agent prompt, raw response, parsed
  13 append_inquiry                — new inquiry written to disk
  14 respond                       — totals + total elapsed ms
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request

from csf_recommendation_engine.core.state import app_state
from csf_recommendation_engine.domain.intelligence_layer import IntelligenceService
from csf_recommendation_engine.domain.recommendation_engine import generate_ranked_candidates
from csf_recommendation_engine.domain.schemas import (
    RecommendRequest,
    VoiceInquiryMatchRequest,
    VoiceInquiryMatchResponse,
)
from csf_recommendation_engine.domain.voice_inquiry_store import (
    append_inquiry,
    group_inquiries_by_entity,
    load_dossiers,
    load_inquiries,
    upsert_dossier,
)
from csf_recommendation_engine.infra.db.pool import get_optional_db_connection

router = APIRouter(prefix="/voice_inquiry", tags=["voice_inquiry"])

logger = logging.getLogger(__name__)


_REQUIRED_TAG_KEYS_FOR_RECOMMEND = ("desk", "structure", "side", "venue", "instrument_name", "qty")


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


# def _trace_step(trace: list[dict[str, Any]], t0: float, step: str, summary: str, data: dict | None = None) -> None:
#     """Append a trace entry AND emit a structured INFO log line.

#     `step` is a sortable, snake_case identifier. `summary` is a short
#     human-readable label. `data` is arbitrary structured detail.
#     """
#     elapsed_ms = int((time.monotonic() - t0) * 1000)
#     entry = {
#         "step": step,
#         "elapsed_ms": elapsed_ms,
#         "summary": summary,
#         "data": data or {},
#     }
#     trace.append(entry)
#     logger.info(
#         "voice_inquiry.%s | %s",
#         step,
#         summary,
#         extra={"elapsed_ms": elapsed_ms, "trace_data": data or {}},
#     )


def _build_recommend_request(
    *, originator_entity_id: str, tags: dict, top_k: int
) -> RecommendRequest:
    missing = [k for k in _REQUIRED_TAG_KEYS_FOR_RECOMMEND if not tags.get(k)]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Parsed tags missing required fields for ranking: {missing}",
        )
    try:
        quantity = int(tags["qty"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid qty: {tags.get('qty')!r}") from exc
    if quantity <= 0:
        raise HTTPException(status_code=422, detail=f"qty must be > 0 (got {quantity})")

    return RecommendRequest(
        client_id=originator_entity_id,
        desk=str(tags["desk"]),
        structure=str(tags["structure"]),
        side=str(tags["side"]),
        venue=str(tags["venue"]),
        instrument_name=str(tags["instrument_name"]),
        quantity=quantity,
        top_k=top_k,
    )


@router.post("/match", response_model=VoiceInquiryMatchResponse)
async def match_voice_inquiry(
    request: Request,
    payload: VoiceInquiryMatchRequest,
    conn: asyncpg.Connection | None = Depends(get_optional_db_connection),
):
    t0 = time.monotonic()
    # trace: list[dict[str, Any]] = []

    settings = request.app.state.settings
    top_k = int(payload.top_k or settings.voice_inquiry_top_k)

    # ─────────────────────────────── Step 01 ───────────────────────────────
    # _trace_step(
    #     trace, t0, "01_receive_request",
    #     "Endpoint hit; preparing to process voice IOI",
    #     {
    #         "ioi_text_preview": payload.ioi_text[:200],
    #         "ioi_text_chars": len(payload.ioi_text),
    #         "originator_entity_id": payload.originator_entity_id,
    #         "top_k": top_k,
    #         "db_connection_present": conn is not None,
    #     },
    # )

    # ─────────────────────────────── Step 02 ───────────────────────────────
    service: IntelligenceService | None = await app_state.get("intelligence_service")
    if service is None:
        # _trace_step(trace, t0, "02_lookup_intelligence_service", "IntelligenceService not in app_state", {"present": False})
        raise HTTPException(status_code=503, detail="Intelligence service is not loaded")
    # _trace_step(
    #     trace, t0, "02_lookup_intelligence_service",
    #     "Loaded IntelligenceService from app_state",
    #     {
    #         "present": True,
    #         "configured_agent_ids": {
    #             "tagger": settings.daf_tagger_agent_id or "(empty)",
    #             "dossier": settings.daf_dossier_agent_id or "(empty)",
    #             "matcher": settings.daf_matcher_agent_id or "(empty)",
    #         },
    #     },
    # )

    # ─────────────────────────────── Step 03 ───────────────────────────────
    tagger_debug: dict[str, Any] = {}
    tags = await asyncio.to_thread(service.parse_ioi_tags, payload.ioi_text, debug=tagger_debug)
    if tags is None:
        # _trace_step(
        #     trace, t0, "03_parse_ioi_tags",
        #     "Tag parsing FAILED (agent error or non-JSON response)",
        #     tagger_debug,
        # )
        raise HTTPException(status_code=502, detail="Tag parsing failed")
    # _trace_step(
    #     trace, t0, "03_parse_ioi_tags",
    #     "Parsed IOI text into structured tags",
    #     {
    #         "agent_id": tagger_debug.get("agent_id"),
    #         "prompt": tagger_debug.get("prompt"),
    #         "raw_response": tagger_debug.get("raw_response"),
    #         "parsed_tags": tags,
    #         "missing_required_keys": [k for k in _REQUIRED_TAG_KEYS_FOR_RECOMMEND if not tags.get(k)],
    #     },
    # )

    # ─────────────────────────────── Step 04 ───────────────────────────────
    req = _build_recommend_request(
        originator_entity_id=payload.originator_entity_id, tags=tags, top_k=top_k
    )
    # _trace_step(
    #     trace, t0, "04_build_recommend_request",
    #     "Built RecommendRequest from parsed tags",
    #     {"request": req.model_dump()},
    # )

    # ─────────────────────────────── Step 05 ───────────────────────────────
    try:
        model_data = await app_state.get_model_data()
    except ValueError:
        # _trace_step(trace, t0, "05_load_model_data", "Champion model NOT LOADED", {"loaded": False})
        raise HTTPException(status_code=503, detail="Champion model is not loaded")

    live = model_data.get("live") or {}
    valid_ids = model_data.get("client_entity_ids")
    # _trace_step(
    #     trace, t0, "05_load_model_data",
    #     "Loaded champion model + artifacts from app_state",
    #     {
    #         "has_live": bool(live),
    #         "has_mats": "mats" in model_data,
    #         "has_heuristics": model_data.get("heuristics") is not None,
    #         "n_entities_in_model": len(live.get("entity_index", {})) if live else 0,
    #         "n_proxies_in_model": len(live.get("proxy_index", {})) if live else 0,
    #         "n_valid_client_entity_ids": len(valid_ids) if valid_ids is not None else None,
    #         "originator_in_valid_ids": (
    #             payload.originator_entity_id in valid_ids if valid_ids is not None else None
    #         ),
    #     },
    # )

    # ─────────────────────────────── Step 06 ───────────────────────────────
    try:
        proxy_str, top = await asyncio.to_thread(
            generate_ranked_candidates,
            req=req,
            model_data=model_data,
            settings=settings,
            top_k=top_k,
        )
    except ValueError as exc:
        # _trace_step(
        #     trace, t0, "06_generate_ranked_candidates",
        #     "Ranker raised ValueError (likely no proxy for the desk/structure/side combo)",
        #     {"error": str(exc)},
        # )
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # _trace_step(
    #     trace, t0, "06_generate_ranked_candidates",
    #     f"Ranked candidates via LightFM + heuristics rerank ({len(top)} returned)",
    #     {
    #         "proxy_str": proxy_str,
    #         "rerank_enabled": settings.rerank_enabled,
    #         "candidate_pool_size_setting": settings.rerank_candidate_pool_size,
    #         "top_count": len(top),
    #         "top": [
    #             {
    #                 "client_id": c.get("client_id"),
    #                 "affinity": c.get("affinity"),
    #                 "lightfm_normalized": c.get("lightfm_normalized"),
    #                 "size_score": c.get("size_score"),
    #                 "recency_score": c.get("recency_score"),
    #                 "time_score": c.get("time_score"),
    #                 "final_score": c.get("final_score"),
    #             }
    #             for c in top
    #         ],
    #     },
    # )

    if not top:
        # No candidates after ranking — still append the inquiry and return early.
        appended = await asyncio.to_thread(
            append_inquiry,
            settings.voice_inquiries_path,
            entity_id=payload.originator_entity_id,
            ioi_text=payload.ioi_text,
            tags=tags,
        )
        # _trace_step(
        #     trace, t0, "13_append_inquiry",
        #     "No candidates returned by ranker; appending originator IOI and returning empty match",
        #     {"appended": appended, "path": settings.voice_inquiries_path},
        # )
        # _trace_step(
        #     trace, t0, "14_respond",
        #     "Returning empty-candidate response",
        #     {"candidate_count": 0, "has_best_match": False},
        # )
        return VoiceInquiryMatchResponse(
            parsed_tags=tags,
            candidates=[],
            best_match=None,
            appended_inquiry=appended,
            # pipeline_trace=trace,
        )

    # ─────────────────────────────── Step 07 ───────────────────────────────
    candidate_ids = [str(c["client_id"]) for c in top]
    if conn is not None:
        name_map = await app_state.get_entity_names(candidate_ids, conn)
    else:
        name_map = {}
    # _trace_step(
    #     trace, t0, "07_resolve_entity_names",
    #     f"Resolved {len(name_map)} of {len(candidate_ids)} candidate names from DB cache",
    #     {
    #         "db_connection_present": conn is not None,
    #         "candidate_ids": candidate_ids,
    #         "resolved": name_map,
    #         "unresolved": [eid for eid in candidate_ids if eid not in name_map],
    #     },
    # )

    # ─────────────────────────────── Step 08 ───────────────────────────────
    heuristics = await app_state.get("heuristics")
    if heuristics is not None:
        heuristics_summary = {
            "present": True,
            "n_entities_with_venues": len(heuristics.active_venues_by_entity),
            "n_entities_with_hourly_ratios": len(heuristics.hourly_ratios_by_entity),
            "n_entity_instrument_size_profiles": len(heuristics.size_profile_by_entity_instrument),
            "n_entity_instrument_last_trade_dates": len(heuristics.last_trade_date_by_entity_instrument),
        }
    else:
        heuristics_summary = {"present": False}
    # _trace_step(
    #     trace, t0, "08_load_heuristics",
    #     "Loaded HeuristicsIndex from app_state",
    #     heuristics_summary,
    # )

    # ─────────────────────────────── Step 09 ───────────────────────────────
    dossier_cache = await asyncio.to_thread(load_dossiers, settings.voice_inquiry_dossiers_path)
    dossier_hits_in_topk = [eid for eid in candidate_ids if eid in dossier_cache]
    # _trace_step(
    #     trace, t0, "09_load_dossier_cache",
    #     f"Loaded dossier cache ({len(dossier_cache)} entries); {len(dossier_hits_in_topk)} hit(s) within top-K",
    #     {
    #         "path": settings.voice_inquiry_dossiers_path,
    #         "cache_size": len(dossier_cache),
    #         "cache_hit_entity_ids_in_topk": dossier_hits_in_topk,
    #         "cache_miss_entity_ids_in_topk": [eid for eid in candidate_ids if eid not in dossier_cache],
    #     },
    # )

    # ─────────────────────────────── Step 10 ───────────────────────────────
    all_inquiries = await asyncio.to_thread(load_inquiries, settings.voice_inquiries_path)
    inquiries_by_entity = group_inquiries_by_entity(all_inquiries, candidate_ids)
    # _trace_step(
    #     trace, t0, "10_load_all_inquiries",
    #     f"Loaded {len(all_inquiries)} demo inquiries from disk; filtered to candidates",
    #     {
    #         "path": settings.voice_inquiries_path,
    #         "total_inquiries_on_disk": len(all_inquiries),
    #         "inquiry_count_per_top_candidate": {
    #             eid: len(inquiries_by_entity.get(eid, [])) for eid in candidate_ids
    #         },
    #         "total_candidate_inquiries": sum(len(v) for v in inquiries_by_entity.values()),
    #     },
    # )

    # ─────────────────────────── Step 11 (per cand) ─────────────────────────
    candidate_entries: list[dict] = []
    for idx, cand in enumerate(top, start=1):
        eid = str(cand["client_id"])
        name = name_map.get(eid, "Unknown Entity")

        dossier_text = dossier_cache.get(eid)
        cache_hit = dossier_text is not None
        dossier_debug: dict[str, Any] = {}

        if not dossier_text:
            dossier_text = await asyncio.to_thread(
                service.build_entity_dossier,
                entity_id=eid,
                entity_name=name,
                heuristics=heuristics,
                instrument_name=tags.get("instrument_name"),
                debug=dossier_debug,
            )
            if dossier_text:
                await asyncio.to_thread(
                    upsert_dossier,
                    settings.voice_inquiry_dossiers_path,
                    entity_id=eid,
                    dossier_text=dossier_text,
                )

        entity_inquiries = inquiries_by_entity.get(eid, [])
        candidate_entries.append(
            {
                "entity_id": eid,
                "entity_name": name,
                "final_score": cand.get("final_score"),
                "lightfm_normalized": cand.get("lightfm_normalized"),
                "size_score": cand.get("size_score"),
                "recency_score": cand.get("recency_score"),
                "time_score": cand.get("time_score"),
                "dossier_text": dossier_text,
                "inquiries": entity_inquiries,
            }
        )

        # _trace_step(
        #     trace, t0, f"11_enrich_candidate_{idx}",
        #     (
        #         f"Candidate #{idx} ({name}): "
        #         + ("dossier CACHE HIT" if cache_hit else "dossier BUILT via agent")
        #         + f"; {len(entity_inquiries)} open inquiries on file"
        #     ),
        #     {
        #         "position": idx,
        #         "entity_id": eid,
        #         "entity_name": name,
        #         "final_score": cand.get("final_score"),
        #         "score_components": {
        #             "lightfm_normalized": cand.get("lightfm_normalized"),
        #             "size_score": cand.get("size_score"),
        #             "recency_score": cand.get("recency_score"),
        #             "time_score": cand.get("time_score"),
        #         },
        #         "dossier_cache_hit": cache_hit,
        #         "dossier_chars": len(dossier_text) if dossier_text else 0,
        #         "dossier_text": dossier_text,
        #         "dossier_agent_debug": dossier_debug if not cache_hit else None,
        #         "open_inquiry_count": len(entity_inquiries),
        #         "open_inquiry_ids": [inq.get("inquiry_id") for inq in entity_inquiries],
        #     },
        # )

    # ─────────────────────────────── Step 12 ───────────────────────────────
    matcher_debug: dict[str, Any] = {}
    best_match = await asyncio.to_thread(
        service.match_voice_inquiry,
        originator_ioi_text=payload.ioi_text,
        originator_tags=tags,
        candidate_entries=candidate_entries,
        debug=matcher_debug,
    )
    # _trace_step(
    #     trace, t0, "12_match_voice_inquiry",
    #     (
    #         "Matcher agent returned a selection"
    #         if best_match is not None
    #         else "Matcher agent returned None (failure)"
    #     ),
    #     {
    #         "agent_id": matcher_debug.get("agent_id"),
    #         "candidate_inquiry_counts": matcher_debug.get("candidate_inquiry_counts"),
    #         "total_candidate_inquiries": matcher_debug.get("total_candidate_inquiries"),
    #         "prompt": matcher_debug.get("prompt"),
    #         "raw_response": matcher_debug.get("raw_response"),
    #         "parsed": matcher_debug.get("parsed"),
    #         "error": matcher_debug.get("error"),
    #     },
    # )

    # ─── Resolve the full matched inquiry record from the candidate pool ───
    matched_inquiry: dict[str, Any] | None = None
    if best_match:
        target_inq_id = best_match.get("best_match_inquiry_id")
        target_eid = best_match.get("best_match_entity_id")
        if target_inq_id:
            for cand in candidate_entries:
                if target_eid and cand.get("entity_id") != target_eid:
                    continue
                for inq in (cand.get("inquiries") or []):
                    if inq.get("inquiry_id") == target_inq_id:
                        matched_inquiry = inq
                        break
                if matched_inquiry is not None:
                    break

    # ─────────────────────────────── Step 13 ───────────────────────────────
    appended = await asyncio.to_thread(
        append_inquiry,
        settings.voice_inquiries_path,
        entity_id=payload.originator_entity_id,
        ioi_text=payload.ioi_text,
        tags=tags,
    )
    # _trace_step(
    #     trace, t0, "13_append_inquiry",
    #     "Appended new originator inquiry to demo inquiries file",
    #     {
    #         "path": settings.voice_inquiries_path,
    #         "appended_inquiry": appended,
    #     },
    # )

    # ─────────────────────────────── Step 14 ───────────────────────────────
    # _trace_step(
    #     trace, t0, "14_respond",
    #     "Returning response to client",
    #     {
    #         "candidate_count": len(candidate_entries),
    #         "has_best_match": best_match is not None,
    #         "best_match_percent": (best_match or {}).get("match_percent"),
    #         "best_match_inquiry_id": (best_match or {}).get("best_match_inquiry_id"),
    #         "best_match_entity_id": (best_match or {}).get("best_match_entity_id"),
    #     },
    # )

    return VoiceInquiryMatchResponse(
        parsed_tags=tags,
        candidates=candidate_entries,
        best_match=best_match,
        matched_inquiry=matched_inquiry,
        appended_inquiry=appended,
        # pipeline_trace=trace,
    )


@router.post("/tags", response_model=dict)
async def parse_tags_only(
    request: Request,
    payload: VoiceInquiryMatchRequest,
):
    """
    Endpoint to return tag parsing in isolation.
    """
    service: IntelligenceService | None = await app_state.get("intelligence_service")
    if service is None:
        raise HTTPException(status_code=503, detail="Intelligence service is not loaded")

    tagger_debug: dict[str, Any] = {}
    tags = await asyncio.to_thread(service.parse_ioi_tags, payload.ioi_text, debug=tagger_debug)
    if tags is None:
        raise HTTPException(status_code=502, detail="Tag parsing failed")

    return {
        "parsed_tags": tags,
        "agent_id": tagger_debug.get("agent_id"),
        "prompt": tagger_debug.get("prompt"),
        "raw_response": tagger_debug.get("raw_response"),
        "debug": tagger_debug,
    }