import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, Depends
import asyncpg

from csf_recommendation_engine.core.config import get_settings
from csf_recommendation_engine.domain.schemas import RecommendRequest, RecommendResponse
from csf_recommendation_engine.core.state import app_state
from csf_recommendation_engine.infra.db.pool import get_optional_db_connection
from csf_recommendation_engine.infra.db.client_entities import fetch_entity_names_by_ids
from csf_recommendation_engine.domain.recommendation_engine import generate_ranked_candidates

router = APIRouter(tags=["recommendations"])

logger = logging.getLogger(__name__)

    
@router.post("/recommend", response_model=RecommendResponse)
async def recommend(
    request: Request,
    payload: RecommendRequest,
    conn: asyncpg.Connection | None = Depends(get_optional_db_connection),
):
    try:
        try:
            model_data = await app_state.get_model_data()
        except ValueError:
            raise HTTPException(status_code=503, detail="Champion model is not loaded")

        settings = request.app.state.settings
        try:
            proxy_str, top = generate_ranked_candidates(
                req=payload,
                model_data=model_data,
                settings=settings,
                top_k=payload.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if conn is not None and top:
            name_map = await app_state.get_entity_names([res["client_id"] for res in top], conn)
            for res in top:
                res["entity_name"] = name_map.get(res["client_id"], "Unknown Entity")
        else:
            for res in top:
                res["entity_name"] = "Unknown Entity"

        return RecommendResponse(counterparties=top, queried_proxy=proxy_str)
    except HTTPException as e:
        raise e
