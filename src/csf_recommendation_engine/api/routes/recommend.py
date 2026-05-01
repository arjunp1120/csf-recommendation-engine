import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, Depends
import asyncpg

from csf_recommendation_engine.core.config import get_settings
from csf_recommendation_engine.domain.schemas import RecommendRequest, RecommendResponse
from csf_recommendation_engine.core.startup import get_model_data
from csf_recommendation_engine.infra.db.pool import get_optional_db_connection
from csf_recommendation_engine.domain.recommendation_engine import generate_ranked_candidates

router = APIRouter(tags=["recommendations"])

logger = logging.getLogger(__name__)

#TODO: move this away to an interface in my opiinion:
async def get_entity_names(client_ids: list[str], conn: asyncpg.Connection) -> dict[str, str]:
    model_data = await get_model_data()
    cached_names = model_data.get("client_entity_names", {})
    resolved = {client_id: cached_names[client_id] for client_id in client_ids if client_id in cached_names}
    missing_ids = [client_id for client_id in client_ids if client_id not in resolved]
    if not missing_ids:
        return resolved

    rows = await conn.fetch(
        "SELECT entity_id, entity_name FROM client_entities WHERE entity_id = ANY($1)",
        missing_ids
    )
    resolved.update({str(row["entity_id"]): row["entity_name"] for row in rows})
    return resolved
    
    


#TODO: make model a singleton and load at startup, not per request
@router.post("/recommend", response_model=RecommendResponse)
async def recommend(
    req: RecommendRequest,
    conn: asyncpg.Connection | None = Depends(get_optional_db_connection),
):
    try:
        try:
            model_data = await get_model_data()
        except ValueError:
            raise HTTPException(status_code=503, detail="Champion model is not loaded")

        settings = get_settings()
        try:
            proxy_str, top = generate_ranked_candidates(
                req=req,
                model_data=model_data,
                settings=settings,
                top_k=req.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if conn is not None and top:
            name_map = await get_entity_names([res["client_id"] for res in top], conn)
            for res in top:
                res["entity_name"] = name_map.get(res["client_id"], "Unknown Entity")
        else:
            for res in top:
                res["entity_name"] = "Unknown Entity"

        return RecommendResponse(counterparties=top, queried_proxy=proxy_str)
    except HTTPException as e:
        raise e
