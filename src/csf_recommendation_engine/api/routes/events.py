"""Engagement-event ingest endpoint (plan §15.5, Step 0.7).

UI services POST broker engagement events here; the engine appends a
row to ``ai_engagement_events``. No-auth in v1 local dev; production
auth shape is deferred to deployment.
"""

from __future__ import annotations

import logging

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request

from csf_recommendation_engine.core.observability import M_ENGAGEMENT_EVENT_INGEST
from csf_recommendation_engine.domain.schemas import (
    EngagementEventRequest,
    EngagementEventResponse,
)
from csf_recommendation_engine.infra.db.ai_engagement_events import insert_event
from csf_recommendation_engine.infra.db.pool import get_db_connection


router = APIRouter(prefix="/events", tags=["events"])

logger = logging.getLogger(__name__)


@router.post("/engagement", response_model=EngagementEventResponse, status_code=201)
async def post_engagement_event(
    request: Request,
    payload: EngagementEventRequest,
    conn: asyncpg.Connection = Depends(get_db_connection),
) -> EngagementEventResponse:
    metrics = getattr(request.app.state, "metrics", None)
    request_id = getattr(request.state, "request_id", None)

    try:
        event_id = await insert_event(
            conn,
            event_type=payload.event_type,
            serve_id=payload.serve_id,
            match_id=payload.match_id,
            user_id=payload.user_id,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"referenced row not found: {exc.detail or exc}",
        ) from exc
    except asyncpg.CheckViolationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if metrics is not None:
        metrics.increment(M_ENGAGEMENT_EVENT_INGEST)

    logger.info(
        "engagement_event.ingested",
        extra={
            "event_id": str(event_id),
            "event_type": payload.event_type,
            "serve_id": str(payload.serve_id) if payload.serve_id else None,
            "match_id": str(payload.match_id) if payload.match_id else None,
            "request_id": request_id,
        },
    )
    return EngagementEventResponse(event_id=event_id)
