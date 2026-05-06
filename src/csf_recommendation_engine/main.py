from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from fastapi import FastAPI
import uvicorn
from pathlib import Path
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from csf_recommendation_engine.api.middleware.request_id import RequestIdMiddleware
from csf_recommendation_engine.api.routes import admin, health, internal, recommend
from csf_recommendation_engine.core.config import get_settings
from csf_recommendation_engine.core.logging import configure_logging


from csf_recommendation_engine.core.startup import (
    preload_champion_state,
    preload_heuristics_state,
    preload_client_entity_catalog,
    preload_intelligence_service,
)
from csf_recommendation_engine.infra.db.pool import init_db_pool, close_db_pool
from csf_recommendation_engine.jobs.nightly_pipeline import run_full_nightly_pipeline
from csf_recommendation_engine.jobs.rec_refresh_pipeline import run_rec_refresh_pipeline

logger = logging.getLogger(__name__)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    configure_logging(debug=settings.app_debug)

    app.state.settings = settings
    # app.state.cached_item_features = None
    # app.state.champion_model = None
    # app.state.shadow_model = None
    # app.state.shadow_model = None

    if settings.database_url:
        await init_db_pool(settings.database_url)
    else:
        logger.warning("DATABASE_URL is empty; DB pool will not be initialized")

    try:
        await preload_champion_state(
            model_path=Path(settings.champion_model_path),
            artifact_path=Path(settings.model_artifacts_path),
        )
    except Exception:
        logger.exception(
            "Failed to preload champion model; /recommend will return 503 until fixed",
            extra={
                "champion_model_path": settings.champion_model_path,
                "model_artifacts_path": settings.model_artifacts_path,
            },
        )

    try:
        entity_path = Path(settings.heuristics_artifact_dir) / settings.heuristics_entity_features_latest_filename
        inst_path = (
            Path(settings.heuristics_artifact_dir)
            / settings.heuristics_instrument_features_latest_filename
        )
        await preload_heuristics_state(
            entity_features_path=entity_path,
            instrument_features_path=inst_path,
            require=settings.rerank_require_heuristics,
        )
    except Exception:
        logger.exception(
            "Heuristics preload failed; reranking will be disabled until fixed",
            extra={"heuristics_artifact_dir": settings.heuristics_artifact_dir},
        )

    try:
        await preload_client_entity_catalog()
    except Exception:
        logger.exception("Failed to preload client entity catalog; candidate filtering may be incomplete")

    try:
        await preload_intelligence_service()
    except Exception:
        logger.exception("Failed to preload intelligence service; LLM layer will be disabled")

    scheduler = AsyncIOScheduler(timezone=ZoneInfo(settings.nightly_schedule_timezone))
    app.state.scheduler = scheduler
    if settings.nightly_pipeline_enabled:
        scheduler.add_job(
            run_full_nightly_pipeline,
            CronTrigger.from_crontab(settings.nightly_schedule_cron, timezone=ZoneInfo(settings.nightly_schedule_timezone)),
            id="csf-nightly-pipeline",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
    if settings.rec_refresh_enabled:
        
        scheduler.add_job(
            run_rec_refresh_pipeline,
            IntervalTrigger(minutes=15),
            id="csf-rec-refresh",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            next_run_time=datetime.now(ZoneInfo(settings.nightly_schedule_timezone)),
        )

    if settings.nightly_pipeline_enabled or settings.rec_refresh_enabled:
        scheduler.start()
        logger.info(
            "Scheduler started",
            extra={
                "nightly_cron": settings.nightly_schedule_cron if settings.nightly_pipeline_enabled else "disabled",
                "rec_refresh": "every 15 min" if settings.rec_refresh_enabled else "disabled",
                "timezone": settings.nightly_schedule_timezone,
            },
        )

    yield

    # Shutdown tasks
    scheduler = getattr(app.state, "scheduler", None)
    if scheduler is not None:
        scheduler.shutdown(wait=False)
    await close_db_pool()


def create_app() -> FastAPI:
    app = FastAPI(
        title="CSF Recommendation Engine",
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)

    app.include_router(health.router)
    app.include_router(recommend.router)
    app.include_router(internal.router)
    app.include_router(admin.router)
    return app


app = create_app()