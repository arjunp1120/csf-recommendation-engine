"""Nightly pipeline manager

Currently: builds and reloads heuristics artifacts (neightly_heuristics.py)
TODO: extraction for shadow model (nightly_shadow_model.py)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path

from csf_recommendation_engine.core.config import get_settings
from csf_recommendation_engine.core.startup import preload_heuristics_state
from csf_recommendation_engine.infra.db.pool import get_db_pool
from csf_recommendation_engine.jobs.nightly_heuristics import run_nightly_heuristics_build

logger = logging.getLogger(__name__)

nightly_run_lock = asyncio.Lock()
ADVISORY_LOCK_ID = 424242


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


async def run_full_nightly_pipeline() -> None:
    """Nightly pipeline: builds and reloads heuristics artifacts.

    Recommendation generation and cross-block matching have been extracted
    into the rec refresh pipeline (rec_refresh_pipeline.py) which runs
    every 15 minutes.
    """
    settings = get_settings()
    if not settings.nightly_pipeline_enabled:
        logger.info("Nightly pipeline disabled via config")
        return

    async with nightly_run_lock:
        pool = get_db_pool()
        async with pool.acquire() as conn:
            acquired = await _acquire_advisory_lock(conn)
            if not acquired:
                logger.info("Nightly pipeline skipped because advisory lock is already held")
                return

            try:
                logger.info("Starting nightly pipeline")

                # -------------------------------------------------------
                # Step 1: Build heuristics parquet files
                # -------------------------------------------------------
                await run_nightly_heuristics_build()
                logger.info("Nightly heuristics build completed")

                # -------------------------------------------------------
                # Step 2: Reload heuristics index
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

            finally:
                with suppress(Exception):
                    await _release_advisory_lock(conn)
                logger.info("Nightly pipeline finished")
