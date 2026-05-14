import logging
import pickle
from pathlib import Path

from csf_recommendation_engine.core.config import get_settings

from csf_recommendation_engine.domain.heuristics_index import HeuristicsIndex, load_heuristics_from_parquet
from csf_recommendation_engine.infra.db.client_entities import fetch_client_entity_names
from csf_recommendation_engine.infra.db.pool import get_db_pool
from csf_recommendation_engine.core.state import app_state

logger = logging.getLogger(__name__)

async def preload_champion_state(model_path: Path, artifact_path: Path) -> None:
    logger.info("Preloading champion model and artifacts into memory")
    await app_state.load_champion_state(model_path, artifact_path)
    logger.info("Preloaded champion model and artifacts into memory")


async def preload_heuristics_state(
    *,
    entity_features_path: Path,
    instrument_features_path: Path,
    require: bool = False,
) -> None:
    logger.info("Preloading heuristics artifacts into memory")
    await app_state.load_heuristics_state(entity_features_path, instrument_features_path, require)
    logger.info("Preloaded heuristics artifacts into memory")


async def preload_client_entity_catalog() -> None:
    pool = get_db_pool()
    async with pool.acquire() as conn:
        logger.info("Preloading client entity catalog into memory")
        await app_state.load_client_entity_catalog(conn)
    logger.info("Preloaded client entity catalog into memory")


async def preload_product_resolver() -> None:
    """Load the ``instrument_products`` table into an in-memory
    :class:`ProductResolver` and stash it under
    ``app_state.get('product_resolver')`` (plan Step 0.9).

    Safe to call before the table has been seeded — the resolver simply
    resolves nothing until the table is populated by Step 0.10.
    """
    from csf_recommendation_engine.domain.products import ProductResolver

    pool = get_db_pool()
    async with pool.acquire() as conn:
        resolver = await ProductResolver.load(conn)
    await app_state.set("product_resolver", resolver)
    logger.info(
        "ProductResolver preloaded",
        extra=resolver.size,
    )


async def preload_intelligence_service() -> None:
    """Instantiate the LLM intelligence service and store it in app state.

    Only runs if ``LLM_ENABLED`` is ``True``. The rebuilt
    ``IntelligenceService`` (plan Step 0.6) is an async HTTP client that
    holds a single ``httpx.AsyncClient`` shared across all swarm calls;
    instantiate once at startup, close via ``aclose()`` on shutdown.
    Agents/swarms themselves are configured externally on the DAF
    platform and addressed by id (see ``DAF_*_SWARM_ID`` settings).
    """
    settings = get_settings()
    if not settings.llm_enabled:
        logger.info("LLM intelligence layer disabled via config")
        return

    from csf_recommendation_engine.domain.intelligence.intelligence_service import (
        IntelligenceService,
    )

    service = IntelligenceService(settings)
    await app_state.set("intelligence_service", service)
    logger.info("Intelligence service initialized")
