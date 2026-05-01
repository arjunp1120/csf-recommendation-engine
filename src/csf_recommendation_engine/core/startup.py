import logging
import pickle
from pathlib import Path

from csf_recommendation_engine.domain.heuristics_index import HeuristicsIndex, load_heuristics_from_parquet
from csf_recommendation_engine.infra.db.client_entities import fetch_client_entity_names
from csf_recommendation_engine.infra.db.pool import get_db_pool

logger = logging.getLogger(__name__)

model_data = {}

async def preload_champion_state(model_path: Path, artifact_path: Path) -> None:
    logger.info("Preloading champion model and artifacts into memory")
    global model_data
    # Load the artifacts we generated in the previous steps
    # champion_model_path = model_path
    # model_artifacts_path = artifact_path
    with open(model_path, "rb") as f:
        model_data["live"] = pickle.load(f)
    with open(artifact_path, "rb") as f:
        model_data["mats"] = pickle.load(f)
    
    # Reverse index for fast UUID lookup
    model_data["rev_entity"] = {v: k for k, v in model_data["live"]["entity_index"].items()}
    logger.info("Preloaded champion model and artifacts into memory")


async def preload_heuristics_state(
    *,
    entity_features_path: Path,
    instrument_features_path: Path,
    require: bool = False,
) -> None:
    global model_data
    try:
        heuristics = load_heuristics_from_parquet(
            entity_features_path=entity_features_path,
            instrument_features_path=instrument_features_path,
        )
        model_data["heuristics"] = heuristics
        logger.info(
            "Preloaded heuristics artifacts into memory",
            extra={
                "entity_features_path": str(entity_features_path),
                "instrument_features_path": str(instrument_features_path),
            },
        )
    except FileNotFoundError as e:
        model_data["heuristics"] = None
        logger.warning(
            "Heuristics artifacts not found; reranking will be disabled",
            extra={
                "entity_features_path": str(entity_features_path),
                "instrument_features_path": str(instrument_features_path),
            },
        )
        if require:
            raise
    except Exception:
        model_data["heuristics"] = None
        logger.exception("Failed to preload heuristics artifacts")
        if require:
            raise


async def preload_client_entity_catalog() -> None:
    global model_data
    pool = get_db_pool()
    async with pool.acquire() as conn:
        entity_names = await fetch_client_entity_names(conn)
    model_data["client_entity_names"] = entity_names
    model_data["client_entity_ids"] = set(entity_names.keys())
    logger.info("Preloaded client entity catalog into memory", extra={"entity_count": len(entity_names)})
    
async def get_model_data() -> dict:
    global model_data
    if not model_data:
        raise ValueError("Model data has not been preloaded")
    return model_data