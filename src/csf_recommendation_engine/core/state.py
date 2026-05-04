"""
Application state container.
"""
import pickle
import logging
import asyncpg
from typing import Any
from pathlib import Path

from csf_recommendation_engine.domain.heuristics_index import HeuristicsIndex, load_heuristics_from_parquet
from csf_recommendation_engine.infra.db.client_entities import fetch_client_entity_names, fetch_entity_names_by_ids

logger = logging.getLogger(__name__)

class AppState:
    def __init__(self):
        self.model_data = {}
    
    async def get(self, key: str, default: Any = None) -> Any:
        return self.model_data.get(key, default)
    
    async def set(self, key: str, value: Any) -> None:
        logger.info(f"Setting key {key}", extra={"value_type": type(value)})
        self.model_data[key] = value
    
    async def get_model_data(self) -> dict[str, Any]:
        return self.model_data

    async def load_champion_state(self, model_path: Path, artifact_path: Path) -> None:
        logger.info("Loading champion model and artifacts")
        with open(model_path, "rb") as f:
            live = pickle.load(f)
        with open(artifact_path, "rb") as f:
            mats = pickle.load(f)
        # Reverse index for fast UUID lookup
        rev_entity = {v: k for k, v in live["entity_index"].items()}
        await self.set(key="live", value=live) # live/champion model
        await self.set(key="mats", value=mats) # model training artifacts
        await self.set(key="rev_entity", value=rev_entity) # model-known entities
        logger.info("Loaded champion model and artifacts")
        
        
    async def load_heuristics_state(
        self,
        entity_features_path: Path,
        instrument_features_path: Path,
        require: bool = False,
    ) -> None:
        logger.info("Loading heuristics artifacts")
        try:
            heuristics = load_heuristics_from_parquet(
                entity_features_path=entity_features_path,
                instrument_features_path=instrument_features_path,
            )
            await self.set(key="heuristics", value=heuristics)
            logger.info(
                "Loaded heuristics artifacts",
                extra={
                    "entity_features_path": str(entity_features_path),
                    "instrument_features_path": str(instrument_features_path),
                },
            )
        except FileNotFoundError as e:
            await self.set(key="heuristics", value=None)
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
            await self.set(key="heuristics", value=None)
            logger.exception("Failed to load heuristics artifacts")
            if require:
                raise
    
    async def load_client_entity_catalog(self, conn: asyncpg.Connection) -> None:
        entity_names = await fetch_client_entity_names(conn)
        client_entity_ids = set(entity_names.keys())
        await self.set(key="client_entity_ids", value=client_entity_ids)
        await self.set(key="client_entity_names", value=entity_names)
        logger.info("Loaded client entity catalog")

    async def get_entity_names(self, entity_ids: list[str], conn: asyncpg.Connection) -> dict[str, str]:
        cached_names = await self.get("client_entity_names", {})
        resolved = {client_id: cached_names[client_id] for client_id in entity_ids if client_id in cached_names}
        missing_ids = [client_id for client_id in entity_ids if client_id not in resolved]
        if not missing_ids:
            return resolved
        
        rows = await fetch_entity_names_by_ids(conn, missing_ids)
        resolved.update(rows)

        await self._update_entity_names_and_ids(list(resolved.keys()), resolved)

        return resolved

    async def _update_entity_names_and_ids(self, entity_ids: list[str], entity_names: dict[str, str]) -> None:
        current_names = await self.get("client_entity_names", {})
        new_names = {k: v for k, v in entity_names.items() if k not in current_names}
        current_names.update(new_names)
        await self.set("client_entity_names", current_names)
        
        current_ids = await self.get("client_entity_ids", set())
        new_ids = {k for k in entity_ids if k not in current_ids}
        current_ids.update(new_ids)
        await self.set("client_entity_ids", current_ids)


app_state = AppState()

