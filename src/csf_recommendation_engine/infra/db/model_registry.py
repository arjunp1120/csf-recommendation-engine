from dataclasses import dataclass
from datetime import datetime

import asyncpg


@dataclass
class ModelRegistryRow:
    model_id: str
    created_at: datetime
    rmse_score: float | None
    warp_score: float | None
    blob_storage_path: str
    status: str


FETCH_ACTIVE_CHAMPION_SQL = """
SELECT model_id, created_at, rmse_score, warp_score, blob_storage_path, status
FROM csf_model_registry
WHERE status = 'active_production'
ORDER BY created_at DESC
LIMIT 1
"""


async def fetch_active_champion(conn: asyncpg.Connection) -> ModelRegistryRow | None:
    row = await conn.fetchrow(FETCH_ACTIVE_CHAMPION_SQL)
    if row is None:
        return None
    return ModelRegistryRow(**dict(row))


INSERT_PENDING_MODEL_SQL = """
INSERT INTO csf_model_registry (model_id, rmse_score, warp_score, blob_storage_path, status)
VALUES ($1, $2, $3, $4, 'pending')
"""


async def insert_pending_model(
    conn: asyncpg.Connection,
    model_id: str,
    blob_storage_path: str,
    rmse_score: float | None = None,
    warp_score: float | None = None,
) -> None:
    await conn.execute(
        INSERT_PENDING_MODEL_SQL,
        model_id,
        rmse_score,
        warp_score,
        blob_storage_path,
    )


ARCHIVE_CURRENT_CHAMPION_SQL = """
UPDATE csf_model_registry
SET status = 'archived'
WHERE status = 'active_production'
"""


ACTIVATE_MODEL_SQL = """
UPDATE csf_model_registry
SET status = 'active_production'
WHERE model_id = $1
"""


async def promote_model_to_active(conn: asyncpg.Connection, model_id: str) -> None:
    # IMPORTANT TODO CHECKPOINT:
    # Confirm final status lifecycle decision before using this function in API mutation logic.
    async with conn.transaction():
        await conn.execute(ARCHIVE_CURRENT_CHAMPION_SQL)
        await conn.execute(ACTIVATE_MODEL_SQL, model_id)
