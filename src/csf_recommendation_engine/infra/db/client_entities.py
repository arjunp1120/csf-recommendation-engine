from __future__ import annotations

import asyncpg


async def fetch_client_entity_names(conn: asyncpg.Connection) -> dict[str, str]:
    rows = await conn.fetch(
        "SELECT entity_id, entity_name FROM client_entities"
    )
    return {str(row["entity_id"]): row["entity_name"] for row in rows}
