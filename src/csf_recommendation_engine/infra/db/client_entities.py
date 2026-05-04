from __future__ import annotations

import asyncpg


async def fetch_client_entity_names(conn: asyncpg.Connection) -> dict[str, str]:
    rows = await conn.fetch(
        "SELECT entity_id, entity_name FROM client_entities"
    )
    return {str(row["entity_id"]): row["entity_name"] for row in rows}

async def fetch_entity_names_by_ids(conn: asyncpg.Connection, entity_ids: list[str]) -> dict[str, str]:
    if not entity_ids:
        return {}
    rows = await conn.fetch(
        "SELECT entity_id, entity_name FROM client_entities WHERE entity_id = ANY($1)",
        entity_ids
    )
    return {str(row["entity_id"]): row["entity_name"] for row in rows}