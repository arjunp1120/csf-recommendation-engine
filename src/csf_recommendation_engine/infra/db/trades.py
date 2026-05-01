from __future__ import annotations

import logging
from typing import Sequence
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)


async def fetch_working_trades(
    conn: asyncpg.Connection, *, min_quantity: int = 5
) -> list[dict]:
    """Load trades where status = 'Working', quantity > min_quantity, and side in (BUY, SELL).

    Returns a list of dicts with the fields needed by the nightly recommendation pipeline.
    """
    rows = await conn.fetch(
        """
        SELECT
            trade_id,
            trade_date,
            trade_time,
            instrument_name,
            structure::text   AS structure,
            side::text        AS side,
            quantity,
            venue,
            entity_id,
            desk_type::text   AS desk_type
        FROM public.trades

        WHERE quantity > $1
          AND side::text IN ('BUY', 'SELL')
        ORDER BY updated_at DESC
        """,
        min_quantity,
    )
    return [dict(r) for r in rows]


async def resolve_entity_names(
    conn: asyncpg.Connection, entity_ids: Sequence[str | UUID]
) -> dict[str, str]:
    """Resolve entity_id -> entity_name for a batch of IDs."""
    if not entity_ids:
        return {}
    str_ids = [str(eid) for eid in entity_ids]
    rows = await conn.fetch(
        "SELECT entity_id, entity_name FROM client_entities WHERE entity_id = ANY($1::uuid[])",
        str_ids,
    )
    return {str(row["entity_id"]): row["entity_name"] for row in rows}


async def insert_ai_recommendations(
    conn: asyncpg.Connection, rows: list[dict]
) -> int:
    """Insert rows into public.ai_recommendations. Returns count of inserted rows."""
    if not rows:
        return 0
    await conn.executemany(
        """
        INSERT INTO public.ai_recommendations
            (entity_id, recommendation_type, product, description, details, created_by_ai_agent)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        [
            (
                row["entity_id"],
                row["recommendation_type"],
                row["product"],
                row["description"],
                row.get("details"),
                row.get("created_by_ai_agent", "Caddie AI"),
            )
            for row in rows
        ],
    )
    return len(rows)


async def insert_cross_block_matches(
    conn: asyncpg.Connection, rows: list[dict]
) -> int:
    """Insert rows into public.cross_block_matches. Returns count of inserted rows."""
    if not rows:
        return 0
    await conn.executemany(
        """
        INSERT INTO public.cross_block_matches
            (product_name, match_percentage, buyer_entity_id, buyer_side,
             buyer_lots, seller_entity_id, seller_side, seller_lots, description)
        VALUES ($1, $2, $3, $4::trade_side, $5, $6, $7::trade_side, $8, $9)
        """,
        [
            (
                row["product_name"],
                row["match_percentage"],
                row["buyer_entity_id"],
                row["buyer_side"],
                row["buyer_lots"],
                row["seller_entity_id"],
                row["seller_side"],
                row["seller_lots"],
                row.get("description"),
            )
            for row in rows
        ],
    )
    return len(rows)


async def fetch_recent_ai_recommendations(
    conn: asyncpg.Connection, *, agent_name: str
) -> list[dict]:
    """Fetch ai_recommendations created in the last 24 hours by a specific agent.

    Returns only the logical-key fields needed for deduplication.
    """
    rows = await conn.fetch(
        """
        SELECT entity_id::text, product, description
        FROM public.ai_recommendations
        WHERE created_by_ai_agent = $1
          AND created_at >= NOW() - INTERVAL '24 hours'
        """,
        agent_name,
    )
    return [dict(r) for r in rows]


async def fetch_recent_cross_block_matches(
    conn: asyncpg.Connection,
) -> list[dict]:
    """Fetch cross_block_matches created in the last 24 hours.

    Returns only the logical-key fields needed for deduplication.
    """
    rows = await conn.fetch(
        """
        SELECT buyer_entity_id::text, seller_entity_id::text, product_name
        FROM public.cross_block_matches
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        """,
    )
    return [dict(r) for r in rows]
