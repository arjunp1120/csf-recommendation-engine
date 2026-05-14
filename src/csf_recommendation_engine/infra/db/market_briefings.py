"""DB access for ``market_briefings`` (plan §8.3).

Per-product market briefings produced by the S3 Market Reader swarm.
Refreshed every 15 min during market hours. Each call appends a new
row; "latest" is selected by ``generated_at DESC`` via the
``(product_name, generated_at)`` index.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg


SELECT_COLUMNS = """
    briefing_id::text       AS briefing_id,
    product_name,
    regime_label,
    recent_move_summary,
    liquidity_assessment,
    broker_book_posture,
    generated_at,
    swarm_packet_hash,
    packet_json
"""


async def upsert_briefing(
    conn: asyncpg.Connection,
    *,
    product_name: str,
    regime_label: str | None = None,
    recent_move_summary: str | None = None,
    liquidity_assessment: str | None = None,
    broker_book_posture: str | None = None,
    swarm_packet_hash: str | None = None,
    packet_json: dict[str, Any] | None = None,
) -> UUID:
    """Append a new briefing row; return ``briefing_id``.

    Like :func:`entity_dossiers.upsert_dossier`, this is append-only —
    "latest" is the row with the highest ``generated_at`` for the
    product. The function is named ``upsert_briefing`` per plan §0.8.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO public.market_briefings
            (product_name, regime_label, recent_move_summary,
             liquidity_assessment, broker_book_posture,
             swarm_packet_hash, packet_json)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
        RETURNING briefing_id
        """,
        product_name,
        regime_label,
        recent_move_summary,
        liquidity_assessment,
        broker_book_posture,
        swarm_packet_hash,
        json.dumps(packet_json) if packet_json is not None else None,
    )
    return row["briefing_id"]


async def fetch_latest_briefing(
    conn: asyncpg.Connection, product_name: str
) -> dict[str, Any] | None:
    """Return the most recent briefing for ``product_name`` or ``None``."""
    row = await conn.fetchrow(
        f"""
        SELECT {SELECT_COLUMNS}
        FROM public.market_briefings
        WHERE product_name = $1
        ORDER BY generated_at DESC
        LIMIT 1
        """,
        product_name,
    )
    return dict(row) if row is not None else None
