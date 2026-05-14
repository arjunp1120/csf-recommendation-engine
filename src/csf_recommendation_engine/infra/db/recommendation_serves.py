"""DB access for ``recommendation_serves`` (plan §15.4).

One row per served ``/recommend`` (form-driven) or ``/coach/draft``
response. Carries ``packet_hash`` + ``packet_json`` for exact replay
and ``variant_id`` for A/B routing. ``/recommend/ioi`` is parse-only
and writes NOTHING here.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg


SELECT_COLUMNS = """
    serve_id::text          AS serve_id,
    request_type,
    originator_entity_id::text AS originator_entity_id,
    intent_tags,
    candidates,
    packet_hash,
    packet_json,
    variant_id,
    swarm_response,
    served_at,
    served_to_user_id::text AS served_to_user_id
"""


async def insert_serve(
    conn: asyncpg.Connection,
    *,
    request_type: str,
    packet_hash: str,
    packet_json: dict[str, Any],
    candidates: list[dict[str, Any]],
    originator_entity_id: UUID | str | None = None,
    intent_tags: dict[str, Any] | None = None,
    variant_id: str | None = None,
    swarm_response: dict[str, Any] | None = None,
    served_to_user_id: UUID | str | None = None,
) -> UUID:
    """Insert one served response; return the generated ``serve_id``.

    ``candidates`` is the ordered list of per-candidate dicts the
    response carried (entity_id + scores + rationale). ``packet_json``
    is the full Intelligence Packet that produced the response, used
    later for replay and correlative attribution.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO public.recommendation_serves
            (request_type, originator_entity_id, intent_tags, candidates,
             packet_hash, packet_json, variant_id, swarm_response, served_to_user_id)
        VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6::jsonb, $7, $8::jsonb, $9)
        RETURNING serve_id
        """,
        request_type,
        originator_entity_id,
        json.dumps(intent_tags) if intent_tags is not None else None,
        json.dumps(candidates),
        packet_hash,
        json.dumps(packet_json),
        variant_id,
        json.dumps(swarm_response) if swarm_response is not None else None,
        served_to_user_id,
    )
    return row["serve_id"]


async def fetch_serve(
    conn: asyncpg.Connection, serve_id: UUID | str
) -> dict[str, Any] | None:
    """Return one serve row by ``serve_id`` or ``None``."""
    row = await conn.fetchrow(
        f"SELECT {SELECT_COLUMNS} FROM public.recommendation_serves WHERE serve_id = $1",
        serve_id,
    )
    return dict(row) if row is not None else None
