"""DB access for ``entity_dossiers`` (plan §8.2).

Per-entity dossier produced by the S2 Profiler swarm. The ``embedding``
column is DEFERRED in v1 (pgvector not installed locally — see plan
§15.9, §17). ``embedding_text`` (the 200-word version) is preserved so
embeddings can be back-filled retroactively.

Each call to :func:`upsert_dossier` appends a NEW row; "latest" is
selected by ``generated_at DESC`` via the ``(entity_id, generated_at)``
index. There is no unique constraint on ``entity_id`` — history is
retained for replay.

``nearest_neighbors(embedding, k)`` is DEFERRED until pgvector is enabled.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg


SELECT_COLUMNS = """
    dossier_id::text         AS dossier_id,
    entity_id::text          AS entity_id,
    archetype,
    current_posture,
    motive_priors,
    friction_notes,
    recent_activity,
    embedding_text,
    features_snapshot,
    generated_by_swarm,
    swarm_packet_hash,
    packet_json,
    generated_at
"""


async def upsert_dossier(
    conn: asyncpg.Connection,
    *,
    entity_id: UUID | str,
    archetype: str,
    embedding_text: str,
    features_snapshot: dict[str, Any],
    current_posture: str | None = None,
    motive_priors: dict[str, Any] | None = None,
    friction_notes: str | None = None,
    recent_activity: str | None = None,
    generated_by_swarm: str | None = None,
    swarm_packet_hash: str | None = None,
    packet_json: dict[str, Any] | None = None,
) -> UUID:
    """Append a new dossier row for ``entity_id``; return ``dossier_id``.

    Function is named ``upsert_dossier`` per plan §0.8, but semantically
    appends a new versioned row (history is retained). "Latest" is the
    row with the highest ``generated_at`` for the entity.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO public.entity_dossiers
            (entity_id, archetype, current_posture, motive_priors,
             friction_notes, recent_activity, embedding_text,
             features_snapshot, generated_by_swarm, swarm_packet_hash, packet_json)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8::jsonb, $9, $10, $11::jsonb)
        RETURNING dossier_id
        """,
        entity_id,
        archetype,
        current_posture,
        json.dumps(motive_priors) if motive_priors is not None else None,
        friction_notes,
        recent_activity,
        embedding_text,
        json.dumps(features_snapshot),
        generated_by_swarm,
        swarm_packet_hash,
        json.dumps(packet_json) if packet_json is not None else None,
    )
    return row["dossier_id"]


async def fetch_latest_dossier(
    conn: asyncpg.Connection, entity_id: UUID | str
) -> dict[str, Any] | None:
    """Return the most recent dossier for ``entity_id`` or ``None``."""
    row = await conn.fetchrow(
        f"""
        SELECT {SELECT_COLUMNS}
        FROM public.entity_dossiers
        WHERE entity_id = $1
        ORDER BY generated_at DESC
        LIMIT 1
        """,
        entity_id,
    )
    return dict(row) if row is not None else None


async def bulk_fetch_latest(
    conn: asyncpg.Connection,
    entity_ids: list[UUID | str],
) -> dict[str, dict[str, Any]]:
    """Return ``entity_id_str -> latest_dossier_dict`` for the given entities.

    Uses a single ``DISTINCT ON (entity_id)`` query ordered by
    ``generated_at DESC``. Entities with no dossier are simply absent
    from the result map.
    """
    if not entity_ids:
        return {}
    str_ids = [str(e) for e in entity_ids]
    rows = await conn.fetch(
        f"""
        SELECT DISTINCT ON (entity_id)
            {SELECT_COLUMNS}
        FROM public.entity_dossiers
        WHERE entity_id = ANY($1::uuid[])
        ORDER BY entity_id, generated_at DESC
        """,
        str_ids,
    )
    return {str(r["entity_id"]): dict(r) for r in rows}
