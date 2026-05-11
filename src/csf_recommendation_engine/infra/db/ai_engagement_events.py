"""DB access for ``ai_engagement_events`` (plan §15.5).

Append-only log of broker-UI engagement events. Each row points at
exactly one of ``recommendation_serves`` or ``cross_block_matches`` — the
DB enforces this via the ``one_target`` CHECK constraint.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg


VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "impression",
        "click",
        "dwell",
        "thumbs_up",
        "thumbs_down",
        "in_progress",
        "coach_drafted",
        "copy_text",
    }
)


async def insert_event(
    conn: asyncpg.Connection,
    *,
    event_type: str,
    serve_id: UUID | None = None,
    match_id: UUID | None = None,
    user_id: UUID | None = None,
    metadata: dict[str, Any] | None = None,
) -> UUID:
    """Insert a single engagement event row; return the generated ``event_id``.

    Caller MUST pass exactly one of ``serve_id`` / ``match_id`` (the DB's
    ``one_target`` CHECK enforces this; we pre-check here to surface a
    clean ``ValueError`` rather than a Postgres CheckViolationError).
    ``event_type`` must be one of ``VALID_EVENT_TYPES`` (plan §15.5);
    the column itself is ``varchar(40)`` and does not enforce this.
    """
    targets_set = (serve_id is not None) + (match_id is not None)
    if targets_set != 1:
        raise ValueError(
            "exactly one of serve_id / match_id must be provided "
            f"(got serve_id={serve_id!r}, match_id={match_id!r})"
        )
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"event_type {event_type!r} not in {sorted(VALID_EVENT_TYPES)}"
        )

    metadata_json = json.dumps(metadata) if metadata is not None else None

    row = await conn.fetchrow(
        """
        INSERT INTO public.ai_engagement_events
            (serve_id, match_id, user_id, event_type, metadata)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        RETURNING event_id
        """,
        serve_id,
        match_id,
        user_id,
        event_type,
        metadata_json,
    )
    return row["event_id"]
