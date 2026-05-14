"""DB access for ``voice_inquiries`` (base table + plan §15.8 operational columns).

The engine is the **owner** of the operational columns added by
``0009_alter_voice_inquiries_processing.sql``:

  - ``processed``           BOOLEAN NOT NULL DEFAULT FALSE
  - ``processed_at``        TIMESTAMPTZ
  - ``processing_attempts`` INTEGER NOT NULL DEFAULT 0
  - ``processing_error``    TEXT

The engine **never writes** to the content fields (``inquiry_id``,
``broker_name``, ``side``, ``product``, ``quantity``, ``entity_id``,
``trader_contact_info``, ``description``) — those are owned by the
external ingestion service that calls ``POST /ioi/accept``. See plan
§2.3 (point 4), §6, §15.8.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg


SELECT_COLUMNS = """
    inquiry_id::text          AS inquiry_id,
    inquiry_timestamp,
    broker_name,
    side::text                AS side,
    product,
    quantity,
    entity_id::text           AS entity_id,
    trader_contact_info,
    description,
    created_at,
    processed,
    processed_at,
    processing_attempts,
    processing_error
"""


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


async def fetch_active_inquiries(
    conn: asyncpg.Connection, *, within_minutes: int
) -> list[dict[str, Any]]:
    """Return inquiries created within the last ``within_minutes`` minutes.

    The expiry window is set via ``VOICE_INQUIRY_EXPIRY_MINUTES`` in Step 0.3.
    Note: ``voice_inquiries.created_at`` is ``timestamp without time zone``;
    we cast ``NOW()`` to match.
    """
    rows = await conn.fetch(
        f"""
        SELECT {SELECT_COLUMNS}
        FROM public.voice_inquiries
        WHERE created_at >= (NOW() - make_interval(mins => $1::int))::timestamp
        ORDER BY created_at DESC
        """,
        within_minutes,
    )
    return [dict(r) for r in rows]


async def fetch_by_id(
    conn: asyncpg.Connection, inquiry_id: UUID | str
) -> dict[str, Any] | None:
    """Return one inquiry row by ``inquiry_id`` or ``None``."""
    row = await conn.fetchrow(
        f"SELECT {SELECT_COLUMNS} FROM public.voice_inquiries WHERE inquiry_id = $1",
        inquiry_id,
    )
    return dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# Processing-state helpers (engine-owned columns only; see §15.8)
# ---------------------------------------------------------------------------


async def mark_processed(
    conn: asyncpg.Connection, inquiry_id: UUID | str
) -> None:
    """Mark an inquiry as successfully processed.

    Sets ``processed=TRUE`` and ``processed_at=NOW()``. Idempotent —
    re-applying yields the same final state (processed_at advances).
    """
    await conn.execute(
        """
        UPDATE public.voice_inquiries
           SET processed    = TRUE,
               processed_at = NOW()
         WHERE inquiry_id = $1
        """,
        inquiry_id,
    )


async def mark_processing_failed(
    conn: asyncpg.Connection,
    inquiry_id: UUID | str,
    error: str,
) -> None:
    """Record a processing failure for ``inquiry_id``.

    Increments ``processing_attempts`` and stores the truncated error
    message. The partial index ``idx_voice_inquiries_unprocessed`` drops
    the row once attempts reaches ``max_attempts`` (3 per Step 0.3).
    """
    await conn.execute(
        """
        UPDATE public.voice_inquiries
           SET processing_attempts = processing_attempts + 1,
               processing_error    = $2
         WHERE inquiry_id = $1
        """,
        inquiry_id,
        error,
    )


async def claim_unprocessed_for_processing(
    conn: asyncpg.Connection,
    *,
    limit: int,
    grace_seconds: int,
    max_attempts: int,
) -> list[UUID]:
    """Atomically claim a batch of unprocessed inquiries; return ``inquiry_id`` UUIDs.

    Uses ``FOR UPDATE SKIP LOCKED`` so concurrent pollers do not collide
    on the same rows. The CALLER MUST be inside a transaction for the
    lock to be held for the lifetime of the unit of work.

    Filters:
      - ``processed = FALSE``
      - ``processing_attempts < max_attempts`` (3 per Step 0.3)
      - ``created_at < NOW() - grace_seconds`` (lets the sync ``/ioi/accept``
        path win normal cases; back-stop only picks up rows the sync path
        failed to process).

    The query is index-friendly: it matches the partial index
    ``idx_voice_inquiries_unprocessed (created_at) WHERE processed = FALSE
    AND processing_attempts < 3``.
    """
    rows = await conn.fetch(
        """
        SELECT inquiry_id
        FROM public.voice_inquiries
        WHERE processed = FALSE
          AND processing_attempts < $1
          AND created_at < (NOW() - make_interval(secs => $2::int))::timestamp
        ORDER BY created_at
        LIMIT $3
        FOR UPDATE SKIP LOCKED
        """,
        max_attempts,
        grace_seconds,
        limit,
    )
    return [r["inquiry_id"] for r in rows]
