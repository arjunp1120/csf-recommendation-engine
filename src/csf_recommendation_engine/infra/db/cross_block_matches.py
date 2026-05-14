"""DB access for ``cross_block_matches`` — new-shape helpers (plan §15.7).

The legacy bulk-insert / recent-summary helpers remain in ``trades.py``
for the nightly batch path. This module is the **new** single-match
write path used by the synchronous match pipeline (``/ioi/accept``,
back-stop polling) plus the dedup query helpers required by plan §10.2.

Schema additions in 0008 (per plan §15.7):
  - packet_hash, packet_json, variant_id, swarm_response
  - mutual_fit_score (numeric 0..1) — canonical scorer output
    (distinct from the legacy ``match_percentage`` int 0-100 that is
    kept for display)
  - strategist_script (jsonb) — Strategist swarm output, populated in
    background after the sync match path returns
  - originator_inquiry_id, counter_inquiry_id — FKs to
    ``voice_inquiries.inquiry_id`` used for §10.2 inquiry-pair dedup
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg


SELECT_COLUMNS = """
    match_id::text             AS match_id,
    product_name,
    match_percentage,
    buyer_entity_id::text      AS buyer_entity_id,
    buyer_side::text           AS buyer_side,
    buyer_lots,
    seller_entity_id::text     AS seller_entity_id,
    seller_side::text          AS seller_side,
    seller_lots,
    description,
    packet_hash,
    packet_json,
    variant_id,
    swarm_response,
    mutual_fit_score,
    strategist_script,
    originator_inquiry_id::text AS originator_inquiry_id,
    counter_inquiry_id::text    AS counter_inquiry_id,
    created_at,
    updated_at
"""


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


async def insert_match(
    conn: asyncpg.Connection,
    *,
    product_name: str,
    buyer_entity_id: UUID | str,
    buyer_side: str,
    buyer_lots: int,
    seller_entity_id: UUID | str,
    seller_side: str,
    seller_lots: int,
    match_percentage: int,
    mutual_fit_score: float | None = None,
    description: str | None = None,
    packet_hash: str | None = None,
    packet_json: dict[str, Any] | None = None,
    variant_id: str | None = None,
    swarm_response: dict[str, Any] | None = None,
    originator_inquiry_id: UUID | str | None = None,
    counter_inquiry_id: UUID | str | None = None,
) -> UUID:
    """Insert one new-shape match row; return ``match_id``.

    ``strategist_script`` is intentionally NOT a parameter — it is
    populated later by :func:`update_with_strategist` once the
    background Strategist swarm completes (plan §11.2 / §4.2).
    """
    row = await conn.fetchrow(
        """
        INSERT INTO public.cross_block_matches
            (product_name, match_percentage, buyer_entity_id, buyer_side,
             buyer_lots, seller_entity_id, seller_side, seller_lots, description,
             packet_hash, packet_json, variant_id, swarm_response,
             mutual_fit_score, originator_inquiry_id, counter_inquiry_id)
        VALUES ($1, $2, $3, $4::trade_side, $5, $6, $7::trade_side, $8, $9,
                $10, $11::jsonb, $12, $13::jsonb, $14, $15, $16)
        RETURNING match_id
        """,
        product_name,
        match_percentage,
        buyer_entity_id,
        buyer_side,
        buyer_lots,
        seller_entity_id,
        seller_side,
        seller_lots,
        description,
        packet_hash,
        json.dumps(packet_json) if packet_json is not None else None,
        variant_id,
        json.dumps(swarm_response) if swarm_response is not None else None,
        mutual_fit_score,
        originator_inquiry_id,
        counter_inquiry_id,
    )
    return row["match_id"]


async def update_with_strategist(
    conn: asyncpg.Connection,
    *,
    match_id: UUID | str,
    strategist_script: dict[str, Any],
    swarm_response: dict[str, Any] | None = None,
) -> None:
    """Persist the Strategist swarm's narrative output for a match.

    Called from the background job after the sync match path has already
    returned. Sets ``updated_at`` so callers can detect newly-narrated
    matches via the index on ``updated_at``.
    """
    await conn.execute(
        """
        UPDATE public.cross_block_matches
           SET strategist_script = $2::jsonb,
               swarm_response    = COALESCE($3::jsonb, swarm_response),
               updated_at        = NOW()
         WHERE match_id = $1
        """,
        match_id,
        json.dumps(strategist_script),
        json.dumps(swarm_response) if swarm_response is not None else None,
    )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def fetch_pending_strategist(
    conn: asyncpg.Connection, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Return matches with no ``strategist_script`` yet, oldest first.

    Bounded by ``packet_json IS NOT NULL`` so legacy nightly-job rows
    (which never carried a packet) are excluded — the Strategist
    background worker needs the packet for replay.
    """
    rows = await conn.fetch(
        f"""
        SELECT {SELECT_COLUMNS}
        FROM public.cross_block_matches
        WHERE strategist_script IS NULL
          AND packet_json IS NOT NULL
        ORDER BY created_at ASC
        LIMIT $1
        """,
        limit,
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Dedup helpers (plan §10.2)
# ---------------------------------------------------------------------------


async def pair_already_matched_recently(
    conn: asyncpg.Connection,
    *,
    originator_inquiry_id: UUID | str,
    counter_inquiry_id: UUID | str,
    window_hours: int,
) -> bool:
    """Has this exact inquiry pair been matched within the time window?

    Ordering-agnostic: ``(A, B)`` and ``(B, A)`` are the same pair.
    """
    row = await conn.fetchrow(
        """
        SELECT EXISTS (
            SELECT 1
            FROM public.cross_block_matches
            WHERE created_at >= (NOW() - make_interval(hours => $3::int))::timestamp
              AND (
                  (originator_inquiry_id = $1 AND counter_inquiry_id = $2) OR
                  (originator_inquiry_id = $2 AND counter_inquiry_id = $1)
              )
        ) AS exists_pair
        """,
        originator_inquiry_id,
        counter_inquiry_id,
        window_hours,
    )
    return bool(row["exists_pair"])


async def entity_pair_recently_dismissed(
    conn: asyncpg.Connection,
    *,
    buyer_entity_id: UUID | str,
    seller_entity_id: UUID | str,
    product_name: str,
    window_hours: int,
) -> bool:
    """Has this entity pair on this product been thumbs-down'd recently?

    Joins ``cross_block_matches`` to ``cross_block_feedback`` for
    ``is_accurate=FALSE`` within the window. Ordering-agnostic on the
    entity pair. Used by the §10.2 deduper to suppress re-proposing
    pairs the broker has already rejected.
    """
    row = await conn.fetchrow(
        """
        SELECT EXISTS (
            SELECT 1
            FROM public.cross_block_matches m
            JOIN public.cross_block_feedback f USING (match_id)
            WHERE f.is_accurate = FALSE
              AND f.created_at >= (NOW() - make_interval(hours => $4::int))::timestamp
              AND m.product_name = $3
              AND (
                  (m.buyer_entity_id = $1 AND m.seller_entity_id = $2) OR
                  (m.buyer_entity_id = $2 AND m.seller_entity_id = $1)
              )
        ) AS dismissed
        """,
        buyer_entity_id,
        seller_entity_id,
        product_name,
        window_hours,
    )
    return bool(row["dismissed"])
