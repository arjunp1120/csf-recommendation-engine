"""DB access for ``market_snapshots`` and ``market_color``.

Inputs to the S3 Market Reader swarm (plan §11.2). The base tables key
on ``instrument_name`` / no product axis, respectively — there is no
``product_name`` column on either. Therefore:

  - :func:`fetch_recent_snapshots` requires a **pre-resolved**
    ``instrument_names`` list. Callers use the in-memory
    :class:`ProductResolver` (Step 0.9) to convert product → instruments.
  - :func:`fetch_recent_color` returns recent rows unfiltered; the
    downstream pipeline / LLM does the product-relevance filtering since
    ``market_color`` has no structured product axis.

The plan's signature ``fetch_recent_snapshots(product, window)``
deliberately translates to ``(instrument_names, window_minutes)`` at
this layer; the ``product``→``instruments`` pre-resolution belongs in
the orchestration code that owns the resolver.
"""

from __future__ import annotations

from typing import Any

import asyncpg


SNAPSHOT_COLUMNS = """
    snapshot_id::text     AS snapshot_id,
    instrument_code,
    instrument_name,
    price,
    price_change,
    price_change_pct,
    bid_price,
    ask_price,
    volume,
    open_interest,
    high_price,
    low_price,
    snapshot_timestamp,
    created_at
"""

COLOR_COLUMNS = """
    color_id::text  AS color_id,
    headline,
    description,
    source,
    published_at,
    created_at,
    created_by_ai_agent
"""


async def fetch_recent_snapshots(
    conn: asyncpg.Connection,
    *,
    instrument_names: list[str],
    window_minutes: int = 60,
) -> list[dict[str, Any]]:
    """Return ``market_snapshots`` rows for the given instruments within the window.

    ``instrument_names`` is pre-resolved by the caller (see module
    docstring). Empty input returns ``[]``.
    """
    if not instrument_names:
        return []
    rows = await conn.fetch(
        f"""
        SELECT {SNAPSHOT_COLUMNS}
        FROM public.market_snapshots
        WHERE instrument_name = ANY($1::varchar[])
          AND snapshot_timestamp >= (NOW() - make_interval(mins => $2::int))::timestamp
        ORDER BY snapshot_timestamp DESC
        """,
        instrument_names,
        window_minutes,
    )
    return [dict(r) for r in rows]


async def fetch_recent_color(
    conn: asyncpg.Connection,
    *,
    window_minutes: int = 60,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return recent ``market_color`` rows ordered by ``published_at`` desc.

    Filters by ``published_at`` (the canonical publish time); falls back
    to ``created_at`` for rows with NULL ``published_at`` by including
    them when they're recent enough.
    """
    rows = await conn.fetch(
        f"""
        SELECT {COLOR_COLUMNS}
        FROM public.market_color
        WHERE COALESCE(published_at, created_at)
              >= (NOW() - make_interval(mins => $1::int))::timestamp
        ORDER BY COALESCE(published_at, created_at) DESC
        LIMIT $2
        """,
        window_minutes,
        limit,
    )
    return [dict(r) for r in rows]
