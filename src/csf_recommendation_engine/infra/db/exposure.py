"""DB access for ``exposure_summary`` and ``exposure_by_product``.

Brokerage-side aggregate exposure. **Soft signal only in v1** — surfaced
to the LLM Strategist for narrative context; never a numeric ranker term
or hard filter (plan principle #8, §10.4).

``exposure_summary`` has one row (broker book level), no entity FK.
``exposure_by_product`` is broken out per product, FK-linked back to
``exposure_summary`` via ``exposure_id``.
"""

from __future__ import annotations

from typing import Any

import asyncpg


SUMMARY_COLUMNS = """
    exposure_id::text          AS exposure_id,
    calculation_timestamp,
    gross_exposure,
    net_exposure,
    delta_equivalent,
    day_change,
    day_change_avg_pct,
    created_at,
    updated_at
"""

PRODUCT_COLUMNS = """
    product_exposure_id::text  AS product_exposure_id,
    exposure_id::text          AS exposure_id,
    product_name,
    delta,
    futures_position,
    option_position,
    swap_position,
    delta_notional,
    total_exposure,
    day_over_day_pct,
    week_over_week_pct,
    created_at
"""


async def fetch_exposure_summary(
    conn: asyncpg.Connection,
) -> dict[str, Any] | None:
    """Return the most recent ``exposure_summary`` row, or ``None``.

    There is typically a single live row; we pick the latest by
    ``calculation_timestamp`` for safety.
    """
    row = await conn.fetchrow(
        f"""
        SELECT {SUMMARY_COLUMNS}
        FROM public.exposure_summary
        ORDER BY calculation_timestamp DESC
        LIMIT 1
        """
    )
    return dict(row) if row is not None else None


async def fetch_exposure_by_product(
    conn: asyncpg.Connection,
    *,
    exposure_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return per-product exposure rows.

    If ``exposure_id`` is given, returns the breakdown for that summary;
    otherwise returns the breakdown for the most recent summary.
    """
    if exposure_id is not None:
        rows = await conn.fetch(
            f"""
            SELECT {PRODUCT_COLUMNS}
            FROM public.exposure_by_product
            WHERE exposure_id = $1
            ORDER BY product_name
            """,
            exposure_id,
        )
    else:
        rows = await conn.fetch(
            f"""
            SELECT {PRODUCT_COLUMNS}
            FROM public.exposure_by_product
            WHERE exposure_id = (
                SELECT exposure_id FROM public.exposure_summary
                ORDER BY calculation_timestamp DESC
                LIMIT 1
            )
            ORDER BY product_name
            """
        )
    return [dict(r) for r in rows]
