"""DB access for ``instrument_products`` (plan §7.1).

Canonical mapping from ``trade_history.instrument_name`` /
``trades.instrument_name`` (long-tail strings) to the canonical
``product_name`` vocabulary used by ``voice_inquiries.product``,
``exposure_by_product.product_name``, and ``entity_products.product_name``.

This module is the low-level DB shim. The in-memory
``ProductResolver`` (Step 0.9) loads the full table via
:func:`load_all` at startup and serves all hot lookups.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID

import asyncpg


SELECT_ALL_COLUMNS = """
    mapping_id::text       AS mapping_id,
    instrument_name,
    symbol_pattern,
    product_name,
    product_family,
    contract_month,
    expiry_date,
    structure_type,
    source,
    confidence,
    reviewed_at,
    reviewed_by::text      AS reviewed_by,
    created_at,
    updated_at
"""


async def load_all(conn: asyncpg.Connection) -> list[dict[str, Any]]:
    """Return every row from ``instrument_products``.

    The resolver caches this in-memory and refreshes on the nightly job
    cycle. Order is undefined; callers index by ``instrument_name`` or
    use the resolver.
    """
    rows = await conn.fetch(
        f"SELECT {SELECT_ALL_COLUMNS} FROM public.instrument_products"
    )
    return [dict(r) for r in rows]


async def flag_unresolved_instruments(
    conn: asyncpg.Connection,
    instrument_names: list[str],
) -> list[str]:
    """Return the subset of ``instrument_names`` not present in the mapping table.

    The plan's nightly unresolved-flag job (Step 0.9) calls this with the
    distinct ``instrument_name`` values from ``trade_history`` and logs
    the result. Callers handle the surface (log / metric / ops alert).
    """
    if not instrument_names:
        return []
    rows = await conn.fetch(
        """
        SELECT DISTINCT name
        FROM unnest($1::varchar[]) AS s(name)
        WHERE NOT EXISTS (
            SELECT 1
            FROM public.instrument_products ip
            WHERE ip.instrument_name = s.name
        )
        """,
        instrument_names,
    )
    return [str(r["name"]) for r in rows]


async def upsert(
    conn: asyncpg.Connection,
    *,
    product_name: str,
    source: str,
    instrument_name: str | None = None,
    symbol_pattern: str | None = None,
    product_family: str | None = None,
    contract_month: str | None = None,
    expiry_date: date | None = None,
    structure_type: str | None = None,
    confidence: float = 1.0,
    reviewed_at: Any = None,
    reviewed_by: UUID | str | None = None,
) -> UUID:
    """Insert-or-update one mapping; return the ``mapping_id``.

    The unique index on ``instrument_name`` is partial
    (``WHERE instrument_name IS NOT NULL``), so ``ON CONFLICT`` must
    match the same predicate. Rows with ``instrument_name=NULL`` and a
    ``symbol_pattern`` always insert as new — there is no implicit
    dedup for pattern-based rows in v1 (operator-curated).
    """
    if instrument_name is None and symbol_pattern is None:
        raise ValueError(
            "either instrument_name or symbol_pattern must be provided"
        )

    if instrument_name is not None:
        row = await conn.fetchrow(
            """
            INSERT INTO public.instrument_products
                (instrument_name, symbol_pattern, product_name, product_family,
                 contract_month, expiry_date, structure_type, source, confidence,
                 reviewed_at, reviewed_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (instrument_name) WHERE instrument_name IS NOT NULL
            DO UPDATE SET
                symbol_pattern = EXCLUDED.symbol_pattern,
                product_name   = EXCLUDED.product_name,
                product_family = EXCLUDED.product_family,
                contract_month = EXCLUDED.contract_month,
                expiry_date    = EXCLUDED.expiry_date,
                structure_type = EXCLUDED.structure_type,
                source         = EXCLUDED.source,
                confidence     = EXCLUDED.confidence,
                reviewed_at    = EXCLUDED.reviewed_at,
                reviewed_by    = EXCLUDED.reviewed_by,
                updated_at     = NOW()
            RETURNING mapping_id
            """,
            instrument_name,
            symbol_pattern,
            product_name,
            product_family,
            contract_month,
            expiry_date,
            structure_type,
            source,
            confidence,
            reviewed_at,
            reviewed_by,
        )
        return row["mapping_id"]

    row = await conn.fetchrow(
        """
        INSERT INTO public.instrument_products
            (instrument_name, symbol_pattern, product_name, product_family,
             contract_month, expiry_date, structure_type, source, confidence,
             reviewed_at, reviewed_by)
        VALUES (NULL, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING mapping_id
        """,
        symbol_pattern,
        product_name,
        product_family,
        contract_month,
        expiry_date,
        structure_type,
        source,
        confidence,
        reviewed_at,
        reviewed_by,
    )
    return row["mapping_id"]
