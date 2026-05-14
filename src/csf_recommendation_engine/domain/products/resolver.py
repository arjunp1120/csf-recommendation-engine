"""In-memory ``instrument_name`` -> product mapping (plan §7.2, §0.9).

Loaded once at startup from ``instrument_products`` rows. Two lookups
per ``resolve()`` call:

  1. Exact match on ``instrument_name`` (case-sensitive; matches the
     partial unique index on the mapping table).
  2. Regex fallback over ``symbol_pattern`` rows. First match wins
     (insertion order is ``mapping_id`` order — operator-curated).

Returns a ``ProductMatch`` with the resolved fields plus ``via``
("exact" or "regex") for telemetry.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

import asyncpg

from csf_recommendation_engine.infra.db import instrument_products

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductMatch:
    """One resolved mapping returned by :meth:`ProductResolver.resolve`."""

    product_name: str
    product_family: str | None
    contract_month: str | None
    expiry_date: date | None
    structure_type: str | None
    source: str
    confidence: float
    mapping_id: str
    via: Literal["exact", "regex"]


def _build_match(row: dict, *, via: Literal["exact", "regex"]) -> ProductMatch:
    return ProductMatch(
        product_name=str(row["product_name"]),
        product_family=row.get("product_family"),
        contract_month=row.get("contract_month"),
        expiry_date=row.get("expiry_date"),
        structure_type=row.get("structure_type"),
        source=str(row.get("source") or "manual"),
        confidence=float(row.get("confidence") or 1.0),
        mapping_id=str(row["mapping_id"]),
        via=via,
    )


class ProductResolver:
    """Lookup ``instrument_name`` -> :class:`ProductMatch`.

    Construct via :meth:`load` or directly with a list of row-dicts
    (rows must come from :func:`instrument_products.load_all`).
    """

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self._exact: dict[str, dict] = {}
        self._patterns: list[tuple[re.Pattern[str], dict]] = []

        for row in rows:
            name = row.get("instrument_name")
            if name:
                # Exact dedup: last row for a given instrument_name wins. The
                # partial unique index in the DB prevents two rows with the
                # same name in practice — this is just defensive.
                self._exact[str(name)] = row
            pattern = row.get("symbol_pattern")
            if pattern:
                try:
                    compiled = re.compile(str(pattern))
                except re.error:
                    logger.warning(
                        "Skipping invalid symbol_pattern in instrument_products",
                        extra={
                            "mapping_id": str(row.get("mapping_id")),
                            "symbol_pattern": pattern,
                        },
                        exc_info=True,
                    )
                    continue
                self._patterns.append((compiled, row))

        logger.info(
            "ProductResolver loaded",
            extra={
                "row_count": len(rows),
                "exact_entries": len(self._exact),
                "regex_entries": len(self._patterns),
            },
        )

    @classmethod
    async def load(cls, conn: asyncpg.Connection) -> "ProductResolver":
        """Convenience constructor — fetch all rows then build the index."""
        rows = await instrument_products.load_all(conn)
        return cls(rows)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def resolve(self, instrument_name: str) -> ProductMatch | None:
        """Return the best :class:`ProductMatch` for ``instrument_name`` or ``None``.

        Order of precedence:
          1. Exact match on ``instrument_products.instrument_name``.
          2. First regex (by ``mapping_id`` insertion order) that matches.

        First-match-wins keeps the resolver deterministic. If multiple
        regex patterns could legitimately apply to one instrument the
        operator should disambiguate by reordering / narrowing patterns.
        """
        if not instrument_name:
            return None

        exact_row = self._exact.get(instrument_name)
        if exact_row is not None:
            return _build_match(exact_row, via="exact")

        for compiled, row in self._patterns:
            if compiled.search(instrument_name):
                return _build_match(row, via="regex")

        return None

    # ------------------------------------------------------------------
    # Introspection helpers (telemetry, tests)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def size(self) -> dict[str, int]:
        return {
            "row_count": len(self._rows),
            "exact_entries": len(self._exact),
            "regex_entries": len(self._patterns),
        }
