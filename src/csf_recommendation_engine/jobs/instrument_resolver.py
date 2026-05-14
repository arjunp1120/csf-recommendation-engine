"""Nightly unresolved-instrument-flag job (plan §0.9).

Goal: every distinct ``instrument_name`` in ``trade_history`` should
resolve to a ``product_name`` via the :class:`ProductResolver`. Any new
instrument that the resolver fails on is an **operator-review surface**:
the curated mapping table is missing an entry (or a regex pattern).

The job is read-only — it logs a structured event with the unresolved
names plus per-source counts (exact / regex / unresolved) so operators
can decide whether to add a manual mapping, a regex pattern, or to ask
the LLM-suggestion path (Step 0.10) for a hint.

This module exposes :func:`run_unresolved_flag_job`. The scheduler
wiring (cron / interval) belongs in ``main.py`` and is intentionally
left to a follow-up step — Step 0.9 only needs the callable to exist.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from csf_recommendation_engine.core.state import app_state
from csf_recommendation_engine.domain.products import ProductResolver
from csf_recommendation_engine.infra.db.pool import get_db_pool

logger = logging.getLogger(__name__)

# Cap the unresolved-names sample we emit in logs so a runaway long tail
# doesn't blow up the log line; full count is always reported.
_MAX_LOGGED_NAMES = 50


async def _fetch_distinct_trade_history_instruments(
    conn: asyncpg.Connection,
) -> list[str]:
    """Return distinct ``trade_history.instrument_name`` values, sorted."""
    rows = await conn.fetch(
        """
        SELECT DISTINCT instrument_name
        FROM public.trade_history
        WHERE instrument_name IS NOT NULL
        ORDER BY instrument_name
        """
    )
    return [str(r["instrument_name"]) for r in rows]


async def _get_resolver(conn: asyncpg.Connection) -> ProductResolver:
    """Use the resolver loaded into ``app_state`` if present; otherwise
    build a fresh one from the DB (e.g., when the job is invoked from a
    standalone scheduler that has no app context)."""
    cached = await app_state.get("product_resolver")
    if isinstance(cached, ProductResolver):
        return cached
    return await ProductResolver.load(conn)


async def run_unresolved_flag_job() -> dict[str, Any]:
    """Run one pass of the unresolved-instrument-flag job.

    Returns a summary dict:

        {
            "total_distinct": int,
            "resolved_via_exact": int,
            "resolved_via_regex": int,
            "unresolved": int,
            "unresolved_names": list[str],   # full list (no truncation)
        }

    Emits a structured ``logger.info`` event ``instrument_resolver.summary``
    plus, when ``unresolved > 0``, a ``logger.warning`` event
    ``instrument_resolver.unresolved`` carrying a truncated sample
    (≤50 names) so log lines stay bounded.
    """
    pool = get_db_pool()
    async with pool.acquire() as conn:
        resolver = await _get_resolver(conn)
        names = await _fetch_distinct_trade_history_instruments(conn)

    n_exact = 0
    n_regex = 0
    unresolved: list[str] = []
    for name in names:
        match = resolver.resolve(name)
        if match is None:
            unresolved.append(name)
        elif match.via == "exact":
            n_exact += 1
        elif match.via == "regex":
            n_regex += 1

    summary: dict[str, Any] = {
        "total_distinct": len(names),
        "resolved_via_exact": n_exact,
        "resolved_via_regex": n_regex,
        "unresolved": len(unresolved),
        "unresolved_names": unresolved,
    }
    logger.info("instrument_resolver.summary", extra=summary)

    if unresolved:
        logger.warning(
            "instrument_resolver.unresolved",
            extra={
                "unresolved_count": len(unresolved),
                "examples": unresolved[:_MAX_LOGGED_NAMES],
                "truncated": len(unresolved) > _MAX_LOGGED_NAMES,
            },
        )

    return summary
