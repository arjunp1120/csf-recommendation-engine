"""DB access for ``correlative_attribution_metrics`` (plan §15.6).

Nightly correlative-attribution job output (plan §14.5). For each served
suggestion (recommendation or match), did compatible activity occur in
``trade_history`` within ``window_hours`` after ``served_at``? Compared
against a baseline rate. NOT a causal-attribution claim.

The ``one_target`` CHECK constraint enforces exactly one of
``serve_id`` / ``match_id`` is non-null — :func:`insert_metric`
pre-checks this to surface a clean ``ValueError`` rather than a
Postgres ``CheckViolationError``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg


SELECT_COLUMNS = """
    metric_id::text                  AS metric_id,
    serve_id::text                   AS serve_id,
    match_id::text                   AS match_id,
    window_hours,
    observed_compatible_activity,
    baseline_rate,
    time_to_activity_min,
    computed_at
"""


async def insert_metric(
    conn: asyncpg.Connection,
    *,
    window_hours: int,
    observed_compatible_activity: bool,
    serve_id: UUID | str | None = None,
    match_id: UUID | str | None = None,
    baseline_rate: float | None = None,
    time_to_activity_min: int | None = None,
) -> UUID:
    """Insert one attribution metric row; return the generated ``metric_id``.

    Caller MUST pass exactly one of ``serve_id`` / ``match_id`` (the DB
    ``one_target`` CHECK enforces this; we pre-check here for a clean
    error message).
    """
    targets_set = (serve_id is not None) + (match_id is not None)
    if targets_set != 1:
        raise ValueError(
            "exactly one of serve_id / match_id must be provided "
            f"(got serve_id={serve_id!r}, match_id={match_id!r})"
        )

    row = await conn.fetchrow(
        """
        INSERT INTO public.correlative_attribution_metrics
            (serve_id, match_id, window_hours,
             observed_compatible_activity, baseline_rate, time_to_activity_min)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING metric_id
        """,
        serve_id,
        match_id,
        window_hours,
        observed_compatible_activity,
        baseline_rate,
        time_to_activity_min,
    )
    return row["metric_id"]


async def summary_lift_over_baseline(
    conn: asyncpg.Connection,
    *,
    window_hours: int,
    since_days: int = 30,
) -> dict[str, Any]:
    """Aggregate observed vs baseline rate over recent attribution rows.

    Returns ``{n_metrics, n_serves, n_matches, observed_rate,
    avg_baseline_rate, lift, avg_time_to_activity_min}``. ``lift`` is
    ``observed_rate - avg_baseline_rate`` (None if either is null).
    """
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*)                                       AS n_metrics,
            COUNT(*) FILTER (WHERE serve_id IS NOT NULL)   AS n_serves,
            COUNT(*) FILTER (WHERE match_id IS NOT NULL)   AS n_matches,
            AVG(observed_compatible_activity::int)         AS observed_rate,
            AVG(baseline_rate)                             AS avg_baseline_rate,
            AVG(time_to_activity_min) FILTER (WHERE observed_compatible_activity)
                                                           AS avg_time_to_activity_min
        FROM public.correlative_attribution_metrics
        WHERE window_hours = $1
          AND computed_at >= NOW() - make_interval(days => $2::int)
        """,
        window_hours,
        since_days,
    )
    observed = row["observed_rate"]
    baseline = row["avg_baseline_rate"]
    lift: float | None = None
    if observed is not None and baseline is not None:
        lift = float(observed) - float(baseline)
    return {
        "n_metrics": int(row["n_metrics"] or 0),
        "n_serves": int(row["n_serves"] or 0),
        "n_matches": int(row["n_matches"] or 0),
        "observed_rate": float(observed) if observed is not None else None,
        "avg_baseline_rate": float(baseline) if baseline is not None else None,
        "lift": lift,
        "avg_time_to_activity_min": (
            float(row["avg_time_to_activity_min"])
            if row["avg_time_to_activity_min"] is not None
            else None
        ),
        "window_hours": window_hours,
        "since_days": since_days,
    }
