"""Nightly heuristics build job for CSF Recommendation Engine.

Heuristics artifacts are used for filtering+reranking - AFTER LightFM output.

Aka, these are not features used by LightFM, but rather features used to adjust 
the final output of the recommendation engine based on recent activity and other factors.

Used via domain/reranking.py

ALSO used by Intelligence Layer for context, in domain/intelligence_layer.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path

import pandas as pd

from csf_recommendation_engine.core.config import get_settings
from csf_recommendation_engine.domain.heuristics_build import (
    assemble_entity_features,
    assemble_instrument_features,
    build_active_venues,
    build_recency_profile,
    build_size_profile,
    build_time_affinity,
    validate_raw_trade_frame,
)
from csf_recommendation_engine.infra.db.pool import get_db_pool
from csf_recommendation_engine.infra.db.trade_history import fetch_trade_history_frame

logger = logging.getLogger(__name__)


def _parse_completed_statuses(raw: str) -> list[str]:
    parts = [p.strip() for p in (raw or "").split(",")]
    return [p for p in parts if p]


def _build_artifact_paths(artifact_dir: Path, *, run_time_utc: datetime) -> dict[str, Path]:
    date_tag = run_time_utc.strftime("%Y%m%d")
    return {
        "entity_latest": artifact_dir / "entity_features_latest.parquet",
        "instrument_latest": artifact_dir / "instrument_features_latest.parquet",
        "entity_dated": artifact_dir / f"entity_features_{date_tag}.parquet",
        "instrument_dated": artifact_dir / f"instrument_features_{date_tag}.parquet",
    }


async def run_nightly_heuristics_build() -> None:
    settings = get_settings()
    pool = get_db_pool()

    completed_statuses = _parse_completed_statuses(settings.heuristics_completed_statuses)
    if not completed_statuses:
        raise RuntimeError("HEURISTICS_COMPLETED_STATUSES must not be empty")

    async with pool.acquire() as conn:
        df_raw = await fetch_trade_history_frame(
            conn,
            completed_statuses=completed_statuses,
            history_days=settings.heuristics_history_days,
        )

    if df_raw.empty:
        logger.warning("No trade_history rows returned; skipping heuristics artifact write")
        return

    # Normalize expected dtypes
    df_raw["entity_id"] = df_raw["entity_id"].astype(str)
    df_raw["venue"] = df_raw["venue"].astype(str)
    df_raw["instrument_name"] = df_raw["instrument_name"].astype(str)
    df_raw["quantity"] = pd.to_numeric(df_raw["quantity"], errors="coerce")
    df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"], errors="coerce").dt.date
    df_raw["trade_hour"] = pd.to_numeric(df_raw["trade_hour"], errors="coerce").fillna(-1).astype(int)

    validate_raw_trade_frame(df_raw)

    df_venues = build_active_venues(
        df_raw,
        lookback_days=settings.heuristics_active_venue_lookback_days,
    )
    df_size = build_size_profile(df_raw)
    df_recency = build_recency_profile(df_raw)
    df_time = build_time_affinity(df_raw)

    df_entity = assemble_entity_features(df_venues, df_time)
    df_instrument = assemble_instrument_features(df_size, df_recency)

    run_time_utc = datetime.now(timezone.utc)
    artifact_dir = Path(settings.heuristics_artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    paths = _build_artifact_paths(artifact_dir, run_time_utc=run_time_utc)

    df_entity.to_parquet(paths["entity_dated"], engine="pyarrow", index=False)
    df_instrument.to_parquet(paths["instrument_dated"], engine="pyarrow", index=False)

    # Maintain stable "latest" aliases
    df_entity.to_parquet(paths["entity_latest"], engine="pyarrow", index=False)
    df_instrument.to_parquet(paths["instrument_latest"], engine="pyarrow", index=False)

    logger.info(
        "Nightly heuristics artifacts written",
        extra={
            "entity_latest": str(paths["entity_latest"]),
            "instrument_latest": str(paths["instrument_latest"]),
            "rows_raw": int(len(df_raw)),
            "rows_entity": int(len(df_entity)),
            "rows_instrument": int(len(df_instrument)),
        },
    )


if __name__ == "__main__":
    asyncio.run(run_nightly_heuristics_build())
