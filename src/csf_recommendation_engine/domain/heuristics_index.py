from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HeuristicsIndex:
    active_venues_by_entity: dict[str, set[str]]
    hourly_ratios_by_entity: dict[str, np.ndarray]
    max_ratio_by_entity: dict[str, float]
    size_profile_by_entity_instrument: dict[tuple[str, str], tuple[float, float]]
    last_trade_date_by_entity_instrument: dict[tuple[str, str], date]


def load_heuristics_from_parquet(
    *, entity_features_path: Path, instrument_features_path: Path
) -> HeuristicsIndex:
    entity_df = pd.read_parquet(entity_features_path, engine="pyarrow")
    inst_df = pd.read_parquet(instrument_features_path, engine="pyarrow")

    # Entity features
    active_venues_by_entity: dict[str, set[str]] = {}
    hourly_ratios_by_entity: dict[str, np.ndarray] = {}
    max_ratio_by_entity: dict[str, float] = {}

    if "entity_id" in entity_df.columns:
        entity_df["entity_id"] = entity_df["entity_id"].astype(str)

    if "active_venues" in entity_df.columns:
        for _, row in entity_df[["entity_id", "active_venues"]].dropna().iterrows():
            entity_id = str(row["entity_id"])
            venues_raw = row["active_venues"]
            if isinstance(venues_raw, (list, tuple, np.ndarray)):
                active_venues_by_entity[entity_id] = {str(v) for v in venues_raw}

    if "hourly_ratios" in entity_df.columns:
        subset_cols = ["entity_id", "hourly_ratios"]
        if "max_ratio" in entity_df.columns:
            subset_cols.append("max_ratio")
        for _, row in entity_df[subset_cols].dropna(subset=["entity_id", "hourly_ratios"]).iterrows():
            entity_id = str(row["entity_id"])
            ratios_raw = row["hourly_ratios"]
            if isinstance(ratios_raw, (list, tuple, np.ndarray)) and len(ratios_raw) == 24:
                hourly_ratios_by_entity[entity_id] = np.array(ratios_raw, dtype=float)
                if "max_ratio" in row and row["max_ratio"] is not None:
                    max_ratio_by_entity[entity_id] = float(row["max_ratio"])

    # Instrument features
    size_profile_by_entity_instrument: dict[tuple[str, str], tuple[float, float]] = {}
    last_trade_date_by_entity_instrument: dict[tuple[str, str], date] = {}

    if "entity_id" in inst_df.columns:
        inst_df["entity_id"] = inst_df["entity_id"].astype(str)
    if "instrument_name" in inst_df.columns:
        inst_df["instrument_name"] = inst_df["instrument_name"].astype(str)

    for _, row in inst_df.iterrows():
        if "entity_id" not in row or "instrument_name" not in row:
            continue
        entity_id = str(row["entity_id"])
        instrument_name = str(row["instrument_name"])
        key = (entity_id, instrument_name)

        if "mean_size" in row and "stddev_size" in row:
            mean_size = row["mean_size"]
            stddev_size = row["stddev_size"]
            if mean_size is not None and stddev_size is not None:
                try:
                    size_profile_by_entity_instrument[key] = (float(mean_size), float(stddev_size))
                except (TypeError, ValueError):
                    pass

        if "last_trade_date" in row and row["last_trade_date"] is not None:
            value = row["last_trade_date"]
            if isinstance(value, pd.Timestamp):
                last_trade_date_by_entity_instrument[key] = value.date()
            elif isinstance(value, date):
                last_trade_date_by_entity_instrument[key] = value

    return HeuristicsIndex(
        active_venues_by_entity=active_venues_by_entity,
        hourly_ratios_by_entity=hourly_ratios_by_entity,
        max_ratio_by_entity=max_ratio_by_entity,
        size_profile_by_entity_instrument=size_profile_by_entity_instrument,
        last_trade_date_by_entity_instrument=last_trade_date_by_entity_instrument,
    )
