from __future__ import annotations

from datetime import date

import pandas as pd


REQUIRED_RAW_COLUMNS = {
    "entity_id",
    "venue",
    "instrument_name",
    "quantity",
    "trade_date",
    "trade_hour",
}


def validate_raw_trade_frame(df_raw: pd.DataFrame) -> None:
    missing = REQUIRED_RAW_COLUMNS.difference(set(df_raw.columns))
    if missing:
        raise ValueError(f"df_raw missing required columns: {sorted(missing)}")


def build_active_venues(
    df_raw: pd.DataFrame, *, lookback_days: int, as_of: date | None = None
) -> pd.DataFrame:
    validate_raw_trade_frame(df_raw)
    if as_of is None:
        as_of = pd.Timestamp.today().date()

    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=int(lookback_days))
    df_recent = df_raw[df_raw["trade_date"] >= cutoff.date()]

    grouped = (
        df_recent.groupby("entity_id")["venue"]
        .unique()
        .reset_index()
        .rename(columns={"venue": "active_venues"})
    )
    grouped["active_venues"] = grouped["active_venues"].apply(lambda x: [str(v) for v in x])
    return grouped


def build_size_profile(df_raw: pd.DataFrame) -> pd.DataFrame:
    validate_raw_trade_frame(df_raw)
    df_size = (
        df_raw.groupby(["entity_id", "instrument_name"])["quantity"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_size", "std": "stddev_size"})
    )
    df_size["stddev_size"] = df_size["stddev_size"].fillna(0.0)
    return df_size


def build_recency_profile(df_raw: pd.DataFrame) -> pd.DataFrame:
    validate_raw_trade_frame(df_raw)
    df_recency = (
        df_raw.groupby(["entity_id", "instrument_name"])["trade_date"]
        .max()
        .reset_index()
        .rename(columns={"trade_date": "last_trade_date"})
    )
    return df_recency


def build_time_affinity(df_raw: pd.DataFrame) -> pd.DataFrame:
    validate_raw_trade_frame(df_raw)

    df_total_trades = df_raw.groupby("entity_id").size().reset_index(name="total_trades")
    df_hourly = (
        df_raw.groupby(["entity_id", "trade_hour"]).size().reset_index(name="hour_count")
    )

    df_affinity = pd.merge(df_hourly, df_total_trades, on="entity_id")
    df_affinity["hour_ratio"] = df_affinity["hour_count"] / df_affinity["total_trades"]

    df_max_ratio = (
        df_affinity.groupby("entity_id")["hour_ratio"]
        .max()
        .reset_index(name="max_ratio")
    )

    df_affinity = pd.merge(df_affinity, df_max_ratio, on="entity_id")

    def _hourly_ratios(group: pd.DataFrame) -> list[float]:
        ratios = [0.0] * 24
        for _, row in group.iterrows():
            hour = int(row["trade_hour"])
            if 0 <= hour <= 23:
                ratios[hour] = float(row["hour_ratio"])
        return ratios

    df_ratios = (
        df_affinity.groupby("entity_id")
        .apply(_hourly_ratios)
        .reset_index(name="hourly_ratios")
    )

    df_time = pd.merge(df_ratios, df_max_ratio, on="entity_id")
    return df_time


def assemble_entity_features(df_venues: pd.DataFrame, df_time: pd.DataFrame) -> pd.DataFrame:
    df_entity = pd.merge(df_venues, df_time, on="entity_id", how="outer")
    return df_entity


def assemble_instrument_features(df_size: pd.DataFrame, df_recency: pd.DataFrame) -> pd.DataFrame:
    df_inst = pd.merge(df_size, df_recency, on=["entity_id", "instrument_name"], how="outer")
    return df_inst
