from datetime import datetime
from io import BytesIO
from pathlib import PurePosixPath

import pandas as pd

REQUIRED_COLUMNS = (
    "initiator_client_id",
    "counterparty_client_id",
    "historical_avg_trade_size",
    "hit_rate",
    "interaction_timestamp",
)


def apply_feature_transformations(raw_df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(raw_df)

    transformed = raw_df.copy()
    transformed["interaction_timestamp"] = pd.to_datetime(
        transformed["interaction_timestamp"],
        utc=True,
        errors="coerce",
    )

    bins = [0.0, 250_000.0, 1_000_000.0, 5_000_000.0, float("inf")]
    labels = ["size_tier_1", "size_tier_2", "size_tier_3", "size_tier_4"]
    transformed["size_tier"] = pd.cut(
        transformed["historical_avg_trade_size"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    return transformed


def validate_feature_frame(
    feature_df: pd.DataFrame,
    min_rows: int,
    null_threshold: float,
) -> None:
    _validate_required_columns(feature_df)

    if min_rows < 1:
        raise ValueError("min_rows must be >= 1")
    if null_threshold < 0 or null_threshold > 1:
        raise ValueError("null_threshold must be between 0 and 1")

    if len(feature_df) < min_rows:
        raise ValueError(f"Feature frame has {len(feature_df)} rows; expected at least {min_rows}")

    null_ratio = feature_df[list(REQUIRED_COLUMNS)].isnull().mean().max()
    if float(null_ratio) > null_threshold:
        raise ValueError(
            f"Feature frame null ratio {null_ratio:.4f} exceeds threshold {null_threshold:.4f}"
        )


def to_parquet_bytes(feature_df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    feature_df.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()


def build_feature_blob_paths(latest_blob_path: str, run_time_utc: datetime) -> tuple[str, str]:
    latest = PurePosixPath(latest_blob_path)
    date_suffix = run_time_utc.strftime("%Y%m%d")

    if latest.name.endswith("_latest.parquet"):
        dated_name = latest.name.replace("_latest.parquet", f"_{date_suffix}.parquet")
    else:
        dated_name = f"daily_features_{date_suffix}.parquet"

    dated = latest.with_name(dated_name)
    return str(dated), str(latest)


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Feature frame missing required columns: {', '.join(missing)}")
