from datetime import datetime, timezone

import pandas as pd
import pytest

from csf_recommendation_engine.domain.extraction import (
    apply_feature_transformations,
    build_feature_blob_paths,
    validate_feature_frame,
)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "initiator_client_id": ["i1", "i2"],
            "counterparty_client_id": ["c1", "c2"],
            "historical_avg_trade_size": [100_000.0, 2_000_000.0],
            "hit_rate": [0.2, 0.8],
            "interaction_timestamp": [
                "2026-04-20T10:00:00Z",
                "2026-04-21T11:00:00Z",
            ],
        }
    )


def test_apply_feature_transformations_adds_size_tier() -> None:
    transformed = apply_feature_transformations(_base_frame())
    assert "size_tier" in transformed.columns
    assert transformed["size_tier"].notna().all()


def test_validate_feature_frame_passes_for_valid_data() -> None:
    transformed = apply_feature_transformations(_base_frame())
    validate_feature_frame(transformed, min_rows=1, null_threshold=0.2)


def test_validate_feature_frame_raises_for_missing_column() -> None:
    invalid = _base_frame().drop(columns=["hit_rate"])
    with pytest.raises(ValueError):
        apply_feature_transformations(invalid)


def test_build_feature_blob_paths_generates_dated_and_latest() -> None:
    dated, latest = build_feature_blob_paths(
        latest_blob_path="features/daily_features_latest.parquet",
        run_time_utc=datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc),
    )
    assert dated == "features/daily_features_20260427.parquet"
    assert latest == "features/daily_features_latest.parquet"
