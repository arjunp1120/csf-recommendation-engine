"""Unit tests for plan Step 1.1 extended L2 features.

Covers the seven new builders added to ``heuristics_build.py``:

  * build_structure_mix
  * build_curve_preference
  * build_flow_polarity
  * build_follow_through_rate
  * build_size_profile_at_product (uses a fake ProductResolver duck)
  * build_per_product_liquidity
  * build_co_activity_matrix

Each test uses a small synthetic DataFrame so the expected output is
computable by hand. No DB, no network.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from csf_recommendation_engine.domain.heuristics_build import (
    build_co_activity_matrix,
    build_curve_preference,
    build_flow_polarity,
    build_follow_through_rate,
    build_per_product_liquidity,
    build_size_profile_at_product,
    build_structure_mix,
)


# ===========================================================================
# build_structure_mix
# ===========================================================================


class TestStructureMix:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"entity_id": "e1", "structure": "Flat Price"},
                {"entity_id": "e1", "structure": "Flat Price"},
                {"entity_id": "e1", "structure": "Flat Price"},
                {"entity_id": "e1", "structure": "Spread"},
                {"entity_id": "e1", "structure": "Butterfly"},
                {"entity_id": "e2", "structure": "Spread"},
                {"entity_id": "e2", "structure": "Spread"},
                {"entity_id": "e2", "structure": "Crack"},
                {"entity_id": "e2", "structure": "Crack"},
            ]
        )

    def test_ratios_sum_to_one_per_entity(self) -> None:
        out = build_structure_mix(self._df())
        for _, row in out.iterrows():
            assert sum(row["structure_mix"].values()) == pytest.approx(1.0)

    def test_e1_distribution(self) -> None:
        out = build_structure_mix(self._df())
        e1 = out.loc[out["entity_id"] == "e1", "structure_mix"].iloc[0]
        assert e1["Flat Price"] == pytest.approx(0.6)
        assert e1["Spread"] == pytest.approx(0.2)
        assert e1["Butterfly"] == pytest.approx(0.2)
        assert "Crack" not in e1

    def test_empty_df_returns_empty_frame(self) -> None:
        empty = pd.DataFrame(columns=["entity_id", "structure"])
        out = build_structure_mix(empty)
        assert out.empty
        assert set(out.columns) == {"entity_id", "structure_mix"}

    def test_missing_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="build_structure_mix"):
            build_structure_mix(pd.DataFrame({"entity_id": ["e1"]}))


# ===========================================================================
# build_curve_preference
# ===========================================================================


class TestCurvePreference:
    def _df(self) -> pd.DataFrame:
        # Trade date 2026-05-13. Front-month horizon = 2 (May/Jun/Jul).
        return pd.DataFrame(
            [
                {"entity_id": "e1", "instrument_name": "CL Jul26", "trade_date": date(2026, 5, 13)},  # +2 → front
                {"entity_id": "e1", "instrument_name": "CL Jun26", "trade_date": date(2026, 5, 13)},  # +1 → front
                {"entity_id": "e1", "instrument_name": "CL Dec26", "trade_date": date(2026, 5, 13)},  # +7 → deferred
                {"entity_id": "e1", "instrument_name": "WTI Cal 26 Strip", "trade_date": date(2026, 5, 13)},  # unparseable
                {"entity_id": "e2", "instrument_name": "CL Dec27", "trade_date": date(2026, 5, 13)},  # +19 → deferred
                {"entity_id": "e2", "instrument_name": "HO May26", "trade_date": date(2026, 5, 13)},  # 0 → front
            ]
        )

    def test_e1_two_of_three_classified_are_front(self) -> None:
        out = build_curve_preference(self._df(), front_month_horizon=2)
        e1 = out.loc[out["entity_id"] == "e1"].iloc[0]
        assert e1["trades_classified"] == 3  # strip excluded
        assert e1["front_month_ratio"] == pytest.approx(2 / 3)

    def test_e2_one_front_one_deferred(self) -> None:
        out = build_curve_preference(self._df(), front_month_horizon=2)
        e2 = out.loc[out["entity_id"] == "e2"].iloc[0]
        assert e2["trades_classified"] == 2
        assert e2["front_month_ratio"] == pytest.approx(0.5)

    def test_strip_excluded_from_denominator(self) -> None:
        """Instruments that can't be parsed to a contract month
        (Cal-year strips) drop out of the classification entirely."""
        df = pd.DataFrame([
            {"entity_id": "e1", "instrument_name": "WTI Cal 26 Strip", "trade_date": date(2026, 5, 13)},
        ])
        out = build_curve_preference(df)
        assert out.empty  # nothing parseable

    def test_symbol_grammar_fallback(self) -> None:
        """When the human form lacks a month but a CME symbol-style
        token is present, the symbol provides the front month."""
        df = pd.DataFrame([
            {"entity_id": "e1", "instrument_name": "CLN6", "trade_date": date(2026, 5, 13)},
        ])
        out = build_curve_preference(df, front_month_horizon=3)
        assert out["trades_classified"].iloc[0] == 1
        # CLN6 = July 2026, trade_date May 2026 → 2 months ahead → front
        assert out["front_month_ratio"].iloc[0] == 1.0


# ===========================================================================
# build_flow_polarity
# ===========================================================================


class TestFlowPolarity:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"entity_id": "e1", "product_name": "WTI Crude", "side": "BUY",  "quantity": 100.0, "trade_date": date(2026, 5, 10)},
            {"entity_id": "e1", "product_name": "WTI Crude", "side": "SELL", "quantity": 50.0,  "trade_date": date(2026, 5, 11)},
            {"entity_id": "e1", "product_name": "Heating Oil", "side": "SELL","quantity": 80.0, "trade_date": date(2026, 5, 12)},
            {"entity_id": "e2", "product_name": "WTI Crude", "side": "BUY",  "quantity": 50.0,  "trade_date": date(2026, 5, 13)},
            {"entity_id": "e2", "product_name": "WTI Crude", "side": "SELL", "quantity": 50.0,  "trade_date": date(2026, 5, 13)},
            # Out-of-window trade (excluded with as_of=2026-05-13, lookback=30)
            {"entity_id": "e1", "product_name": "WTI Crude", "side": "BUY",  "quantity": 99999.0, "trade_date": date(2024, 1, 1)},
        ])

    def test_e1_wti_polarity(self) -> None:
        out = build_flow_polarity(self._df(), lookback_days=30, as_of=date(2026, 5, 13))
        row = out[(out["entity_id"] == "e1") & (out["product_name"] == "WTI Crude")].iloc[0]
        # buy=100, sell=50 → net=+50, polarity = 50/150 ≈ 0.333
        assert row["buy_volume"] == 100.0
        assert row["sell_volume"] == 50.0
        assert row["net_volume"] == 50.0
        assert row["polarity"] == pytest.approx(50 / 150)

    def test_e1_heating_oil_one_sided(self) -> None:
        out = build_flow_polarity(self._df(), lookback_days=30, as_of=date(2026, 5, 13))
        row = out[(out["entity_id"] == "e1") & (out["product_name"] == "Heating Oil")].iloc[0]
        assert row["buy_volume"] == 0.0
        assert row["sell_volume"] == 80.0
        assert row["polarity"] == -1.0

    def test_e2_perfectly_balanced(self) -> None:
        out = build_flow_polarity(self._df(), lookback_days=30, as_of=date(2026, 5, 13))
        row = out[out["entity_id"] == "e2"].iloc[0]
        assert row["polarity"] == 0.0
        assert row["net_volume"] == 0.0

    def test_lookback_window_excludes_old(self) -> None:
        out = build_flow_polarity(self._df(), lookback_days=30, as_of=date(2026, 5, 13))
        row = out[(out["entity_id"] == "e1") & (out["product_name"] == "WTI Crude")].iloc[0]
        # The 99999-qty trade from 2024-01-01 is excluded
        assert row["buy_volume"] == 100.0

    def test_empty_df(self) -> None:
        empty = pd.DataFrame(columns=["entity_id", "product_name", "side", "quantity", "trade_date"])
        out = build_flow_polarity(empty, lookback_days=30, as_of=date(2026, 5, 13))
        assert out.empty


# ===========================================================================
# build_follow_through_rate
# ===========================================================================


class TestFollowThroughRate:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"entity_id": "e1", "product_name": "WTI Crude", "status": "Filled"},
            {"entity_id": "e1", "product_name": "WTI Crude", "status": "Filled"},
            {"entity_id": "e1", "product_name": "WTI Crude", "status": "Filled"},
            {"entity_id": "e1", "product_name": "WTI Crude", "status": "Filled"},
            {"entity_id": "e1", "product_name": "WTI Crude", "status": "Filled"},
            {"entity_id": "e1", "product_name": "WTI Crude", "status": "Cancelled"},
            {"entity_id": "e1", "product_name": "WTI Crude", "status": "Cancelled"},
            {"entity_id": "e1", "product_name": "WTI Crude", "status": "Working"},  # excluded — not final
            {"entity_id": "e2", "product_name": "Heating Oil", "status": "Cancelled"},
            {"entity_id": "e2", "product_name": "Heating Oil", "status": "Cancelled"},
            {"entity_id": "e2", "product_name": "Heating Oil", "status": "Cancelled"},
        ])

    def test_e1_wti_rate(self) -> None:
        out = build_follow_through_rate(self._df())
        row = out[(out["entity_id"] == "e1") & (out["product_name"] == "WTI Crude")].iloc[0]
        assert row["filled_count"] == 5
        assert row["aged_out_count"] == 2
        assert row["follow_through_rate"] == pytest.approx(5 / 7)

    def test_e2_all_aged_out_rate_zero(self) -> None:
        out = build_follow_through_rate(self._df())
        row = out[(out["entity_id"] == "e2") & (out["product_name"] == "Heating Oil")].iloc[0]
        assert row["filled_count"] == 0
        assert row["aged_out_count"] == 3
        assert row["follow_through_rate"] == 0.0

    def test_working_excluded_from_denominator(self) -> None:
        out = build_follow_through_rate(self._df())
        row = out[(out["entity_id"] == "e1") & (out["product_name"] == "WTI Crude")].iloc[0]
        # Working trade does not contribute to either count
        assert row["filled_count"] + row["aged_out_count"] == 7


# ===========================================================================
# build_size_profile_at_product
# ===========================================================================


class _FakeResolver:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def resolve(self, instrument_name: str):
        product = self._mapping.get(instrument_name)
        if product is None:
            return None
        return SimpleNamespace(product_name=product)


class TestSizeProfileAtProduct:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"entity_id": "e1", "instrument_name": "CL Jul26", "quantity": 50.0},
            {"entity_id": "e1", "instrument_name": "CL Aug26", "quantity": 60.0},
            {"entity_id": "e1", "instrument_name": "HO May26", "quantity": 30.0},
            {"entity_id": "e1", "instrument_name": "TOTALLY_UNKNOWN", "quantity": 99999.0},  # dropped
        ])

    def _resolver(self) -> _FakeResolver:
        return _FakeResolver({
            "CL Jul26": "WTI Crude",
            "CL Aug26": "WTI Crude",
            "HO May26": "Heating Oil",
        })

    def test_aggregates_per_product(self) -> None:
        out = build_size_profile_at_product(self._df(), self._resolver())
        wti = out[(out["entity_id"] == "e1") & (out["product_name"] == "WTI Crude")].iloc[0]
        assert wti["mean_size_at_product"] == pytest.approx(55.0)

    def test_unresolved_dropped(self) -> None:
        out = build_size_profile_at_product(self._df(), self._resolver())
        # Only WTI Crude and Heating Oil — no row for TOTALLY_UNKNOWN
        assert set(out["product_name"].unique()) == {"WTI Crude", "Heating Oil"}

    def test_stddev_filled_for_single_sample(self) -> None:
        out = build_size_profile_at_product(self._df(), self._resolver())
        ho = out[(out["entity_id"] == "e1") & (out["product_name"] == "Heating Oil")].iloc[0]
        assert ho["stddev_size_at_product"] == 0.0  # one sample → NaN → coerced to 0


# ===========================================================================
# build_per_product_liquidity
# ===========================================================================


class TestPerProductLiquidity:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"product_name": "WTI Crude", "trade_hour": 10, "quantity": 100.0, "trade_date": date(2026, 5, 13)},
            {"product_name": "WTI Crude", "trade_hour": 14, "quantity": 200.0, "trade_date": date(2026, 5, 13)},
            {"product_name": "WTI Crude", "trade_hour": 14, "quantity": 50.0,  "trade_date": date(2026, 5, 12)},
            {"product_name": "Heating Oil", "trade_hour": 9, "quantity": 75.0, "trade_date": date(2026, 5, 13)},
        ])

    def test_hourly_buckets(self) -> None:
        out = build_per_product_liquidity(self._df(), lookback_days=30, as_of=date(2026, 5, 13))
        wti_row = out[out["product_name"] == "WTI Crude"].iloc[0]
        assert wti_row["hourly_volume"][10] == 100.0
        assert wti_row["hourly_volume"][14] == 250.0
        # Untouched hours stay at 0
        assert wti_row["hourly_volume"][0] == 0.0

    def test_24_element_lists(self) -> None:
        out = build_per_product_liquidity(self._df(), lookback_days=30, as_of=date(2026, 5, 13))
        for _, row in out.iterrows():
            assert len(row["hourly_volume"]) == 24

    def test_outside_window_excluded(self) -> None:
        df = pd.DataFrame([
            {"product_name": "WTI Crude", "trade_hour": 10, "quantity": 100.0, "trade_date": date(2025, 1, 1)},
        ])
        out = build_per_product_liquidity(df, lookback_days=30, as_of=date(2026, 5, 13))
        assert out.empty


# ===========================================================================
# build_co_activity_matrix
# ===========================================================================


class TestCoActivityMatrix:
    def _df(self) -> pd.DataFrame:
        # Week 2026-W19 (May 4-10): e1 + e2 both trade WTI
        # Week 2026-W20 (May 11-17): e1 alone in WTI, e3 alone in HO
        return pd.DataFrame([
            {"entity_id": "e1", "product_name": "WTI Crude", "trade_date": date(2026, 5,  6)},
            {"entity_id": "e1", "product_name": "WTI Crude", "trade_date": date(2026, 5,  7)},  # dup-suppressed
            {"entity_id": "e2", "product_name": "WTI Crude", "trade_date": date(2026, 5,  8)},
            {"entity_id": "e1", "product_name": "WTI Crude", "trade_date": date(2026, 5, 13)},
            {"entity_id": "e3", "product_name": "Heating Oil", "trade_date": date(2026, 5, 14)},
        ])

    def test_symmetric_matrix(self) -> None:
        ordering, matrix = build_co_activity_matrix(self._df())
        assert (matrix == matrix.T).all()

    def test_pair_count_for_shared_week(self) -> None:
        ordering, matrix = build_co_activity_matrix(self._df())
        idx = {e: i for i, e in enumerate(ordering)}
        # e1+e2 share exactly one (product, week) bucket — WTI W19
        assert matrix[idx["e1"], idx["e2"]] == 1
        assert matrix[idx["e2"], idx["e1"]] == 1
        # e1+e3 share none
        assert matrix[idx["e1"], idx["e3"]] == 0

    def test_diagonal_counts_buckets_active(self) -> None:
        ordering, matrix = build_co_activity_matrix(self._df())
        idx = {e: i for i, e in enumerate(ordering)}
        # e1 is active in WTI W19 + WTI W20 = 2 buckets
        assert matrix[idx["e1"], idx["e1"]] == 2
        # e2 is active in WTI W19 only
        assert matrix[idx["e2"], idx["e2"]] == 1
        # e3 is active in HO W20 only
        assert matrix[idx["e3"], idx["e3"]] == 1

    def test_explicit_entity_universe(self) -> None:
        """When a universe is supplied, the matrix is sized to it and
        rows/cols appear in the supplied order — even for entities with
        no trades."""
        universe = ["e1", "e2", "e3", "e_no_trades"]
        ordering, matrix = build_co_activity_matrix(self._df(), entity_universe=universe)
        assert ordering == universe
        assert matrix.shape == (4, 4)
        # e_no_trades row + column are all zero
        assert matrix[3].sum() == 0
        assert matrix[:, 3].sum() == 0

    def test_empty_df_returns_zero_matrix(self) -> None:
        empty = pd.DataFrame(columns=["entity_id", "product_name", "trade_date"])
        ordering, matrix = build_co_activity_matrix(empty, entity_universe=["a", "b"])
        assert ordering == ["a", "b"]
        assert matrix.shape == (2, 2)
        assert matrix.sum() == 0
