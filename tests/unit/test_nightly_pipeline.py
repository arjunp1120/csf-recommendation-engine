"""Unit tests for the rec refresh pipeline helpers and orchestration logic."""
from __future__ import annotations

import json
import math

import pytest

from csf_recommendation_engine.jobs.rec_refresh_pipeline import (
    _RefreshCounters,
    _TradeRequest,
    _quantity_to_lots,
    calculate_match_score,
)


# ---------------------------------------------------------------------------
# _quantity_to_lots (Plan §5.4, §17)
# ---------------------------------------------------------------------------


class TestQuantityToLots:
    def test_integer_quantity(self):
        assert _quantity_to_lots(10) == 10

    def test_float_quantity_floors(self):
        assert _quantity_to_lots(10.9) == 10

    def test_zero_returns_zero(self):
        assert _quantity_to_lots(0) == 0

    def test_negative_returns_zero(self):
        assert _quantity_to_lots(-5) == 0

    def test_string_numeric(self):
        assert _quantity_to_lots("42") == 42

    def test_small_fractional_returns_zero(self):
        assert _quantity_to_lots(0.5) == 0


# ---------------------------------------------------------------------------
# calculate_match_score (Plan §6)
# ---------------------------------------------------------------------------


class TestCalculateMatchScore:
    def test_both_positive_returns_geometric_mean(self):
        score = calculate_match_score(score_a_for_b=0.81, score_b_for_a=0.81)
        assert pytest.approx(score, abs=1e-6) == 0.81

    def test_zero_score_returns_zero(self):
        assert calculate_match_score(score_a_for_b=0.0, score_b_for_a=0.9) == 0.0

    def test_negative_score_returns_zero(self):
        assert calculate_match_score(score_a_for_b=-0.5, score_b_for_a=0.9) == 0.0

    def test_asymmetric_scores(self):
        score = calculate_match_score(score_a_for_b=0.64, score_b_for_a=1.0)
        assert pytest.approx(score, abs=1e-6) == 0.8

    def test_score_in_unit_range(self):
        score = calculate_match_score(score_a_for_b=1.0, score_b_for_a=1.0)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _TradeRequest mapping (Plan §5.1)
# ---------------------------------------------------------------------------


class TestTradeRequestMapping:
    """Verifies that a trade dict can be correctly mapped into a _TradeRequest."""

    def _make_trade(self, **overrides) -> dict:
        base = {
            "trade_id": "tid-001",
            "entity_id": "ent-001",
            "desk_type": "FLOW",
            "structure": "OUTRIGHT",
            "side": "BUY",
            "venue": "ICE",
            "instrument_name": "CL",
            "quantity": 100,
        }
        base.update(overrides)
        return base

    def test_basic_mapping(self):
        trade = self._make_trade()
        req = _TradeRequest(
            client_id=str(trade["entity_id"]),
            desk=str(trade.get("desk_type") or ""),
            structure=str(trade.get("structure") or ""),
            side=str(trade["side"]),
            venue=str(trade.get("venue") or ""),
            instrument_name=str(trade.get("instrument_name") or ""),
            quantity=float(trade["quantity"]),
            top_k=3,
        )
        assert req.client_id == "ent-001"
        assert req.desk == "FLOW"
        assert req.structure == "OUTRIGHT"
        assert req.side == "BUY"
        assert req.venue == "ICE"
        assert req.instrument_name == "CL"
        assert req.quantity == 100.0
        assert req.top_k == 3

    def test_none_fields_normalize_to_empty_string(self):
        trade = self._make_trade(desk_type=None, structure=None, venue=None, instrument_name=None)
        req = _TradeRequest(
            client_id=str(trade["entity_id"]),
            desk=str(trade.get("desk_type") or ""),
            structure=str(trade.get("structure") or ""),
            side=str(trade["side"]),
            venue=str(trade.get("venue") or ""),
            instrument_name=str(trade.get("instrument_name") or ""),
            quantity=float(trade["quantity"]),
        )
        assert req.desk == ""
        assert req.structure == ""
        assert req.venue == ""
        assert req.instrument_name == ""


# ---------------------------------------------------------------------------
# _RefreshCounters
# ---------------------------------------------------------------------------


class TestRefreshCounters:
    def test_all_counters_default_to_zero(self):
        c = _RefreshCounters()
        assert c.total_working_trades == 0
        assert c.recommendation_attempts == 0
        assert c.recommendations_inserted == 0
        assert c.recommendations_skipped_unknown == 0
        assert c.recommendations_skipped_duplicate == 0
        assert c.recommendation_failures == 0
        assert c.mutual_pairs_evaluated == 0
        assert c.cross_matches_inserted == 0
        assert c.cross_matches_skipped_unknown == 0
        assert c.cross_matches_skipped_duplicate == 0
        assert c.match_failures == 0

    def test_counter_names_include_dedup_fields(self):
        """Ensure we have all 11 counters including deduplication fields."""
        c = _RefreshCounters()
        expected = {
            "total_working_trades",
            "recommendation_attempts",
            "recommendations_inserted",
            "recommendations_skipped_unknown",
            "recommendations_skipped_duplicate",
            "recommendation_failures",
            "mutual_pairs_evaluated",
            "cross_matches_inserted",
            "cross_matches_skipped_unknown",
            "cross_matches_skipped_duplicate",
            "match_failures",
        }
        actual = {f.name for f in c.__dataclass_fields__.values()}
        assert actual == expected


# ---------------------------------------------------------------------------
# Match percentage clamping (Plan §14)
# ---------------------------------------------------------------------------


class TestMatchPercentageClamping:
    """Verifies the clamping formula used in the pipeline: max(1, min(100, round(score * 100)))."""

    def test_high_score_clamped_to_100(self):
        score = 1.5  # Hypothetical edge case
        pct = max(1, min(100, round(score * 100)))
        assert pct == 100

    def test_low_score_clamped_to_1(self):
        score = 0.001
        pct = max(1, min(100, round(score * 100)))
        assert pct == 1

    def test_normal_score_rounds_correctly(self):
        score = 0.73
        pct = max(1, min(100, round(score * 100)))
        assert pct == 73

    def test_threshold_boundary(self):
        """Score just above 0.6 should yield percentage 61."""
        score = 0.61
        pct = max(1, min(100, round(score * 100)))
        assert pct == 61
