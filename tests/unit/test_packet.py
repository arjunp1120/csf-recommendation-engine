"""Unit tests for the Intelligence Packet Pydantic models (plan Step 0.4).

Covers: field defaults, ``extra="forbid"`` strictness, range validation
on ``current_hour`` (0..23) and the normalized score fields (0..1), the
schema version constant, and the top-level ``IntelligencePacket`` envelope.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from csf_recommendation_engine.domain.intelligence.packet import (
    PACKET_SCHEMA_VERSION,
    Candidate,
    Eligibility,
    EntityDossier,
    ExposureContext,
    FeedbackNote,
    FitSignals,
    IntelligencePacket,
    IntentTags,
    MarketBriefing,
    PolicyBundle,
    RankerScores,
    RequestContext,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 5, 13, 14, 0, 0, tzinfo=timezone.utc)


def _ranker_scores(**overrides: object) -> RankerScores:
    base: dict[str, object] = {
        "lightfm_affinity": 2.3,
        "lightfm_normalized": 0.7,
        "size_fit": 0.5,
        "recency": 0.4,
        "time_affinity": 0.6,
        "flow_polarity_penalty": 0.0,
        "final_score": 0.62,
    }
    base.update(overrides)
    return RankerScores(**base)  # type: ignore[arg-type]


def _eligibility(passed: bool = True) -> Eligibility:
    return Eligibility(
        venue_ok=passed, clearing_ok=passed, structure_ok=passed,
        hard_constraints_passed=passed,
        reasons=[] if passed else ["clearing_mismatch"],
    )


def _candidate(eid: UUID | None = None, **overrides: object) -> Candidate:
    base = {
        "entity_id": eid or uuid4(),
        "entity_name": "Vitol",
        "dossier": None,
        "ranker_scores": _ranker_scores(),
        "eligibility": _eligibility(),
        "fit_signals": FitSignals(),
        "cold_start": False,
    }
    base.update(overrides)
    return Candidate(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# IntentTags
# ---------------------------------------------------------------------------


class TestIntentTags:
    def test_all_fields_default_none(self) -> None:
        t = IntentTags()
        assert t.side is None
        assert t.product is None
        assert t.qty is None
        assert t.tenor is None
        assert t.price is None

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            IntentTags(side="Buy", unknown_key="boom")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# RequestContext
# ---------------------------------------------------------------------------


class TestRequestContext:
    def test_minimum_fields(self) -> None:
        rc = RequestContext(request_type="sync_form", current_hour=14)
        assert rc.request_type == "sync_form"
        assert rc.current_hour == 14
        assert rc.intent_tags is None
        assert rc.originator_entity_id is None

    def test_current_hour_lower_bound(self) -> None:
        RequestContext(request_type="x", current_hour=0)  # OK
        with pytest.raises(ValidationError):
            RequestContext(request_type="x", current_hour=-1)

    def test_current_hour_upper_bound(self) -> None:
        RequestContext(request_type="x", current_hour=23)  # OK
        with pytest.raises(ValidationError):
            RequestContext(request_type="x", current_hour=24)

    def test_required_request_type(self) -> None:
        with pytest.raises(ValidationError):
            RequestContext(current_hour=14)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# RankerScores
# ---------------------------------------------------------------------------


class TestRankerScores:
    def test_happy_path(self) -> None:
        rs = _ranker_scores()
        assert rs.final_score == 0.62
        assert 0.0 <= rs.size_fit <= 1.0

    def test_normalized_score_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            _ranker_scores(size_fit=-0.1)

    def test_normalized_score_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            _ranker_scores(recency=1.1)

    def test_lightfm_affinity_is_unbounded(self) -> None:
        """Raw LightFM scores can be any real value — only the normalized
        variants are bounded."""
        rs = _ranker_scores(lightfm_affinity=-9.9)
        assert rs.lightfm_affinity == -9.9


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_passed_default_empty_reasons(self) -> None:
        e = _eligibility(passed=True)
        assert e.hard_constraints_passed is True
        assert e.reasons == []

    def test_failed_carries_reasons(self) -> None:
        e = _eligibility(passed=False)
        assert e.hard_constraints_passed is False
        assert "clearing_mismatch" in e.reasons


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


class TestCandidate:
    def test_minimum_fields(self) -> None:
        c = _candidate()
        assert c.cold_start is False
        assert c.dossier is None

    def test_cold_start_flag(self) -> None:
        c = _candidate(cold_start=True)
        assert c.cold_start is True


# ---------------------------------------------------------------------------
# IntelligencePacket
# ---------------------------------------------------------------------------


class TestIntelligencePacket:
    def _packet(self, **overrides: object) -> IntelligencePacket:
        base: dict[str, object] = {
            "packet_id": uuid4(),
            "generated_at": _now(),
            "request_context": RequestContext(request_type="sync_form", current_hour=14),
        }
        base.update(overrides)
        return IntelligencePacket(**base)  # type: ignore[arg-type]

    def test_schema_version_constant(self) -> None:
        assert PACKET_SCHEMA_VERSION == "1.0.0"

    def test_minimum_fields(self) -> None:
        p = self._packet()
        assert p.schema_version == PACKET_SCHEMA_VERSION
        assert p.packet_hash == ""
        assert p.candidates == []
        assert p.market_briefings == []
        assert isinstance(p.exposure_context, ExposureContext)
        assert isinstance(p.policies, PolicyBundle)

    def test_with_candidates_and_dossier(self) -> None:
        eid = uuid4()
        dossier = EntityDossier(
            dossier_id=uuid4(), entity_id=eid, archetype="Hedger",
            current_posture="Long-biased",
            embedding_text="200-word",
            features_snapshot={"trade_count_30d": 12},
            generated_at=_now(),
        )
        cand = _candidate(eid=eid, dossier=dossier)
        p = self._packet(candidates=[cand], originator_dossier=dossier)
        assert len(p.candidates) == 1
        assert p.originator_dossier is dossier

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            self._packet(unknown_root_field="boom")
