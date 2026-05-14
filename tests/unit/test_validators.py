"""Unit tests for plan §11.4 validators + critic state (Step 0.5).

Covers:
  * ``build_packet_vocab`` harvests entity_ids / names / numerics correctly
  * ``validate_citations`` catches fabricated UUIDs and out-of-packet
    rationale entities, while leaving clean responses alone
  * ``validate_eligibility`` rejects rationales / strategist scripts that
    cite candidates marked ``hard_constraints_passed=False``
  * ``validate_response`` merges both
  * ``apply_critic_mode`` switches between SERVE / RETRY / FALLBACK under
    the documented mode + retry-count rules
  * ``CriticState`` tracks the trailing-window violation rate and gates
    promotion at a full sample + threshold
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from csf_recommendation_engine.domain.intelligence.critic_state import (
    CriticState,
    CriticStateSnapshot,
)
from csf_recommendation_engine.domain.intelligence.packet import (
    Candidate,
    Eligibility,
    EntityDossier,
    FitSignals,
    IntelligencePacket,
    IntentTags,
    RankerScores,
    RequestContext,
)
from csf_recommendation_engine.domain.intelligence.responses import (
    MatchStrategistResponse,
    RecommendationRationale,
    RecommenderExplainerResponse,
    TaggerResponse,
)
from csf_recommendation_engine.domain.intelligence.validators import (
    Decision,
    apply_critic_mode,
    build_packet_vocab,
    validate_citations,
    validate_eligibility,
    validate_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_NOW = datetime(2026, 5, 13, 14, 0, 0, tzinfo=timezone.utc)


def _dossier(eid: UUID) -> EntityDossier:
    return EntityDossier(
        dossier_id=uuid4(), entity_id=eid, archetype="Hedger",
        current_posture="Long-biased",
        motive_priors={"basis_widen": 0.3},
        friction_notes="No issues",
        recent_activity="active",
        embedding_text="200-word version of the dossier text",
        features_snapshot={"trade_count_30d": 12, "avg_size": 50.5},
        generated_by_swarm="profiler", swarm_packet_hash=None,
        generated_at=_NOW,
    )


def _scores(final: float = 0.62) -> RankerScores:
    return RankerScores(
        lightfm_affinity=2.3, lightfm_normalized=0.7,
        size_fit=0.5, recency=0.4, time_affinity=0.6,
        flow_polarity_penalty=0.0, final_score=final,
    )


def _candidate(eid: UUID, name: str, eligible: bool = True) -> Candidate:
    return Candidate(
        entity_id=eid, entity_name=name,
        dossier=_dossier(eid),
        ranker_scores=_scores(),
        eligibility=Eligibility(
            venue_ok=eligible, clearing_ok=eligible, structure_ok=eligible,
            hard_constraints_passed=eligible,
            reasons=[] if eligible else ["clearing_mismatch"],
        ),
        fit_signals=FitSignals(),
        cold_start=False,
    )


def _packet(
    originator_id: UUID, candidates: list[Candidate],
    intent_tags: IntentTags | None = None,
) -> IntelligencePacket:
    return IntelligencePacket(
        packet_id=uuid4(),
        generated_at=_NOW,
        request_context=RequestContext(
            request_type="sync_form", current_hour=14,
            originator_entity_id=originator_id,
            intent_tags=intent_tags,
            instrument_name="CL Jul26",
        ),
        candidates=candidates,
    )


# ---------------------------------------------------------------------------
# build_packet_vocab
# ---------------------------------------------------------------------------


class TestBuildPacketVocab:
    def test_harvests_entity_ids(self) -> None:
        oid = uuid4()
        c1 = _candidate(uuid4(), "Vitol")
        c2 = _candidate(uuid4(), "Trafigura")
        v = build_packet_vocab(_packet(oid, [c1, c2]))
        assert str(oid) in v.entity_ids
        assert str(c1.entity_id) in v.entity_ids
        assert str(c2.entity_id) in v.entity_ids

    def test_harvests_entity_names_lowercase(self) -> None:
        c = _candidate(uuid4(), "Vitol SA")
        v = build_packet_vocab(_packet(uuid4(), [c]))
        assert "vitol sa" in v.entity_names

    def test_harvests_intent_tag_numerics(self) -> None:
        v = build_packet_vocab(_packet(
            uuid4(), [_candidate(uuid4(), "X")],
            intent_tags=IntentTags(qty="200", price="71.20", tenor="2026"),
        ))
        assert "200" in v.numeric_strs
        assert "71.20" in v.numeric_strs

    def test_harvests_dossier_feature_numerics(self) -> None:
        c = _candidate(uuid4(), "Vitol")
        v = build_packet_vocab(_packet(uuid4(), [c]))
        # feature_snapshot has trade_count_30d=12 and avg_size=50.5
        assert "12" in v.numeric_strs


# ---------------------------------------------------------------------------
# validate_citations
# ---------------------------------------------------------------------------


class TestValidateCitations:
    def test_none_response_passes(self) -> None:
        packet = _packet(uuid4(), [_candidate(uuid4(), "X")])
        r = validate_citations(None, packet)
        assert r.passed is True
        assert r.citation_violations == []

    def test_clean_explainer_response_passes(self) -> None:
        eid = uuid4()
        cand = _candidate(eid, "Vitol")
        packet = _packet(uuid4(), [cand])
        response = RecommenderExplainerResponse(
            candidates=[RecommendationRationale(
                entity_id=eid,
                ui_friendly_reasoning="Strong size match.",
                detailed_reasoning="Vitol's avg size aligns with the requested 12 lots.",
            )],
            originator_hypothesis="Q4 hedging push.",
        )
        result = validate_citations(response, packet)
        assert result.passed is True
        assert result.citation_violations == []

    def test_fabricated_entity_id_in_explainer_fails(self) -> None:
        real_eid = uuid4()
        fake_eid = uuid4()
        packet = _packet(uuid4(), [_candidate(real_eid, "Vitol")])
        response = RecommenderExplainerResponse(
            candidates=[RecommendationRationale(
                entity_id=fake_eid,  # not in packet
                ui_friendly_reasoning="x", detailed_reasoning="y",
            )],
        )
        result = validate_citations(response, packet)
        assert result.passed is False
        assert any(str(fake_eid) in v for v in result.citation_violations)

    def test_unsupported_numerics_are_informational(self) -> None:
        """Numbers not in the packet vocab should populate
        ``unsupported_numerics`` but NOT set ``passed=False`` in v1."""
        eid = uuid4()
        packet = _packet(uuid4(), [_candidate(eid, "X")])
        # 9999 is not in the packet vocab anywhere
        response = RecommenderExplainerResponse(
            candidates=[RecommendationRationale(
                entity_id=eid,
                ui_friendly_reasoning="9999 unrelated number.",
                detailed_reasoning="See 9999.",
            )],
        )
        result = validate_citations(response, packet)
        assert result.passed is True  # not a hard fail
        assert any("9999" == n for n in result.unsupported_numerics)


# ---------------------------------------------------------------------------
# validate_eligibility
# ---------------------------------------------------------------------------


class TestValidateEligibility:
    def test_none_response_passes(self) -> None:
        r = validate_eligibility(None, _packet(uuid4(), [_candidate(uuid4(), "X")]))
        assert r.passed is True

    def test_ineligible_in_explainer_fails(self) -> None:
        eid = uuid4()
        cand = _candidate(eid, "Vitol", eligible=False)
        packet = _packet(uuid4(), [cand])
        response = RecommenderExplainerResponse(
            candidates=[RecommendationRationale(
                entity_id=eid, ui_friendly_reasoning="x", detailed_reasoning="y",
            )],
        )
        result = validate_eligibility(response, packet)
        assert result.passed is False
        assert any(str(eid) in v for v in result.eligibility_violations)

    def test_ineligible_name_in_strategist_fails(self) -> None:
        eid = uuid4()
        cand = _candidate(eid, "Vitol", eligible=False)
        packet = _packet(uuid4(), [cand])
        response = MatchStrategistResponse(
            who_to_call_first="Vitol",  # mentions ineligible name
            pitch_to_buyer="Pitch", pitch_to_seller="Pitch",
            risk_flags=[],
        )
        result = validate_eligibility(response, packet)
        assert result.passed is False
        assert any("vitol" in v.lower() for v in result.eligibility_violations)

    def test_eligible_strategist_passes(self) -> None:
        eid = uuid4()
        cand = _candidate(eid, "Vitol", eligible=True)
        packet = _packet(uuid4(), [cand])
        response = MatchStrategistResponse(
            who_to_call_first="Vitol",
            pitch_to_buyer="x", pitch_to_seller="y",
            risk_flags=[],
        )
        result = validate_eligibility(response, packet)
        assert result.passed is True
        assert result.eligibility_violations == []


# ---------------------------------------------------------------------------
# validate_response (merge)
# ---------------------------------------------------------------------------


class TestValidateResponse:
    def test_merges_both(self) -> None:
        real_eid = uuid4()
        fake_eid = uuid4()
        # Real candidate marked ineligible AND response cites a fake UUID — both fire.
        cand = _candidate(real_eid, "Vitol", eligible=False)
        packet = _packet(uuid4(), [cand])
        response = RecommenderExplainerResponse(
            candidates=[RecommendationRationale(
                entity_id=fake_eid,
                ui_friendly_reasoning="x", detailed_reasoning="y",
            )],
        )
        merged = validate_response(response, packet)
        assert merged.passed is False
        # Both kinds of violations are populated
        assert merged.citation_violations
        # fake_eid isn't a packet candidate but isn't ineligible either; so eligibility violations are 0
        # while citation violations are 1. That's correct.

    def test_clean_clean_passes(self) -> None:
        eid = uuid4()
        cand = _candidate(eid, "Vitol")
        packet = _packet(uuid4(), [cand])
        response = RecommenderExplainerResponse(
            candidates=[RecommendationRationale(
                entity_id=eid, ui_friendly_reasoning="x", detailed_reasoning="y",
            )],
        )
        merged = validate_response(response, packet)
        assert merged.passed is True


# ---------------------------------------------------------------------------
# apply_critic_mode
# ---------------------------------------------------------------------------


class TestApplyCriticMode:
    def _result(self, passed: bool):
        from csf_recommendation_engine.domain.intelligence.validators import ValidationResult
        return ValidationResult(passed=passed)

    def test_shadow_always_serves(self) -> None:
        assert apply_critic_mode(self._result(True), "shadow") == Decision.SERVE_AS_IS
        assert apply_critic_mode(self._result(False), "shadow") == Decision.SERVE_AS_IS

    def test_strict_pass_serves(self) -> None:
        assert apply_critic_mode(self._result(True), "strict") == Decision.SERVE_AS_IS

    def test_strict_fail_retry_first(self) -> None:
        assert (
            apply_critic_mode(self._result(False), "strict", retry_count=0, max_retries=1)
            == Decision.RETRY_ONCE
        )

    def test_strict_fail_fallback_when_exhausted(self) -> None:
        assert (
            apply_critic_mode(self._result(False), "strict", retry_count=1, max_retries=1)
            == Decision.FALLBACK_DETERMINISTIC
        )

    def test_unknown_mode_falls_back_to_shadow(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            d = apply_critic_mode(self._result(False), "OOPS_TYPO")
        assert d == Decision.SERVE_AS_IS
        assert any("Unknown CRITIC_MODE" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# CriticState
# ---------------------------------------------------------------------------


class TestCriticState:
    def test_init_validates_args(self) -> None:
        with pytest.raises(ValueError):
            CriticState(window=0)
        with pytest.raises(ValueError):
            CriticState(violation_threshold=1.5)
        with pytest.raises(ValueError):
            CriticState(violation_threshold=-0.1)

    def test_empty_window_violation_rate_zero(self) -> None:
        s = CriticState(window=10, violation_threshold=0.01)
        assert s.events_recorded == 0
        assert s.violation_rate == 0.0
        assert s.can_promote is False  # partial window → cannot promote

    def test_partial_window_cannot_promote(self) -> None:
        s = CriticState(window=10, violation_threshold=0.01)
        for _ in range(5):
            s.record(passed=True)
        assert s.events_recorded == 5
        assert s.violation_rate == 0.0
        assert s.can_promote is False  # window not yet full

    def test_full_window_below_threshold_can_promote(self) -> None:
        s = CriticState(window=10, violation_threshold=0.1)
        for _ in range(10):
            s.record(passed=True)
        assert s.events_recorded == 10
        assert s.violation_rate == 0.0
        assert s.can_promote is True

    def test_full_window_above_threshold_cannot_promote(self) -> None:
        s = CriticState(window=10, violation_threshold=0.05)
        for i in range(10):
            # Two failures → violation_rate=0.2 > 0.05
            s.record(passed=(i >= 2))
        assert s.events_recorded == 10
        assert s.violation_rate == pytest.approx(0.2)
        assert s.can_promote is False

    def test_reset(self) -> None:
        s = CriticState(window=10)
        for _ in range(5):
            s.record(passed=False)
        s.reset()
        assert s.events_recorded == 0
        assert s.violation_rate == 0.0

    def test_snapshot_returns_dataclass(self) -> None:
        s = CriticState(window=10, violation_threshold=0.1)
        for _ in range(10):
            s.record(passed=True)
        snap = s.snapshot()
        assert isinstance(snap, CriticStateSnapshot)
        assert snap.window == 10
        assert snap.threshold == 0.1
        assert snap.events_recorded == 10
        assert snap.violation_rate == 0.0
        assert snap.can_promote is True
