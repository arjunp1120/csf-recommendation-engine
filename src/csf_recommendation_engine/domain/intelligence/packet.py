"""Intelligence Packet — canonical input shape fed to every swarm.

The Intelligence Packet is the contract between deterministic Python and
externally configured DAF swarms.  Every served output (recommendation
serve, match) carries the `packet_hash` of the packet that produced it so
the input can be reconstructed during replay.

See plan §11.3 / §12 for the design rationale.

This module defines:

* The packet's component models (`RequestContext`, `IntentTags`,
  `MarketBriefing`, `EntityDossier`, `RankerScores`, `Eligibility`,
  `FitSignals`, `Candidate`, `ExposureContext`, `FeedbackNote`,
  `PolicyBundle`).
* The top-level `IntelligencePacket` model.

`schema_version` defaults to ``"1.0.0"``; bump on any breaking change.

Embedding-related fields are intentionally absent — pgvector is deferred
for v1 (see plan §17 / §15.9).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared config — every model is strict about extra keys (rejects unknowns)
# ---------------------------------------------------------------------------

_BASE_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False)


# ---------------------------------------------------------------------------
# Tagger output / intent shape (S1 IOI Tagger)
# ---------------------------------------------------------------------------


class IntentTags(BaseModel):
    """Structured tags extracted from a free-text IOI by the S1 Tagger,
    or supplied directly by the broker UI on `/ioi/accept`.

    Fields are all optional strings — the Tagger may extract a subset.
    `qty` and `price` are strings rather than numerics because both can
    legitimately be ranges (e.g., ``"100-200"``, ``"71.20"``, ``"71-72"``).
    """

    model_config = _BASE_MODEL_CONFIG

    side: str | None = Field(default=None, description="Buy / Sell / Either / None")
    product: str | None = Field(default=None, description="WTI / Brent / RBOB / HO / NG / etc.")
    qty: str | None = Field(default=None, description="Integer or range (lots)")
    tenor: str | None = Field(default=None, description="Tenor / contract month")
    price: str | None = Field(default=None, description="Float or range")
    qualifier: str | None = Field(default=None, description="at / better / worse")
    urgency: str | None = Field(default=None, description="Low / Med / High")
    sentiment: str | None = Field(default=None, description="Bullish / Bearish / Neutral")


# ---------------------------------------------------------------------------
# Request context (top of the packet)
# ---------------------------------------------------------------------------


class RequestContext(BaseModel):
    """Per-request envelope: which surface produced this packet, who is
    the originator, what tags drove the intent, what hour is it."""

    model_config = _BASE_MODEL_CONFIG

    request_type: str = Field(
        ...,
        description="One of 'sync_form' | 'ioi_text' | 'ioi_accept' | 'coach_draft' | 'profiler_nightly' | 'market_reader_refresh' | 'match_pipeline' (extensible).",
    )
    originator_entity_id: UUID | None = Field(
        default=None,
        description="Entity initiating the request. Null for nightly profiler / market briefing rebuilds.",
    )
    intent_tags: IntentTags | None = Field(
        default=None,
        description="Finalized tags. Null on /recommend (form-driven; details in originator_dossier instead).",
    )
    current_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour-of-day at request time (server local). Used by hourly_ratios reranker term.",
    )
    instrument_name: str | None = Field(default=None)
    venue: str | None = Field(default=None)
    desk: str | None = Field(default=None)
    structure: str | None = Field(default=None)
    side: str | None = Field(default=None)
    quantity: float | None = Field(default=None)


# ---------------------------------------------------------------------------
# Market briefing (S3 output, persisted in market_briefings)
# ---------------------------------------------------------------------------


class MarketBriefing(BaseModel):
    """One market briefing for a product.  Mirror of `market_briefings`
    rows (plan §8.3) plus the `swarm_packet_hash` for replay."""

    model_config = _BASE_MODEL_CONFIG

    briefing_id: UUID
    product_name: str
    regime_label: str | None = None
    recent_move_summary: str | None = None
    liquidity_assessment: str | None = None
    broker_book_posture: str | None = None
    generated_at: datetime
    swarm_packet_hash: str | None = None


# ---------------------------------------------------------------------------
# Dossier (S2 output, persisted in entity_dossiers)
# ---------------------------------------------------------------------------


class EntityDossier(BaseModel):
    """One entity's dossier.  Mirror of `entity_dossiers` rows (plan §8.2)
    minus the deferred `embedding` column."""

    model_config = _BASE_MODEL_CONFIG

    dossier_id: UUID
    entity_id: UUID
    archetype: str
    current_posture: str | None = None
    motive_priors: dict[str, Any] | None = None
    friction_notes: str | None = None
    recent_activity: str | None = None
    embedding_text: str
    features_snapshot: dict[str, Any]
    generated_by_swarm: str | None = None
    swarm_packet_hash: str | None = None
    generated_at: datetime


# ---------------------------------------------------------------------------
# Per-candidate metadata
# ---------------------------------------------------------------------------


class RankerScores(BaseModel):
    """Component scores attached to every candidate, post deterministic
    ranking.  All scores are min-max normalized to ``[0, 1]`` except
    `lightfm_affinity` (raw model output)."""

    model_config = _BASE_MODEL_CONFIG

    lightfm_affinity: float
    lightfm_normalized: float = Field(ge=0.0, le=1.0)
    size_fit: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    time_affinity: float = Field(ge=0.0, le=1.0)
    flow_polarity_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float


class Eligibility(BaseModel):
    """Hard-constraint result from the Eligibility Auditor.  The LLM
    layer is not allowed to override these."""

    model_config = _BASE_MODEL_CONFIG

    venue_ok: bool
    clearing_ok: bool
    structure_ok: bool
    hard_constraints_passed: bool
    reasons: list[str] = Field(default_factory=list, description="Empty if passed.")


class FitSignals(BaseModel):
    """Match-side fit signals (only populated on the match pipeline path;
    on /recommend they are zero/None)."""

    model_config = _BASE_MODEL_CONFIG

    stated_intent_match: bool = False
    mutual_score: float | None = None
    brokerage_offload_bonus: float = Field(
        default=0.0,
        description="Always 0 in v1 — exposure is a soft signal, not a numeric ranker term (plan §10.4).",
    )


class Candidate(BaseModel):
    """One candidate counterparty in the packet."""

    model_config = _BASE_MODEL_CONFIG

    entity_id: UUID
    entity_name: str
    dossier: EntityDossier | None = None
    ranker_scores: RankerScores
    eligibility: Eligibility
    fit_signals: FitSignals
    cold_start: bool = Field(
        default=False,
        description="True if this entity has no trade_history (12 of 40 entities at v1). UI may badge accordingly.",
    )


# ---------------------------------------------------------------------------
# Brokerage exposure (soft signal, plan §4.1 / §10.4 / §17)
# ---------------------------------------------------------------------------


class ExposureContext(BaseModel):
    """Brokerage-side exposure (firm book), passed into LLM context only."""

    model_config = _BASE_MODEL_CONFIG

    gross_exposure: float | None = None
    net_exposure: float | None = None
    delta_equivalent: float | None = None
    net_delta_by_product: dict[str, float] = Field(default_factory=dict)
    as_of: datetime | None = None


# ---------------------------------------------------------------------------
# Recent feedback (plan §10.4 dampening + §10.2 feedback-aware dedup input)
# ---------------------------------------------------------------------------


class FeedbackNote(BaseModel):
    """One recent feedback row from cross_block_feedback."""

    model_config = _BASE_MODEL_CONFIG

    feedback_id: UUID
    match_id: UUID
    buyer_entity_id: UUID | None = None
    seller_entity_id: UUID | None = None
    product_name: str | None = None
    is_accurate: bool
    feedback_notes: str | None = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Policy bundle (weights, version, cutoffs)
# ---------------------------------------------------------------------------


class PolicyBundle(BaseModel):
    """Policies the deterministic ranker applied while building the
    packet.  Snapshotted into the packet so replay reproduces the same
    ranking under the same policies."""

    model_config = _BASE_MODEL_CONFIG

    variant_id: str | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    version: str = "1.0.0"
    cutoffs: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level packet
# ---------------------------------------------------------------------------


PACKET_SCHEMA_VERSION = "1.0.0"


class IntelligencePacket(BaseModel):
    """Canonical input bundle to any swarm.  Versioned, hashed,
    immutable per call.  See plan §11.3 / §12.

    `packet_hash` is computed via :func:`hashing.compute_packet_hash`,
    which excludes envelope fields (`packet_id`, `packet_hash`,
    `generated_at`) so two semantically identical packets hash the same
    regardless of when they were materialized.
    """

    model_config = _BASE_MODEL_CONFIG

    packet_id: UUID
    packet_hash: str = Field(default="")
    schema_version: str = Field(default=PACKET_SCHEMA_VERSION)
    generated_at: datetime

    request_context: RequestContext
    market_briefings: list[MarketBriefing] = Field(default_factory=list)
    originator_dossier: EntityDossier | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    exposure_context: ExposureContext = Field(default_factory=ExposureContext)
    recent_feedback: list[FeedbackNote] = Field(default_factory=list)
    policies: PolicyBundle = Field(default_factory=PolicyBundle)
