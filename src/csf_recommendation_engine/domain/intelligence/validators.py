"""Python-side validators for swarm responses.

Plan §11.4 specifies three independent checks that run **in our code**,
regardless of whatever critic stage the DAF-side swarm includes
internally (defense in depth):

1. **Schema validation** — Pydantic `extra="forbid"` already rejects
   unknown keys at construction. Surfaced as a `ValidationResult`
   here only when the swarm-call layer (Step 0.6) catches the
   `ValidationError`; this module's `validate_*` helpers assume the
   response object is already a parsed Pydantic model.
2. **Citation fidelity** (`validate_citations`) — every named entity_id,
   entity_name, instrument/product, or numeric quantity in the swarm
   response must trace to a field present in the input packet.
3. **Eligibility** (`validate_eligibility`) — the response must not
   propose actions involving candidates the deterministic Eligibility
   Auditor flagged as `hard_constraints_passed=False`.

`apply_critic_mode` maps a `ValidationResult` to a `Decision` based on
the configured mode (shadow vs strict). Retry counting is the caller's
responsibility — the decision returned here is the local action, not
the full state machine.

v1 scope notes:
* Numeric-value checking is **deliberately narrow** for v1 to avoid
  false positives. We extract decimal numbers from the response and
  check them against a vocabulary collected from the packet's
  RankerScores, FeatureSnapshot, and a few other obvious places, but
  we do NOT treat unknown numbers as automatic violations — we surface
  them as `unsupported_numerics` informational, not as a hard fail.
* Strict numeric enforcement is a v1+ enhancement once we have data
  on false-positive rates from shadow mode.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from csf_recommendation_engine.domain.intelligence.packet import (
    IntelligencePacket,
)
from csf_recommendation_engine.domain.intelligence.responses import (
    CoverageCoachResponse,
    MarketReaderResponse,
    MatchStrategistResponse,
    ProfilerResponse,
    RecommendationRationale,
    RecommenderExplainerResponse,
    TaggerResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result & decision types
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of running validators against a swarm response.

    `passed = True` means **every** check the validator ran returned
    clean. A single violation in either citations or eligibility sets
    `passed = False`. The detail lists are populated regardless so
    callers can log structured context even in shadow mode.
    """

    passed: bool = True
    citation_violations: list[str] = field(default_factory=list)
    eligibility_violations: list[str] = field(default_factory=list)
    unsupported_numerics: list[str] = field(default_factory=list)  # informational, not hard fail in v1
    notes: list[str] = field(default_factory=list)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Return a combined result. Used when running multiple validators."""
        return ValidationResult(
            passed=self.passed and other.passed,
            citation_violations=[*self.citation_violations, *other.citation_violations],
            eligibility_violations=[*self.eligibility_violations, *other.eligibility_violations],
            unsupported_numerics=[*self.unsupported_numerics, *other.unsupported_numerics],
            notes=[*self.notes, *other.notes],
        )


class Decision(str, Enum):
    """What the caller should do with the swarm response after running
    validators."""

    SERVE_AS_IS = "serve_as_is"
    RETRY_ONCE = "retry_once"
    FALLBACK_DETERMINISTIC = "fallback_deterministic"


# ---------------------------------------------------------------------------
# Regex constants shared by vocabulary harvesting and response scanning
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
# Numeric with optional decimal. Matches 71, 71.20, 200, 21480, etc.
_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


# ---------------------------------------------------------------------------
# Packet vocabulary extraction
# ---------------------------------------------------------------------------


@dataclass
class PacketVocab:
    """Token sets harvested from the packet for citation checking."""

    entity_ids: set[str] = field(default_factory=set)
    entity_names: set[str] = field(default_factory=set)
    product_names: set[str] = field(default_factory=set)
    instrument_names: set[str] = field(default_factory=set)
    numeric_strs: set[str] = field(default_factory=set)


def _harvest_numeric_strs(node: Any, accumulator: set[str]) -> None:
    """Walk a JSON-serializable structure collecting string forms of
    every numeric leaf. We collect multiple textual variants of each
    number (e.g., 71.20 → '71.20', '71.2') because LLM output may use
    either."""
    if isinstance(node, (int, float)):
        accumulator.add(str(node))
        # Common float formatting alternatives
        if isinstance(node, float):
            # Trim trailing zeros: 71.20 -> 71.2
            stripped = f"{node:g}"
            accumulator.add(stripped)
            # Two-decimal variant: 71.2 -> 71.20
            try:
                accumulator.add(f"{float(node):.2f}")
            except (ValueError, TypeError):
                pass
    elif isinstance(node, dict):
        for v in node.values():
            _harvest_numeric_strs(v, accumulator)
    elif isinstance(node, list):
        for v in node:
            _harvest_numeric_strs(v, accumulator)


def build_packet_vocab(packet: IntelligencePacket) -> PacketVocab:
    """Collect the allowed-citation vocabulary from the packet.

    Anything in the returned `PacketVocab` is a fact that a swarm
    response may legitimately cite. Anything else may be a hallucination.
    """
    vocab = PacketVocab()

    # Originator + candidate entity_ids and names
    if packet.request_context.originator_entity_id is not None:
        vocab.entity_ids.add(str(packet.request_context.originator_entity_id))
    for cand in packet.candidates:
        vocab.entity_ids.add(str(cand.entity_id))
        if cand.entity_name:
            vocab.entity_names.add(cand.entity_name.lower())
        if cand.dossier is not None:
            _harvest_numeric_strs(cand.dossier.features_snapshot, vocab.numeric_strs)
            _harvest_numeric_strs(cand.ranker_scores.model_dump(), vocab.numeric_strs)

    # Originator dossier numerics
    if packet.originator_dossier is not None:
        vocab.entity_ids.add(str(packet.originator_dossier.entity_id))
        _harvest_numeric_strs(packet.originator_dossier.features_snapshot, vocab.numeric_strs)

    # Recent feedback entity_ids
    for note in packet.recent_feedback:
        if note.buyer_entity_id is not None:
            vocab.entity_ids.add(str(note.buyer_entity_id))
        if note.seller_entity_id is not None:
            vocab.entity_ids.add(str(note.seller_entity_id))
        if note.product_name:
            vocab.product_names.add(note.product_name.lower())

    # Market briefings — products
    for briefing in packet.market_briefings:
        vocab.product_names.add(briefing.product_name.lower())

    # Request-context instrument / product hints
    if packet.request_context.instrument_name:
        vocab.instrument_names.add(packet.request_context.instrument_name.lower())
    if packet.request_context.intent_tags is not None:
        tags = packet.request_context.intent_tags
        if tags.product:
            vocab.product_names.add(tags.product.lower())
        # intent_tags qty/price/tenor are string-typed (they may be
        # ranges like "100-200" or "71.20") but commonly carry numeric
        # data. Pull every numeric token out of each tag string and
        # add it to the numeric vocab so a response that mentions the
        # same number is considered cited.
        for raw in (tags.qty, tags.price, tags.tenor):
            if raw is None:
                continue
            for m in _NUMERIC_RE.finditer(raw):
                vocab.numeric_strs.add(m.group(0))
    if packet.request_context.quantity is not None:
        _harvest_numeric_strs(packet.request_context.quantity, vocab.numeric_strs)
    # Request-context current_hour as well — a response that says
    # "during this hour (14)" should be supported.
    _harvest_numeric_strs(packet.request_context.current_hour, vocab.numeric_strs)

    # Exposure context numerics
    _harvest_numeric_strs(packet.exposure_context.model_dump(), vocab.numeric_strs)

    return vocab


# ---------------------------------------------------------------------------
# Response text extraction
# ---------------------------------------------------------------------------


def _collect_text_fields(response: Any) -> list[str]:
    """Pull every free-form string field out of a swarm response so we
    can scan it for hallucinations. Operates on the response model's
    JSON dump to be shape-agnostic."""
    if response is None:
        return []
    payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
    chunks: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            chunks.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(payload)
    return chunks


# ---------------------------------------------------------------------------
# Citation-fidelity validator
# ---------------------------------------------------------------------------


def validate_citations(response: Any, packet: IntelligencePacket) -> ValidationResult:
    """Verify every named entity_id, entity_name, instrument/product
    and numeric value in the response traces to the packet.

    v1 heuristics (deliberately narrow):
    * UUIDs in response must appear in the packet vocabulary.
    * Lowercase substring matches of `entity_names` / `product_names` /
      `instrument_names` are considered supported.
    * Numeric tokens unknown to the packet vocabulary are surfaced as
      `unsupported_numerics` (informational only — do not fail the
      result in v1).
    """
    result = ValidationResult(passed=True)
    if response is None:
        return result

    vocab = build_packet_vocab(packet)
    text_chunks = _collect_text_fields(response)

    # UUID check
    for chunk in text_chunks:
        for m in _UUID_RE.finditer(chunk):
            uid = m.group(0)
            # Normalize lowercase for comparison since UUID() always lowercases.
            try:
                normalized = str(UUID(uid))
            except ValueError:
                continue
            if normalized not in vocab.entity_ids:
                violation = f"Response cited entity_id {normalized!r} not present in packet."
                result.citation_violations.append(violation)
                result.passed = False

    # Numeric check (informational only in v1).
    # Strip UUIDs from the text first so their digit chunks aren't
    # counted as numeric tokens (otherwise the four hex-digit groups
    # in a UUID look like '11111111', '1111', etc. and pollute the
    # unsupported_numerics list with false positives).
    for chunk in text_chunks:
        chunk_without_uuids = _UUID_RE.sub(" ", chunk)
        for m in _NUMERIC_RE.finditer(chunk_without_uuids):
            token = m.group(0)
            if token in vocab.numeric_strs:
                continue
            # Try matching with stripped trailing zeros (e.g., '71.2' vs '71.20')
            try:
                f = float(token)
                if str(f) in vocab.numeric_strs or f"{f:g}" in vocab.numeric_strs or f"{f:.2f}" in vocab.numeric_strs:
                    continue
            except ValueError:
                pass
            result.unsupported_numerics.append(token)

    # Entity-name / product-name / instrument-name checks happen
    # against the union of those sets, but only when the response is a
    # type that we know to check. We avoid false positives on words
    # like "ICE" or "Crude" by NOT flagging unmatched name-shaped
    # tokens — we only check that names *mentioned* in the packet
    # vocabulary are not contradicted. For v1, name-checking is
    # advisory only; we record a note if any candidate entity_name
    # from the packet is absent from a per-candidate rationale in
    # RecommenderExplainerResponse (cheap structural check).
    if isinstance(response, RecommenderExplainerResponse):
        packet_candidate_ids = {str(c.entity_id) for c in packet.candidates}
        response_candidate_ids = {str(r.entity_id) for r in response.candidates}
        unknown = response_candidate_ids - packet_candidate_ids
        for uid in unknown:
            violation = (
                f"Explainer response includes rationale for entity_id {uid!r} "
                f"that is not in the packet's candidate list."
            )
            result.citation_violations.append(violation)
            result.passed = False

    return result


# ---------------------------------------------------------------------------
# Eligibility validator
# ---------------------------------------------------------------------------


def validate_eligibility(response: Any, packet: IntelligencePacket) -> ValidationResult:
    """Verify the response does not propose actions involving candidates
    that failed the deterministic Eligibility Auditor.

    Today this is meaningful for:
    * `RecommenderExplainerResponse` — rationale must not be served for
      a candidate whose `eligibility.hard_constraints_passed=False`
      (such candidates should already be filtered upstream, but we
      double-check).
    * `MatchStrategistResponse` — `who_to_call_first` and the pitches
      must not reference an ineligible counterparty.

    Other response types pass trivially.
    """
    result = ValidationResult(passed=True)
    if response is None:
        return result

    ineligible_ids = {
        str(c.entity_id) for c in packet.candidates if not c.eligibility.hard_constraints_passed
    }
    ineligible_names = {
        c.entity_name.lower()
        for c in packet.candidates
        if not c.eligibility.hard_constraints_passed and c.entity_name
    }

    if isinstance(response, RecommenderExplainerResponse):
        for r in response.candidates:
            if str(r.entity_id) in ineligible_ids:
                violation = (
                    f"Explainer rationale provided for entity_id {r.entity_id} "
                    f"which is marked ineligible in the packet."
                )
                result.eligibility_violations.append(violation)
                result.passed = False

    if isinstance(response, MatchStrategistResponse):
        # Scan free-text pitches for any ineligible name or UUID.
        full_text = " ".join(
            [
                response.who_to_call_first,
                response.pitch_to_buyer,
                response.pitch_to_seller,
                response.residual_handling or "",
                response.brokerage_economic_note or "",
                *response.risk_flags,
            ]
        )
        for ineligible_uid in ineligible_ids:
            if ineligible_uid in full_text:
                result.eligibility_violations.append(
                    f"Strategist script references ineligible entity_id {ineligible_uid!r}."
                )
                result.passed = False
        lowered = full_text.lower()
        for ineligible_name in ineligible_names:
            if ineligible_name and ineligible_name in lowered:
                result.eligibility_violations.append(
                    f"Strategist script references ineligible entity name {ineligible_name!r}."
                )
                result.passed = False

    return result


# ---------------------------------------------------------------------------
# Combined validation entry point
# ---------------------------------------------------------------------------


def validate_response(response: Any, packet: IntelligencePacket) -> ValidationResult:
    """Run citation-fidelity + eligibility validators and return their
    merged result. Schema validation is assumed to have already
    succeeded at Pydantic-parse time (caller's responsibility).
    """
    citations = validate_citations(response, packet)
    eligibility = validate_eligibility(response, packet)
    merged = citations.merge(eligibility)
    return merged


# ---------------------------------------------------------------------------
# Critic-mode decision
# ---------------------------------------------------------------------------


def apply_critic_mode(
    result: ValidationResult,
    mode: str,
    retry_count: int = 0,
    max_retries: int = 1,
) -> Decision:
    """Map a `ValidationResult` to a `Decision` under the configured
    critic mode.

    * ``shadow``: every result is served as-is, regardless of violations
      (callers MUST log the structured violations even so).
    * ``strict``: passing results are served; failing results retry
      once (`retry_count == 0`), then fall back to deterministic
      rendering (`retry_count >= max_retries`).

    Unknown modes default to ``shadow`` with a warning logged so a
    typo can't accidentally activate strict behavior.
    """
    normalized = (mode or "").strip().lower()

    if normalized not in {"shadow", "strict"}:
        logger.warning(
            "Unknown CRITIC_MODE %r; falling back to 'shadow'", mode,
        )
        normalized = "shadow"

    if normalized == "shadow":
        return Decision.SERVE_AS_IS

    # Strict mode
    if result.passed:
        return Decision.SERVE_AS_IS
    if retry_count < max_retries:
        return Decision.RETRY_ONCE
    return Decision.FALLBACK_DETERMINISTIC
