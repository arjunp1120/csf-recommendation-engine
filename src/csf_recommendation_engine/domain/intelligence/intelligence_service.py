"""IntelligenceService — the single entry point for every swarm call.

This file replaces both ``intelligence_layer.py`` (DAF SDK based) and
``new_intelligence_service.py`` (incomplete A2A port). Per plan §11
the service owns:

* Building a per-swarm message payload from an Intelligence Packet.
* POSTing to the DAF execute endpoint via :class:`DAFTransport`.
* Parsing the agent text out of the response.
* Pydantic-validating the parsed JSON against the per-swarm response
  model in :mod:`responses`.
* Logging ``swarm_id``, ``packet_hash``, response, and timing.

Validators (citation-fidelity, eligibility) live in
:mod:`validators` — the *caller* invokes them after receiving the
parsed response, per the defense-in-depth pattern in plan §11.4.

Embedding generation is DEFERRED for v1 (pgvector unavailable; see plan
§17). The service does **not** expose a ``generate_embedding`` method
and does **not** import an embedding provider.

All methods are async and target latency budgets stated in plan §4.1 /
§11.2:
* ``tag_ioi`` — 1.5-2.5 s (on the broker-blocking path for /recommend/ioi)
* ``profile_entity`` — nightly batch (no pressure)
* ``read_market`` — 15-min cache refresh (moderate)
* ``explain_recommendations`` — 1.5-2.5 s (sync /recommend path)
* ``strategize_match`` — 5-30 s background
* ``draft_coach_outreach`` — 5-10 s on broker click
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from csf_recommendation_engine.core.config import Settings
from csf_recommendation_engine.core.observability import (
    M_SWARM_CALL_COVERAGE_COACH,
    M_SWARM_CALL_INSTRUMENT_RESOLVER,
    M_SWARM_CALL_MARKET_READER,
    M_SWARM_CALL_MATCH_STRATEGIST,
    M_SWARM_CALL_PROFILER,
    M_SWARM_CALL_RECOMMENDER_EXPLAINER,
    M_SWARM_CALL_TAGGER,
    M_SWARM_LATENCY_COVERAGE_COACH,
    M_SWARM_LATENCY_INSTRUMENT_RESOLVER,
    M_SWARM_LATENCY_MARKET_READER,
    M_SWARM_LATENCY_MATCH_STRATEGIST,
    M_SWARM_LATENCY_PROFILER,
    M_SWARM_LATENCY_RECOMMENDER_EXPLAINER,
    M_SWARM_LATENCY_TAGGER,
    MetricsRecorder,
)
from csf_recommendation_engine.domain.intelligence.packet import IntelligencePacket
from csf_recommendation_engine.domain.intelligence.responses import (
    CoverageCoachResponse,
    InstrumentResolutionResponse,
    MarketReaderResponse,
    MatchStrategistResponse,
    ProfilerResponse,
    RecommenderExplainerResponse,
    TaggerResponse,
)
from csf_recommendation_engine.domain.intelligence.transport import (
    DAFTransport,
    TransportError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text extraction helpers (markdown-fence + JSON parse, shared across methods)
# ---------------------------------------------------------------------------

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```$")


def _strip_markdown_fences(raw: str) -> str:
    """Strip leading ```json / ``` fence and trailing ``` from an LLM
    response payload. Operates on a stripped copy; returns the cleaned
    inner content."""
    cleaned = raw.strip()
    cleaned = _FENCE_OPEN.sub("", cleaned)
    cleaned = _FENCE_CLOSE.sub("", cleaned)
    return cleaned.strip()


def _parse_json_from_text(text: str, swarm_label: str) -> dict[str, Any] | None:
    """Best-effort JSON parse of the agent's text reply. Handles
    markdown code fences. Returns None on failure (and logs)."""
    cleaned = _strip_markdown_fences(text)
    if not cleaned:
        logger.warning("Empty payload after stripping fences from %s", swarm_label)
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse %s response as JSON: %s (preview=%r)",
            swarm_label, exc, text[:300],
        )
        return None
    if not isinstance(parsed, dict):
        logger.warning(
            "%s returned a non-object JSON payload (type=%s)",
            swarm_label, type(parsed).__name__,
        )
        return None
    return parsed


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class IntelligenceService:
    """Async client for every externally configured DAF swarm.

    Each method:

    1. Looks up the target id from ``settings``; logs + returns None
       if it isn't configured yet.
    2. Constructs the user message (packet-JSON for swarms; plain text
       for the Tagger).
    3. POSTs via :class:`DAFTransport`.
    4. Strips markdown fences and JSON-parses the agent's reply.
    5. Pydantic-validates against the per-swarm response model.
    6. Increments call/latency metrics on the supplied
       :class:`MetricsRecorder`.

    Returns either the validated Pydantic model on success or None on
    any failure (transport error, parse failure, schema mismatch).
    Caller is responsible for running the citation/eligibility
    validators from :mod:`validators` against the packet that fed the
    call.
    """

    def __init__(self, settings: Settings, metrics: MetricsRecorder | None = None) -> None:
        self._settings = settings
        self._metrics = metrics
        self._transport = DAFTransport(settings)

    async def aclose(self) -> None:
        await self._transport.aclose()

    # ------------------------------------------------------------------
    # Internal: shared call wrapper
    # ------------------------------------------------------------------

    async def _call_swarm(
        self,
        *,
        target_id: str,
        swarm_label: str,
        message: str,
        packet_hash: str | None,
        metric_call: str,
        metric_latency: str,
    ) -> dict[str, Any] | None:
        """Run a swarm/agent call end-to-end up to the JSON-parse step.

        Returns the parsed JSON dict on success, None on any failure.
        The Pydantic-validation step happens per-swarm in the public
        methods so we can return a typed response.
        """
        if not target_id:
            logger.warning(
                "%s called but its DAF id is not configured; skipping.",
                swarm_label,
            )
            return None

        if self._metrics is not None:
            self._metrics.increment(metric_call)

        t0 = time.perf_counter()
        try:
            result = await self._transport.execute_agent(target_id, message)
        except TransportError as exc:
            logger.exception(
                "%s transport failed (target_id=%s, packet_hash=%s): %s",
                swarm_label, target_id, packet_hash, exc,
            )
            return None
        finally:
            if self._metrics is not None:
                self._metrics.record_timing(metric_latency, time.perf_counter() - t0)

        parsed = _parse_json_from_text(result.response_text, swarm_label)
        if parsed is None:
            return None

        logger.info(
            "%s completed (target_id=%s, packet_hash=%s, response_chars=%d)",
            swarm_label, target_id, packet_hash, len(result.response_text),
        )
        return parsed

    # ------------------------------------------------------------------
    # S1 — IOI Tagger
    # ------------------------------------------------------------------

    async def tag_ioi(self, ioi_text: str) -> TaggerResponse | None:
        """Send a free-text IOI to the Tagger agent and return the
        parsed tags. Used by `/recommend/ioi` (parse-only)."""
        if not ioi_text or not ioi_text.strip():
            return None

        message = f'Here is the input IOI text:\n"""{ioi_text}"""'
        parsed = await self._call_swarm(
            target_id=self._settings.daf_tagger_agent_id,
            swarm_label="S1 IOI Tagger",
            message=message,
            packet_hash=None,
            metric_call=M_SWARM_CALL_TAGGER,
            metric_latency=M_SWARM_LATENCY_TAGGER,
        )
        if parsed is None:
            return None
        return _validate_response_model(parsed, TaggerResponse, "S1 IOI Tagger")

    # ------------------------------------------------------------------
    # S2 — Entity Profiler
    # ------------------------------------------------------------------

    async def profile_entity(self, packet: IntelligencePacket) -> ProfilerResponse | None:
        """Run the Profiler swarm against the packet. Used by the
        nightly dossier job and on-demand 'hot' refreshes."""
        parsed = await self._call_swarm(
            target_id=self._settings.daf_profiler_swarm_id,
            swarm_label="S2 Profiler",
            message=_serialize_packet_for_swarm(packet),
            packet_hash=packet.packet_hash or None,
            metric_call=M_SWARM_CALL_PROFILER,
            metric_latency=M_SWARM_LATENCY_PROFILER,
        )
        if parsed is None:
            return None
        return _validate_response_model(parsed, ProfilerResponse, "S2 Profiler")

    # ------------------------------------------------------------------
    # S3 — Market Reader
    # ------------------------------------------------------------------

    async def read_market(self, packet: IntelligencePacket) -> MarketReaderResponse | None:
        """Run the Market Reader swarm against the packet."""
        parsed = await self._call_swarm(
            target_id=self._settings.daf_market_reader_swarm_id,
            swarm_label="S3 Market Reader",
            message=_serialize_packet_for_swarm(packet),
            packet_hash=packet.packet_hash or None,
            metric_call=M_SWARM_CALL_MARKET_READER,
            metric_latency=M_SWARM_LATENCY_MARKET_READER,
        )
        if parsed is None:
            return None
        return _validate_response_model(parsed, MarketReaderResponse, "S3 Market Reader")

    # ------------------------------------------------------------------
    # S4 — Recommender Explainer
    # ------------------------------------------------------------------

    async def explain_recommendations(
        self, packet: IntelligencePacket
    ) -> RecommenderExplainerResponse | None:
        """Run the Recommender Explainer swarm against the packet.
        Sync /recommend's narration step."""
        parsed = await self._call_swarm(
            target_id=self._settings.daf_recommender_explainer_swarm_id,
            swarm_label="S4 Recommender Explainer",
            message=_serialize_packet_for_swarm(packet),
            packet_hash=packet.packet_hash or None,
            metric_call=M_SWARM_CALL_RECOMMENDER_EXPLAINER,
            metric_latency=M_SWARM_LATENCY_RECOMMENDER_EXPLAINER,
        )
        if parsed is None:
            return None
        return _validate_response_model(
            parsed, RecommenderExplainerResponse, "S4 Recommender Explainer"
        )

    # ------------------------------------------------------------------
    # S5 — Match Strategist
    # ------------------------------------------------------------------

    async def strategize_match(
        self, packet: IntelligencePacket
    ) -> MatchStrategistResponse | None:
        """Run the Match Strategist swarm in the background match
        pipeline; produces a structured negotiation script."""
        parsed = await self._call_swarm(
            target_id=self._settings.daf_match_strategist_swarm_id,
            swarm_label="S5 Match Strategist",
            message=_serialize_packet_for_swarm(packet),
            packet_hash=packet.packet_hash or None,
            metric_call=M_SWARM_CALL_MATCH_STRATEGIST,
            metric_latency=M_SWARM_LATENCY_MATCH_STRATEGIST,
        )
        if parsed is None:
            return None
        return _validate_response_model(parsed, MatchStrategistResponse, "S5 Match Strategist")

    # ------------------------------------------------------------------
    # S6 — Coverage Coach
    # ------------------------------------------------------------------

    async def draft_coach_outreach(
        self, packet: IntelligencePacket
    ) -> CoverageCoachResponse | None:
        """Run the Coverage Coach swarm — produces a broker-editable
        outreach script for a chosen recommendation/match."""
        parsed = await self._call_swarm(
            target_id=self._settings.daf_coverage_coach_swarm_id,
            swarm_label="S6 Coverage Coach",
            message=_serialize_packet_for_swarm(packet),
            packet_hash=packet.packet_hash or None,
            metric_call=M_SWARM_CALL_COVERAGE_COACH,
            metric_latency=M_SWARM_LATENCY_COVERAGE_COACH,
        )
        if parsed is None:
            return None
        return _validate_response_model(parsed, CoverageCoachResponse, "S6 Coverage Coach")

    # ------------------------------------------------------------------
    # Instrument Resolver (single agent — Step 0.10 seeding script)
    # ------------------------------------------------------------------

    async def resolve_instrument(
        self, instrument_name: str, symbol: str | None = None
    ) -> InstrumentResolutionResponse | None:
        """Map one free-text ``instrument_name`` (and optional CME-style
        ``symbol``) to a canonical product via the externally configured
        Instrument Resolver agent (plan Step 0.10).

        The user message is the JSON-encoded pair the agent's system
        instructions expect — this keeps prompt shape under operator
        control on the DAF side.
        """
        if not instrument_name or not instrument_name.strip():
            return None

        payload = {"instrument_name": instrument_name, "symbol": symbol}
        message = json.dumps(payload, ensure_ascii=False)

        parsed = await self._call_swarm(
            target_id=self._settings.daf_instrument_resolver_agent_id,
            swarm_label="Instrument Resolver",
            message=message,
            packet_hash=None,
            metric_call=M_SWARM_CALL_INSTRUMENT_RESOLVER,
            metric_latency=M_SWARM_LATENCY_INSTRUMENT_RESOLVER,
        )
        if parsed is None:
            return None
        return _validate_response_model(
            parsed, InstrumentResolutionResponse, "Instrument Resolver"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _validate_response_model(parsed: dict[str, Any], model_cls: Any, swarm_label: str):
    """Pydantic-parse `parsed` into `model_cls` or log + return None on
    schema mismatch (extra/missing keys, wrong types)."""
    try:
        return model_cls.model_validate(parsed)
    except ValidationError as exc:
        logger.warning(
            "%s response failed schema validation against %s: %s",
            swarm_label, model_cls.__name__, exc,
        )
        return None


def _serialize_packet_for_swarm(packet: IntelligencePacket) -> str:
    """Render the packet as a JSON string suitable for passing to a
    swarm as the user message. Uses Pydantic's JSON-mode dump so UUIDs
    and datetimes serialize as strings.

    Swarms are externally configured to expect this shape; they parse
    it back into structured input internally. The deterministic JSON
    rendering used by hashing is NOT required here — swarms don't hash;
    only our replay store hashes.
    """
    payload = packet.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False)
