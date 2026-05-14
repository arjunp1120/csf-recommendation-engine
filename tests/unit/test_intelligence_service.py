"""Unit tests for :class:`IntelligenceService` (plan Step 0.6 + Step 0.10).

The DAF transport is mocked so no network calls happen. Each test
verifies one of:

  * happy-path: agent returns valid JSON → service returns the typed
    Pydantic response model
  * empty target_id → returns None (no transport call, no crash)
  * transport raises ``TransportError`` → service returns None
  * markdown-fenced JSON is stripped before parsing
  * schema mismatch → returns None (Pydantic ValidationError caught)
  * metrics counters / latency timings increment as documented
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from csf_recommendation_engine.core.config import Settings
from csf_recommendation_engine.core.observability import (
    M_SWARM_CALL_INSTRUMENT_RESOLVER,
    M_SWARM_CALL_TAGGER,
    M_SWARM_LATENCY_INSTRUMENT_RESOLVER,
    M_SWARM_LATENCY_TAGGER,
    MetricsRecorder,
)
from csf_recommendation_engine.domain.intelligence.intelligence_service import (
    IntelligenceService,
)
from csf_recommendation_engine.domain.intelligence.responses import (
    InstrumentResolutionResponse,
    TaggerResponse,
)
from csf_recommendation_engine.domain.intelligence.transport import (
    ExecuteResult,
    TransportError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides: object) -> Settings:
    """Build a Settings instance without loading .env so tests don't
    depend on the developer's local environment. The default values for
    every field are documented in core/config.py."""
    base: dict[str, object] = {
        "DAF_BASE_URL": "http://test.local",
        "DAF_API_KEY": "test-key",
        "DAF_TAGGER_AGENT_ID": "tagger-agent",
        "DAF_PROFILER_SWARM_ID": "profiler-swarm",
        "DAF_MARKET_READER_SWARM_ID": "market-swarm",
        "DAF_RECOMMENDER_EXPLAINER_SWARM_ID": "explainer-swarm",
        "DAF_MATCH_STRATEGIST_SWARM_ID": "strategist-swarm",
        "DAF_COVERAGE_COACH_SWARM_ID": "coach-swarm",
        "DAF_INSTRUMENT_RESOLVER_AGENT_ID": "resolver-agent",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


def _exec_result(text: str) -> ExecuteResult:
    return ExecuteResult(
        response_text=text,
        session_id=None,
        tool_calls=[],
        reasoning_steps=[],
        pending_approval=None,
        raw_body={"response": text},
    )


def _mock_service(
    response_text: str | None,
    *,
    raise_transport_error: bool = False,
    **settings_overrides: object,
) -> tuple[IntelligenceService, AsyncMock, MetricsRecorder]:
    """Construct an IntelligenceService and replace its transport with
    an AsyncMock pre-loaded with the given response."""
    metrics = MetricsRecorder()
    service = IntelligenceService(_settings(**settings_overrides), metrics=metrics)
    mock_exec = AsyncMock()
    if raise_transport_error:
        mock_exec.side_effect = TransportError("boom")
    elif response_text is not None:
        mock_exec.return_value = _exec_result(response_text)
    # Replace the transport's method, not the whole transport — leaves
    # aclose() etc. intact.
    service._transport.execute_agent = mock_exec
    return service, mock_exec, metrics


# ---------------------------------------------------------------------------
# tag_ioi
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTagIoi:
    async def test_happy_path(self) -> None:
        payload = {"side": "Buy", "product": "WTI", "qty": "50", "price": "71.20"}
        service, mock_exec, metrics = _mock_service(json.dumps(payload))

        out = await service.tag_ioi("Vitol buying 50 lots WTI Mar26 at 71.20")
        assert isinstance(out, TaggerResponse)
        assert out.side == "Buy"
        assert out.product == "WTI"
        assert out.qty == "50"
        assert out.price == "71.20"

        mock_exec.assert_awaited_once()
        # Metrics incremented
        assert metrics.counters.get(M_SWARM_CALL_TAGGER) == 1
        assert M_SWARM_LATENCY_TAGGER in metrics.timings
        await service.aclose()

    async def test_empty_text_returns_none_no_transport_call(self) -> None:
        service, mock_exec, _ = _mock_service('{"side": "Buy"}')
        out = await service.tag_ioi("")
        assert out is None
        mock_exec.assert_not_awaited()
        await service.aclose()

    async def test_whitespace_only_returns_none(self) -> None:
        service, mock_exec, _ = _mock_service('{"side": "Buy"}')
        assert await service.tag_ioi("   ") is None
        mock_exec.assert_not_awaited()
        await service.aclose()

    async def test_unset_target_id_returns_none(self) -> None:
        service, mock_exec, _ = _mock_service(
            '{"side": "Buy"}', DAF_TAGGER_AGENT_ID="",
        )
        out = await service.tag_ioi("some text")
        assert out is None
        mock_exec.assert_not_awaited()  # _call_swarm bails before transport
        await service.aclose()

    async def test_markdown_fenced_response(self) -> None:
        fenced = '```json\n{"side": "Buy", "product": "WTI"}\n```'
        service, _, _ = _mock_service(fenced)
        out = await service.tag_ioi("text")
        assert out is not None and out.side == "Buy" and out.product == "WTI"
        await service.aclose()

    async def test_invalid_json_returns_none(self) -> None:
        service, _, _ = _mock_service("not valid json at all {{{")
        out = await service.tag_ioi("text")
        assert out is None
        await service.aclose()

    async def test_schema_violation_returns_none(self) -> None:
        """Tagger response includes a key not in the model (extra='forbid')."""
        service, _, _ = _mock_service('{"side": "Buy", "totally_unknown_field": "x"}')
        out = await service.tag_ioi("text")
        assert out is None
        await service.aclose()

    async def test_transport_error_returns_none(self) -> None:
        service, _, metrics = _mock_service(None, raise_transport_error=True)
        out = await service.tag_ioi("text")
        assert out is None
        # Latency timing still recorded even on failure (finally block)
        assert M_SWARM_LATENCY_TAGGER in metrics.timings
        await service.aclose()


# ---------------------------------------------------------------------------
# resolve_instrument (Step 0.10)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResolveInstrument:
    async def test_happy_path(self) -> None:
        payload = {
            "product_name": "WTI Crude",
            "product_family": "Crude",
            "structure_type": "Outright",
            "contract_month": "2026-07",
            "expiry_date": None,
            "confidence": 0.95,
            "reasoning": "CL Jul26 ticker matches WTI Crude.",
        }
        service, mock_exec, metrics = _mock_service(json.dumps(payload))
        out = await service.resolve_instrument("CL Jul26", symbol="CLN6")
        assert isinstance(out, InstrumentResolutionResponse)
        assert out.product_name == "WTI Crude"
        assert out.structure_type == "Outright"
        assert out.contract_month == "2026-07"

        mock_exec.assert_awaited_once()
        # The message passed must be JSON containing the structured input
        sent_message = mock_exec.call_args.args[1]
        parsed_message = json.loads(sent_message)
        assert parsed_message == {"instrument_name": "CL Jul26", "symbol": "CLN6"}

        # Metrics
        assert metrics.counters.get(M_SWARM_CALL_INSTRUMENT_RESOLVER) == 1
        assert M_SWARM_LATENCY_INSTRUMENT_RESOLVER in metrics.timings
        await service.aclose()

    async def test_empty_input_returns_none(self) -> None:
        service, mock_exec, _ = _mock_service('{"confidence": 0.5}')
        assert await service.resolve_instrument("") is None
        mock_exec.assert_not_awaited()
        await service.aclose()

    async def test_unset_agent_id_returns_none(self) -> None:
        service, mock_exec, _ = _mock_service(
            '{"confidence": 0.5}', DAF_INSTRUMENT_RESOLVER_AGENT_ID="",
        )
        assert await service.resolve_instrument("CL Jul26") is None
        mock_exec.assert_not_awaited()
        await service.aclose()

    async def test_null_product_returns_typed_response(self) -> None:
        """Per the agent's system instructions, ``product_name=null``
        is valid when the agent declines to guess — service should
        return the typed response, not None."""
        payload = {
            "product_name": None,
            "product_family": None,
            "structure_type": "Unknown",
            "contract_month": None,
            "expiry_date": None,
            "confidence": 0.05,
            "reasoning": "Cannot place input in vocabulary.",
        }
        service, _, _ = _mock_service(json.dumps(payload))
        out = await service.resolve_instrument("???")
        assert out is not None
        assert out.product_name is None
        assert out.confidence == 0.05
        await service.aclose()
