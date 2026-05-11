"""
Init this class on app startup to track metrics and timings.
Used for monitoring and debugging.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsRecorder:
    counters: dict[str, int] = field(default_factory=dict)
    timings: dict[str, list[float]] = field(default_factory=dict)

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def record_timing(self, name: str, seconds: float) -> None:
        self.timings.setdefault(name, []).append(seconds)

    def snapshot(self) -> dict[str, Any]:
        return {"counters": dict(self.counters), "timings": {k: list(v) for k, v in self.timings.items()}}


# ---------------------------------------------------------------------------
# Standard metric names used by the Intelligence Engine layer.
#
# These are string constants so callers can reference them via
# `MetricsRecorder.increment(M_PACKET_BUILD)` instead of stringly-typed
# strings sprinkled across the codebase. New metric names should be added
# here when they're introduced, never inlined.
#
# Plan §11.4 / Step 0.11: validator failures + swarm call counts +
# packet build counts. Embedding generation metrics are DEFERRED with
# pgvector (plan §17).
# ---------------------------------------------------------------------------

# Packet construction
M_PACKET_BUILD = "intelligence.packet.build"
M_PACKET_BUILD_LATENCY = "intelligence.packet.build_latency_s"

# Swarm calls — increment per swarm to keep series cardinality low
M_SWARM_CALL_TAGGER = "intelligence.swarm.call.tagger"
M_SWARM_CALL_PROFILER = "intelligence.swarm.call.profiler"
M_SWARM_CALL_MARKET_READER = "intelligence.swarm.call.market_reader"
M_SWARM_CALL_RECOMMENDER_EXPLAINER = "intelligence.swarm.call.recommender_explainer"
M_SWARM_CALL_MATCH_STRATEGIST = "intelligence.swarm.call.match_strategist"
M_SWARM_CALL_COVERAGE_COACH = "intelligence.swarm.call.coverage_coach"

# Swarm-call latency (per swarm)
M_SWARM_LATENCY_TAGGER = "intelligence.swarm.latency.tagger"
M_SWARM_LATENCY_PROFILER = "intelligence.swarm.latency.profiler"
M_SWARM_LATENCY_MARKET_READER = "intelligence.swarm.latency.market_reader"
M_SWARM_LATENCY_RECOMMENDER_EXPLAINER = "intelligence.swarm.latency.recommender_explainer"
M_SWARM_LATENCY_MATCH_STRATEGIST = "intelligence.swarm.latency.match_strategist"
M_SWARM_LATENCY_COVERAGE_COACH = "intelligence.swarm.latency.coverage_coach"

# Validator outcomes (plan §11.4)
M_VALIDATOR_RUN = "intelligence.validator.run"
M_VALIDATOR_PASS = "intelligence.validator.pass"
M_VALIDATOR_FAIL_CITATION = "intelligence.validator.fail.citation"
M_VALIDATOR_FAIL_ELIGIBILITY = "intelligence.validator.fail.eligibility"
M_VALIDATOR_UNSUPPORTED_NUMERIC = "intelligence.validator.unsupported_numeric"

# Critic mode promotion (plan §11.4 + Step 5.4)
M_CRITIC_MODE_SHADOW_SERVE = "intelligence.critic.mode.shadow_serve"
M_CRITIC_MODE_STRICT_SERVE = "intelligence.critic.mode.strict_serve"
M_CRITIC_MODE_STRICT_RETRY = "intelligence.critic.mode.strict_retry"
M_CRITIC_MODE_STRICT_FALLBACK = "intelligence.critic.mode.strict_fallback"

# Engagement-event ingest (plan §15.5, Step 0.7)
M_ENGAGEMENT_EVENT_INGEST = "engagement.event.ingest"
