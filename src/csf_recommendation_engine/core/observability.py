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
