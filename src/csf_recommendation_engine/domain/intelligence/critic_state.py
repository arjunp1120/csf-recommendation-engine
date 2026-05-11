"""Critic shadow→strict promotion state.

Plan §11.4 + §17: each swarm response runs through Python-side
validators (citation fidelity + eligibility) on top of whatever the
DAF-side critic does. We start in **shadow mode** (log violations,
serve anyway) and promote to **strict mode** (reject + retry once,
then deterministic fallback) once the trailing-N violation rate has
fallen below a threshold.

This module owns the rolling violation-rate tracker. It is a
process-local in-memory deque; the metric resets on process restart.
For prod observability the snapshot can be scraped via the
`/metrics` endpoint added in Step 5.4.

Default config (from plan §17 + Step 0.3 settings):
* window = 500 events (last 500 served responses)
* violation_threshold = 0.01 (promote when below 1%)
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


@dataclass
class CriticStateSnapshot:
    """Read-only view of the tracker state for /metrics + ops review."""

    window: int
    threshold: float
    events_recorded: int
    violation_rate: float
    can_promote: bool


class CriticState:
    """Process-local rolling window of pass/fail outcomes from the
    Python-side validators.

    Thread-safe (uses a lock around mutation).
    """

    def __init__(self, window: int = 500, violation_threshold: float = 0.01) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if not 0.0 <= violation_threshold <= 1.0:
            raise ValueError("violation_threshold must be in [0,1]")
        self._window = window
        self._threshold = violation_threshold
        self._events: deque[bool] = deque(maxlen=window)
        self._lock = threading.Lock()

    @property
    def window(self) -> int:
        return self._window

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def events_recorded(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def violation_rate(self) -> float:
        """Fraction of events in the current window that were violations
        (validator failures). Returns 0.0 when the window is empty."""
        with self._lock:
            if not self._events:
                return 0.0
            passes = sum(1 for ok in self._events if ok)
            return 1.0 - passes / len(self._events)

    @property
    def can_promote(self) -> bool:
        """True when the trailing-window violation rate is at or below
        the threshold AND we have a full window of evidence.

        A partial window (fewer than `window` events) is treated as
        insufficient — promotion requires the full sample size to
        prevent premature switching on volatile early data.
        """
        with self._lock:
            if len(self._events) < self._window:
                return False
            passes = sum(1 for ok in self._events if ok)
            rate = 1.0 - passes / len(self._events)
            return rate <= self._threshold

    def record(self, passed: bool) -> None:
        """Append a single outcome (True = validator passed)."""
        with self._lock:
            self._events.append(passed)

    def snapshot(self) -> CriticStateSnapshot:
        """Return a read-only view of the current tracker state."""
        with self._lock:
            events = len(self._events)
            passes = sum(1 for ok in self._events if ok)
            rate = 1.0 - passes / events if events else 0.0
            can_promote = events >= self._window and rate <= self._threshold
        return CriticStateSnapshot(
            window=self._window,
            threshold=self._threshold,
            events_recorded=events,
            violation_rate=rate,
            can_promote=can_promote,
        )

    def reset(self) -> None:
        """Clear all recorded events. Used in tests and on explicit
        operator action."""
        with self._lock:
            self._events.clear()
