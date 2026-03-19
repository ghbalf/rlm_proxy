"""Observability metrics collector for the RLM proxy."""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class MetricsCollector:
    """Singleton metrics collector tracking request and RLM statistics."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Request counters
    total_requests: int = 0
    rlm_requests: int = 0
    passthrough_requests: int = 0
    stream_requests: int = 0
    error_count: int = 0

    # RLM-specific
    total_sub_calls: int = 0
    total_cache_hits: int = 0
    total_iterations: int = 0

    # Timing (for percentile calculation)
    _durations: list[float] = field(default_factory=list, repr=False)

    # Active sessions
    active_sessions: int = 0

    # Iteration histogram (iteration_count -> occurrences)
    _iteration_histogram: dict[int, int] = field(default_factory=lambda: defaultdict(int), repr=False)

    def record_request(self, *, rlm: bool = False, stream: bool = False) -> None:
        with self._lock:
            self.total_requests += 1
            if rlm:
                self.rlm_requests += 1
            else:
                self.passthrough_requests += 1
            if stream:
                self.stream_requests += 1

    def record_error(self) -> None:
        with self._lock:
            self.error_count += 1

    def record_rlm_result(self, *, iterations: int, sub_calls: int, cache_hits: int, duration: float) -> None:
        with self._lock:
            self.total_iterations += iterations
            self.total_sub_calls += sub_calls
            self.total_cache_hits += cache_hits
            self._durations.append(duration)
            self._iteration_histogram[iterations] += 1

    def record_passthrough_duration(self, duration: float) -> None:
        with self._lock:
            self._durations.append(duration)

    def session_start(self) -> None:
        with self._lock:
            self.active_sessions += 1

    def session_end(self) -> None:
        with self._lock:
            self.active_sessions = max(0, self.active_sessions - 1)

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of all metrics."""
        with self._lock:
            durations = sorted(self._durations) if self._durations else []

            def percentile(p):
                if not durations:
                    return 0.0
                idx = int(len(durations) * p / 100)
                return round(durations[min(idx, len(durations) - 1)], 3)

            return {
                "requests": {
                    "total": self.total_requests,
                    "rlm": self.rlm_requests,
                    "passthrough": self.passthrough_requests,
                    "stream": self.stream_requests,
                    "errors": self.error_count,
                },
                "rlm": {
                    "total_sub_calls": self.total_sub_calls,
                    "total_cache_hits": self.total_cache_hits,
                    "total_iterations": self.total_iterations,
                    "iteration_histogram": dict(self._iteration_histogram),
                },
                "active_sessions": self.active_sessions,
                "duration_percentiles": {
                    "p50": percentile(50),
                    "p90": percentile(90),
                    "p95": percentile(95),
                    "p99": percentile(99),
                },
                "total_completed_requests": len(durations),
            }


# Singleton instance
collector = MetricsCollector()
