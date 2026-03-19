"""Circuit breaker for Ollama API calls.

Prevents cascading failures when Ollama is overloaded or down.
Three states: CLOSED (normal) -> OPEN (failing fast) -> HALF_OPEN (probing) -> CLOSED.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum

from config import settings

log = logging.getLogger("rlm_proxy.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when circuit is open and calls are being rejected."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker is open. Retry after {retry_after:.0f}s.")


class CircuitBreaker:
    """Async-safe circuit breaker for outbound API calls."""

    def __init__(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._lock = asyncio.Lock()
        self._half_open_lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= settings.circuit_breaker_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    async def check(self) -> None:
        """Check if a call is allowed. Raises CircuitOpenError if not."""
        current = self.state
        if current == CircuitState.CLOSED:
            return
        if current == CircuitState.HALF_OPEN:
            # Only allow one probe at a time
            if self._half_open_lock.locked():
                remaining = settings.circuit_breaker_timeout - (time.monotonic() - self._last_failure_time)
                raise CircuitOpenError(max(0, remaining))
            return
        # OPEN
        remaining = settings.circuit_breaker_timeout - (time.monotonic() - self._last_failure_time)
        raise CircuitOpenError(max(0, remaining))

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            if self._state != CircuitState.CLOSED:
                log.info("Circuit breaker: %s -> CLOSED (success)", self._state.value)
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            if self._half_open_lock.locked():
                self._half_open_lock.release()

    async def record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._half_open_lock.locked():
                self._half_open_lock.release()

            if self._failure_count >= settings.circuit_breaker_threshold:
                if self._state != CircuitState.OPEN:
                    log.warning(
                        "Circuit breaker: %s -> OPEN after %d consecutive failures",
                        self._state.value, self._failure_count,
                    )
                self._state = CircuitState.OPEN

    def snapshot(self) -> dict:
        """Return state for metrics."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "threshold": settings.circuit_breaker_threshold,
            "timeout_seconds": settings.circuit_breaker_timeout,
        }


# Singleton
breaker = CircuitBreaker()
