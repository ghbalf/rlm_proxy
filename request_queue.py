"""Request queue and session concurrency limiter for RLM sessions."""

from __future__ import annotations

import asyncio
import logging
import time

from config import settings

log = logging.getLogger("rlm_proxy.queue")


class RequestQueue:
    """Limits concurrent RLM sessions with an async semaphore and bounded queue."""

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_sessions)
        self._queue_size = 0
        self._lock = asyncio.Lock()

    @property
    def active_sessions(self) -> int:
        return settings.max_concurrent_sessions - self._semaphore._value

    @property
    def queued_requests(self) -> int:
        return self._queue_size

    async def acquire(self) -> float:
        """Acquire a session slot. Returns wait time in seconds.

        Raises RuntimeError if queue is full.
        """
        async with self._lock:
            total_waiting = self._queue_size
            if total_waiting >= settings.max_queue_size:
                raise RuntimeError(
                    f"RLM queue full ({settings.max_queue_size} waiting). Try again later."
                )
            self._queue_size += 1

        t0 = time.perf_counter()
        log.info("Request queued (position ~%d, active=%d)", self._queue_size, self.active_sessions)

        try:
            await self._semaphore.acquire()
        finally:
            async with self._lock:
                self._queue_size -= 1

        wait_time = time.perf_counter() - t0
        if wait_time > 0.1:
            log.info("Request dequeued after %.1fs wait", wait_time)
        return wait_time

    def release(self) -> None:
        """Release a session slot."""
        self._semaphore.release()

    def snapshot(self) -> dict:
        """Return queue state for metrics."""
        return {
            "active_sessions": self.active_sessions,
            "queued_requests": self.queued_requests,
            "max_concurrent": settings.max_concurrent_sessions,
            "max_queue_size": settings.max_queue_size,
        }


# Singleton
queue = RequestQueue()
