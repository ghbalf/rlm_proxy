"""Trajectory logging for RLM sessions.

Writes JSONL files (one JSON object per line) capturing every event
in an RLM run: root LLM calls, code executions, REPL outputs, and
sub-calls with timing information.

If no log directory is configured the logger is a silent no-op.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger("rlm_proxy.trajectory")


class TrajectoryLogger:
    """Accumulates timestamped events and writes them to a JSONL file."""

    def __init__(self, log_dir: str) -> None:
        self._enabled = bool(log_dir)
        if not self._enabled:
            return

        self._log_dir = Path(log_dir)
        self._session_id = uuid.uuid4().hex
        self._events: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        if not self._enabled:
            return ""
        return self._session_id

    def log_event(self, event_type: str, data: dict) -> None:
        """Record a single event with a relative timestamp."""
        if not self._enabled:
            return
        self._events.append(
            {
                "session_id": self._session_id,
                "ts": round(time.perf_counter() - self._t0, 4),
                "event": event_type,
                **data,
            }
        )

    def finalize(self) -> dict:
        """Write all events to a JSONL file and return a summary dict."""
        if not self._enabled:
            return {}

        self._log_dir.mkdir(parents=True, exist_ok=True)
        path = self._log_dir / f"{self._session_id}.jsonl"

        with open(path, "w") as fh:
            for event in self._events:
                fh.write(json.dumps(event, default=str) + "\n")

        summary = {
            "session_id": self._session_id,
            "log_file": str(path),
            "total_events": len(self._events),
            "elapsed_seconds": round(time.perf_counter() - self._t0, 4),
        }
        log.info("Trajectory written: %s (%d events)", path, len(self._events))
        return summary
