"""In-memory ring buffer for recent log entries, searchable via API."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class LogEntry:
    timestamp: float
    level: str
    logger: str
    message: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "time": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "level": self.level,
            "logger": self.logger,
            "message": self.message,
        }


class BufferHandler(logging.Handler):
    """Logging handler that stores records in a bounded deque."""

    def __init__(self, maxlen: int = 2000) -> None:
        super().__init__()
        self._buffer: deque[LogEntry] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        entry = LogEntry(
            timestamp=record.created,
            level=record.levelname,
            logger=record.name,
            message=self.format(record),
        )
        self._buffer.append(entry)

    def get_entries(
        self,
        *,
        query: str = "",
        level: str = "",
        limit: int = 200,
    ) -> list[dict]:
        results = []
        query_lower = query.lower()
        level_upper = level.upper()

        for entry in reversed(self._buffer):
            if level_upper and entry.level != level_upper:
                continue
            if query_lower and query_lower not in entry.message.lower() and query_lower not in entry.logger.lower():
                continue
            results.append(entry.to_dict())
            if len(results) >= limit:
                break
        return results


# Singleton
buffer_handler = BufferHandler()
