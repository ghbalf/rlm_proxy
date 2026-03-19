"""Pre-loaded utility functions available in the RLM REPL namespace."""

from __future__ import annotations

import re


def chunk_by_lines(text: str, n: int = 100) -> list[str]:
    """Split text into chunks of approximately n lines each."""
    lines = text.split("\n")
    return ["\n".join(lines[i:i + n]) for i in range(0, len(lines), n)]


def chunk_by_chars(text: str, n: int = 50000, overlap: int = 500) -> list[str]:
    """Split text into chunks of approximately n characters with optional overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + n
        chunks.append(text[start:end])
        start = end - overlap if overlap and end < len(text) else end
    return chunks


def search(text: str, pattern: str, context_lines: int = 2) -> list[str]:
    """Search text for a regex pattern and return matching lines with context."""
    lines = text.split("\n")
    matches = []
    for i, line in enumerate(lines):
        if re.search(pattern, line, re.IGNORECASE):
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            snippet = "\n".join(f"{'>>>' if j == i else '   '} {lines[j]}" for j in range(start, end))
            matches.append(f"[Line {i + 1}]\n{snippet}")
    return matches


def count_tokens(text: str, ratio: float = 4.0) -> int:
    """Estimate token count from character count using a ratio (default: ~4 chars/token)."""
    return int(len(text) / ratio)


def chunk_by_sections(text: str, separator: str = r"\n#{1,3}\s+") -> list[str]:
    """Split text by markdown-style headers or a custom separator pattern."""
    parts = re.split(separator, text)
    return [p.strip() for p in parts if p.strip()]
