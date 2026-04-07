from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryWriteCandidate:
    content: str
    metadata: dict[str, object]


def normalize_memory_content(content: str) -> str:
    return content.strip()
