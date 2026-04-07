from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MemoryScope(str, Enum):
    SESSION = "session"
    USER = "user"
    PROJECT = "project"


@dataclass(frozen=True, slots=True)
class MemoryIndexRecord:
    entry_id: str
    path: str | None
    content: str
    created_at: str
    metadata: dict[str, object]
    keywords: tuple[str, ...]
    scope: MemoryScope = MemoryScope.SESSION


def default_project_id() -> str:
    return sanitize_memory_segment(Path.cwd().name)


def default_memory_root() -> Path:
    return Path.cwd() / ".tokentrim" / "memory"


def sanitize_memory_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "default"
