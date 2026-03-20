from __future__ import annotations

from typing import Protocol


class MemoryStore(Protocol):
    def retrieve(self, *, user_id: str, session_id: str) -> str | None:
        """Return retrieved state for the current user/session pair."""


class NoOpMemoryStore:
    def retrieve(self, *, user_id: str, session_id: str) -> str | None:
        return None

