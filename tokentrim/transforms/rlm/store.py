from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryStore(ABC):
    @abstractmethod
    def retrieve(self, *, user_id: str, session_id: str) -> str | None:
        """Return retrieved memory text for the user/session, if available."""


class NoOpMemoryStore(MemoryStore):
    def retrieve(self, *, user_id: str, session_id: str) -> str | None:
        del user_id
        del session_id
        return None
