from __future__ import annotations

from tokentrim.context.store import MemoryStore
from tokentrim.types.message import Message


class RLMStep:
    """
    Retrieve prior state and inject it as a system message when available.
    """

    def __init__(self, memory_store: MemoryStore) -> None:
        self._memory_store = memory_store

    def run(
        self,
        messages: list[Message],
        *,
        user_id: str | None,
        session_id: str | None,
    ) -> list[Message]:
        if not user_id or not session_id:
            return list(messages)

        retrieved = self._memory_store.retrieve(user_id=user_id, session_id=session_id)
        if not retrieved:
            return list(messages)

        injection: Message = {
            "role": "system",
            "content": retrieved,
        }
        return [injection, *messages]

