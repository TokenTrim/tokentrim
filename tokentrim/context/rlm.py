from __future__ import annotations

from tokentrim.context.base import ContextStep
from tokentrim.context.request import ContextRequest
from tokentrim.context.store import MemoryStore
from tokentrim.types.message import Message


class RLMStep(ContextStep):
    """
    Retrieve prior state and inject it as a system message when available.
    """

    def __init__(self, memory_store: MemoryStore) -> None:
        self._memory_store = memory_store

    def run(self, messages: list[Message], request: ContextRequest) -> list[Message]:
        if not request.user_id or not request.session_id:
            return list(messages)

        retrieved = self._memory_store.retrieve(
            user_id=request.user_id,
            session_id=request.session_id,
        )
        if not retrieved:
            return list(messages)

        injection: Message = {
            "role": "system",
            "content": retrieved,
        }
        return [injection, *messages]
