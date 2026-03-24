from __future__ import annotations

from dataclasses import dataclass, replace

from tokentrim.context.base import ContextStep
from tokentrim.context.request import ContextRequest
from tokentrim.context.store import MemoryStore
from tokentrim.types.message import Message


@dataclass(frozen=True, slots=True)
class RetrieveMemory(ContextStep):
    """
    Retrieve prior state and inject it as a system message when available.
    """

    memory_store: MemoryStore | None = None

    @property
    def name(self) -> str:
        return "rlm"

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
        compaction_model: str | None = None,
        memory_store: MemoryStore | None = None,
    ) -> ContextStep:
        del tokenizer_model
        del compaction_model
        return replace(
            self,
            memory_store=self.memory_store if self.memory_store is not None else memory_store,
        )

    def run(self, messages: list[Message], request: ContextRequest) -> list[Message]:
        if self.memory_store is None:
            return list(messages)
        if not request.user_id or not request.session_id:
            return list(messages)

        retrieved = self.memory_store.retrieve(
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


RLMStep = RetrieveMemory

__all__ = ["RetrieveMemory", "RLMStep"]
