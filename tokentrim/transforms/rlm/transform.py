from __future__ import annotations

from dataclasses import dataclass

from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.transforms.rlm.store import MemoryStore
from tokentrim.types.message import Message


@dataclass(frozen=True, slots=True)
class RetrieveMemory(Transform):
    """Retrieve prior state and inject it as a system message when available."""

    memory_store: MemoryStore | None = None

    @property
    def name(self) -> str:
        return "rlm"

    @property
    def kind(self) -> str:
        return "context"

    def run(self, messages: list[Message], request: PipelineRequest) -> list[Message]:
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
