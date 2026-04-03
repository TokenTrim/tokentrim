from __future__ import annotations

from dataclasses import dataclass

from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.transforms.rlm.store import MemoryStore
from tokentrim.types.message import Message
from tokentrim.types.state import PipelineState


@dataclass(frozen=True, slots=True)
class RetrieveMemory(Transform):
    """Retrieve prior state and inject it as a system message when available."""

    memory_store: MemoryStore | None = None

    @property
    def name(self) -> str:
        return "rlm"

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        messages = state.context
        if self.memory_store is None:
            return state
        if not request.user_id or not request.session_id:
            return state

        retrieved = self.memory_store.retrieve(
            user_id=request.user_id,
            session_id=request.session_id,
        )
        if not retrieved:
            return state

        if messages and messages[0]["role"] == "system":
            merged: Message = {
                "role": "system",
                "content": f"{retrieved}\n\n{messages[0]['content']}".strip(),
            }
            return PipelineState(context=[merged, *messages[1:]], tools=state.tools)

        injection: Message = {"role": "system", "content": retrieved}
        return PipelineState(context=[injection, *messages], tools=state.tools)
