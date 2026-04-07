from dataclasses import dataclass

from tokentrim.memory.retriever import build_memory_query, insert_after_leading_system_messages, render_memory_message
from tokentrim.memory.store import DurableMemoryStore
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.types.message import Message
from tokentrim.types.state import PipelineState


@dataclass(frozen=True, slots=True)
class RetrieveDurableMemory(Transform):
    memory_store: DurableMemoryStore | None = None
    max_entries: int = 3

    @property
    def name(self) -> str:
        return "durable_memory"

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        if self.memory_store is None:
            return state
        if not request.user_id or not request.session_id:
            return state

        query = build_memory_query(messages=state.context, task_hint=request.task_hint)
        if not query:
            return state

        entries = self.memory_store.retrieve(
            user_id=request.user_id,
            session_id=request.session_id,
            query=query,
            limit=self.max_entries,
        )
        if not entries:
            return state

        memory_message: Message = {
            "role": "system",
            "content": render_memory_message(entries),
        }
        return PipelineState(
            context=insert_after_leading_system_messages(state.context, memory_message),
            tools=state.tools,
        )
