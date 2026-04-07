from dataclasses import dataclass, field

from tokentrim.memory.policy import DefaultMemoryWritePolicy, MemoryWritePolicy
from tokentrim.memory.store import DurableMemoryStore
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.types.state import PipelineState


@dataclass(frozen=True, slots=True)
class RememberDurableMemory(Transform):
    memory_store: DurableMemoryStore | None = None
    policy: MemoryWritePolicy = field(default_factory=DefaultMemoryWritePolicy)

    @property
    def name(self) -> str:
        return "remember_durable_memory"

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        if self.memory_store is None:
            return state
        if not request.user_id or not request.session_id:
            return state

        candidate = self.policy.build_candidate(state=state, request=request)
        if candidate is None:
            return state

        self.memory_store.remember(
            user_id=request.user_id,
            session_id=request.session_id,
            content=candidate.content,
            metadata=candidate.metadata,
        )
        return state
