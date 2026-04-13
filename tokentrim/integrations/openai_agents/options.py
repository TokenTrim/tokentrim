from __future__ import annotations

from dataclasses import dataclass

from tokentrim.memory import MemoryStore
from tokentrim.tracing import PipelineTracer, TraceStore
from tokentrim.transforms.base import Transform


@dataclass(frozen=True, slots=True)
class OpenAIAgentsOptions:
    """Configuration for the OpenAI Agents integration wrapper."""

    user_id: str | None = None
    session_id: str | None = None
    org_id: str | None = None
    token_budget: int | None = None
    memory_store: MemoryStore | None = None
    agent_aware_memory: bool = False
    trace_store: TraceStore | None = None
    pipeline_tracer: PipelineTracer | None = None
    steps: tuple[Transform, ...] = ()
    apply_to_session_history: bool = True
    apply_to_handoffs: bool = True
