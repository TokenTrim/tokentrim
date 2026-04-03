from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokentrim.types.message import Message
from tokentrim.types.tool import Tool

if TYPE_CHECKING:
    from tokentrim.tracing import PipelineTracer, TraceStore
    from tokentrim.transforms.base import Transform


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    messages: tuple[Message, ...] = ()
    tools: tuple[Tool, ...] = ()
    user_id: str | None = None
    session_id: str | None = None
    task_hint: str | None = None
    token_budget: int | None = None
    trace_store: TraceStore | None = None
    pipeline_tracer: PipelineTracer | None = None
    steps: tuple[Transform, ...] = ()


@dataclass(frozen=True, slots=True, init=False)
class ContextRequest(PipelineRequest):
    def __init__(
        self,
        *,
        messages: tuple[Message, ...],
        user_id: str | None,
        session_id: str | None,
        token_budget: int | None,
        steps: tuple[Transform, ...],
    ) -> None:
        PipelineRequest.__init__(
            self,
            messages=messages,
            tools=(),
            user_id=user_id,
            session_id=session_id,
            task_hint=None,
            token_budget=token_budget,
            trace_store=None,
            pipeline_tracer=None,
            steps=steps,
        )


@dataclass(frozen=True, slots=True, init=False)
class ToolsRequest(PipelineRequest):
    def __init__(
        self,
        *,
        tools: tuple[Tool, ...],
        task_hint: str | None,
        token_budget: int | None,
        steps: tuple[Transform, ...],
    ) -> None:
        PipelineRequest.__init__(
            self,
            messages=(),
            tools=tools,
            user_id=None,
            session_id=None,
            task_hint=task_hint,
            token_budget=token_budget,
            trace_store=None,
            pipeline_tracer=None,
            steps=steps,
        )
