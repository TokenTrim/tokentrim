from __future__ import annotations

from dataclasses import dataclass

from tokentrim.types.message import Message
from tokentrim.types.step_trace import StepTrace


@dataclass(frozen=True, slots=True)
class ContextResult:
    messages: tuple[Message, ...]
    step_traces: tuple[StepTrace, ...]
    token_count: int
    trace_id: str
