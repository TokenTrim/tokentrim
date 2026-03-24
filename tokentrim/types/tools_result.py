from __future__ import annotations

from dataclasses import dataclass

from tokentrim.types.tool import Tool
from tokentrim.types.step_trace import StepTrace


@dataclass(frozen=True, slots=True)
class ToolsResult:
    tools: tuple[Tool, ...]
    step_traces: tuple[StepTrace, ...]
    token_count: int
    trace_id: str
