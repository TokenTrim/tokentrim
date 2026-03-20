from __future__ import annotations

from dataclasses import dataclass

from tokentrim.types.tool import Tool


@dataclass(frozen=True, slots=True)
class ToolsResult:
    tools: tuple[Tool, ...]
    created_tools: tuple[Tool, ...]
    token_count: int
    trace_id: str

