from __future__ import annotations

from dataclasses import dataclass

from tokentrim.types.tool import Tool


@dataclass(frozen=True, slots=True)
class ToolsRequest:
    tools: tuple[Tool, ...]
    task_hint: str | None
    token_budget: int | None
    steps: tuple[str, ...]
