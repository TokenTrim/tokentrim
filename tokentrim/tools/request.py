from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokentrim.types.tool import Tool

if TYPE_CHECKING:
    from tokentrim.tools.base import ToolStep


@dataclass(frozen=True, slots=True)
class ToolsRequest:
    tools: tuple[Tool, ...]
    task_hint: str | None
    token_budget: int | None
    steps: tuple[ToolStep, ...]
