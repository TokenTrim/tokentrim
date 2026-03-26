from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokentrim.types.message import Message
from tokentrim.types.tool import Tool

if TYPE_CHECKING:
    from tokentrim.transforms.base import Transform


@dataclass(frozen=True, slots=True)
class ContextRequest:
    messages: tuple[Message, ...]
    user_id: str | None
    session_id: str | None
    token_budget: int | None
    steps: tuple[Transform, ...]


@dataclass(frozen=True, slots=True)
class ToolsRequest:
    tools: tuple[Tool, ...]
    task_hint: str | None
    token_budget: int | None
    steps: tuple[Transform, ...]
