from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokentrim.types.message import Message

if TYPE_CHECKING:
    from tokentrim.context.base import ContextStep


@dataclass(frozen=True, slots=True)
class ContextRequest:
    messages: tuple[Message, ...]
    user_id: str | None
    session_id: str | None
    token_budget: int | None
    steps: tuple[ContextStep, ...]
