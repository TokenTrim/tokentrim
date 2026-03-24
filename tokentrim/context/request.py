from __future__ import annotations

from dataclasses import dataclass

from tokentrim.types.message import Message


@dataclass(frozen=True, slots=True)
class ContextRequest:
    messages: tuple[Message, ...]
    user_id: str | None
    session_id: str | None
    token_budget: int | None
    steps: tuple[str, ...]
