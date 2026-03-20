from __future__ import annotations

from dataclasses import dataclass

from tokentrim.types.message import Message


@dataclass(frozen=True, slots=True)
class ContextResult:
    messages: tuple[Message, ...]
    token_count: int
    trace_id: str

