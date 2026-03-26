from __future__ import annotations

from dataclasses import dataclass

from tokentrim.types.message import Message
from tokentrim.types.tool import Tool
from tokentrim.types.trace import Trace


@dataclass(frozen=True, slots=True)
class Result:
    trace: Trace
    context: tuple[Message, ...] | None = None
    tools: tuple[Tool, ...] | None = None
