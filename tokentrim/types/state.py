from __future__ import annotations

from dataclasses import dataclass

from tokentrim.types.message import Message
from tokentrim.types.tool import Tool


@dataclass(frozen=True, slots=True)
class PipelineState:
    context: list[Message]
    tools: list[Tool]
