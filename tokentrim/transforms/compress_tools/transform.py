from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from tokentrim.types.tool import Tool
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform


@dataclass(frozen=True, slots=True)
class CompressToolDescriptions(Transform):
    """Deterministically normalise and shorten tool descriptions."""

    max_description_chars: int = 200

    @property
    def name(self) -> str:
        return "bpe"

    @property
    def kind(self) -> str:
        return "tools"

    def run(self, tools: list[Tool], _request: PipelineRequest) -> list[Tool]:
        return [self._compress(tool) for tool in tools]

    def _compress(self, tool: Tool) -> Tool:
        normalized = " ".join(tool["description"].split())
        if len(normalized) > self.max_description_chars:
            normalized = normalized[: self.max_description_chars - 3].rstrip() + "..."

        return {
            "name": tool["name"],
            "description": normalized,
            "input_schema": deepcopy(tool["input_schema"]),
        }
