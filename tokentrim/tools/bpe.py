from __future__ import annotations

from copy import deepcopy

from tokentrim.types.tool import Tool


class ToolBPEStep:
    """
    Deterministically normalise and shorten tool descriptions.
    """

    _MAX_DESCRIPTION_CHARS = 200

    def run(self, tools: list[Tool]) -> list[Tool]:
        return [self._compress(tool) for tool in tools]

    def _compress(self, tool: Tool) -> Tool:
        normalized = " ".join(tool["description"].split())
        if len(normalized) > self._MAX_DESCRIPTION_CHARS:
            normalized = normalized[: self._MAX_DESCRIPTION_CHARS - 3].rstrip() + "..."

        return {
            "name": tool["name"],
            "description": normalized,
            "input_schema": deepcopy(tool["input_schema"]),
        }

