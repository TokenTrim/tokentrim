from __future__ import annotations

from typing import Any, TypedDict


class Tool(TypedDict):
    name: str
    description: str
    input_schema: dict[str, Any]

