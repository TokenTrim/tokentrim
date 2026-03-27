from __future__ import annotations

from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str
