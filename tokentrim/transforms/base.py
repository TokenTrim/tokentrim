from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from tokentrim.types.message import Message
from tokentrim.types.tool import Tool
from tokentrim.pipeline.requests import ContextRequest, ToolsRequest

TransformKind = Literal["context", "tools"]


class Transform(ABC):
    """Abstract base class for all transforms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for selecting and tracing this transform."""

    @property
    @abstractmethod
    def kind(self) -> TransformKind:
        """Payload kind handled by this transform."""

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
    ) -> Transform:
        del tokenizer_model
        return self

    @abstractmethod
    def run(
        self,
        payload: list[Message] | list[Tool],
        request: ContextRequest | ToolsRequest,
    ) -> list[Message] | list[Tool]:
        """Transform the current payload for the given request."""
