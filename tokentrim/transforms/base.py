from __future__ import annotations

from abc import ABC, abstractmethod

from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.types.state import PipelineState


class Transform(ABC):
    """Abstract base class for all transforms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for selecting and tracing this transform."""

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
    ) -> Transform:
        del tokenizer_model
        return self

    def resolve_token_budget(
        self,
        token_budget: int | None,
    ) -> int | None:
        return token_budget

    @abstractmethod
    def run(
        self,
        state: PipelineState,
        request: PipelineRequest,
    ) -> PipelineState:
        """Transform the current pipeline state for the given request."""
