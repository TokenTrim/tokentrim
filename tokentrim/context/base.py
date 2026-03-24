from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tokentrim.context.request import ContextRequest
from tokentrim.types.message import Message

if TYPE_CHECKING:
    from tokentrim.context.store import MemoryStore


class ContextStep(ABC):
    """
    Abstract base class for context pipeline steps.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Stable identifier for selecting and tracing this step.
        """

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
        compaction_model: str | None = None,
        memory_store: MemoryStore | None = None,
    ) -> ContextStep:
        del tokenizer_model
        del compaction_model
        del memory_store
        return self

    @abstractmethod
    def run(self, messages: list[Message], request: ContextRequest) -> list[Message]:
        """
        Transform the current message list for the given context request.
        """
