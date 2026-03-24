from __future__ import annotations

from abc import ABC, abstractmethod

from tokentrim.context.request import ContextRequest
from tokentrim.types.message import Message


class ContextStep(ABC):
    """
    Abstract base class for context pipeline steps.
    """

    @abstractmethod
    def run(self, messages: list[Message], request: ContextRequest) -> list[Message]:
        """
        Transform the current message list for the given context request.
        """
