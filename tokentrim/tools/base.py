from __future__ import annotations

from abc import ABC, abstractmethod

from tokentrim.tools.request import ToolsRequest
from tokentrim.types.tool import Tool


class ToolStep(ABC):
    """
    Abstract base class for tool pipeline steps.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Stable identifier for selecting and tracing this step.
        """

    @abstractmethod
    def run(self, tools: list[Tool], request: ToolsRequest) -> list[Tool]:
        """
        Transform the current tool list for the given tools request.
        """
