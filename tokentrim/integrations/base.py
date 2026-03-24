from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from tokentrim.client import Tokentrim


ConfigT = TypeVar("ConfigT")


class IntegrationAdapter(ABC, Generic[ConfigT]):
    """
    Abstract base class for Tokentrim SDK integration adapters.
    """

    @abstractmethod
    def wrap(self, tokentrim: Tokentrim, config: ConfigT | None = None) -> ConfigT:
        """
        Return an integration-specific config or wrapper with Tokentrim applied.
        """
