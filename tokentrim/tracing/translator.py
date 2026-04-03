from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from tokentrim.tracing.records import TokentrimSpanRecord, TokentrimTraceRecord


class TraceTranslator(ABC):
    source: str
    capture_mode: str

    @abstractmethod
    def translate_trace(
        self,
        payload: Mapping[str, Any],
        *,
        user_id: str,
        session_id: str,
    ) -> TokentrimTraceRecord:
        """Translate a source trace payload into the canonical Tokentrim shape."""

    @abstractmethod
    def translate_span(self, payload: Mapping[str, Any]) -> TokentrimSpanRecord:
        """Translate a source span payload into the canonical Tokentrim shape."""
