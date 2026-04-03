from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from tokentrim.tracing import PipelineSpan, PipelineTracer


class OpenAIAgentsPipelineSpan(PipelineSpan):
    def __init__(
        self,
        *,
        name: str,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._data = _copy_data(data)
        self._span: Any | None = None

    def __enter__(self) -> PipelineSpan:
        from agents.tracing import custom_span, get_current_span, get_current_trace

        if get_current_span() is None and get_current_trace() is None:
            return self

        self._span = custom_span(self._name, data=_copy_data(self._data))
        self._span.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        if self._span is None:
            return None
        try:
            return self._span.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._span = None

    def set_data(self, data: Mapping[str, Any]) -> None:
        self._data = _copy_data(data)
        if self._span is None:
            return
        self._span.span_data.data = _copy_data(self._data)

    def set_error(self, error: BaseException) -> None:
        if self._span is None:
            return
        self._span.set_error(
            {
                "message": str(error),
                "data": {
                    "error_type": error.__class__.__name__,
                },
            }
        )


class OpenAIAgentsPipelineTracer(PipelineTracer):
    def start_span(
        self,
        *,
        name: str,
        data: Mapping[str, Any] | None = None,
    ) -> PipelineSpan:
        return OpenAIAgentsPipelineSpan(name=name, data=data)


def _copy_data(data: Mapping[str, Any] | None) -> dict[str, Any]:
    if data is None:
        return {}
    return deepcopy(dict(data))
