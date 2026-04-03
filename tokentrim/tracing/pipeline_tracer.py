from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

TOKENTRIM_TRANSFORM_SPAN_KIND = "transform"
TOKENTRIM_TRANSFORM_SPAN_NAME_PREFIX = "tokentrim.transform."


class PipelineSpan(ABC):
    """Integration-owned span handle for pipeline-level tracing."""

    @abstractmethod
    def __enter__(self) -> PipelineSpan:
        """Start span capture."""

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        """Finish span capture."""

    @abstractmethod
    def set_data(self, data: Mapping[str, Any]) -> None:
        """Replace the current structured span payload."""

    @abstractmethod
    def set_error(self, error: BaseException) -> None:
        """Record an exception on the span."""


class PipelineTracer(ABC):
    """Neutral tracing hook that integrations can adapt to native tracing systems."""

    @abstractmethod
    def start_span(
        self,
        *,
        name: str,
        data: Mapping[str, Any] | None = None,
    ) -> PipelineSpan:
        """Start a named span with optional structured data."""


class NullPipelineSpan(PipelineSpan):
    def __enter__(self) -> PipelineSpan:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        del exc_type
        del exc_val
        del exc_tb
        return None

    def set_data(self, data: Mapping[str, Any]) -> None:
        del data

    def set_error(self, error: BaseException) -> None:
        del error


class NullPipelineTracer(PipelineTracer):
    def __init__(self) -> None:
        self._span = NullPipelineSpan()

    def start_span(
        self,
        *,
        name: str,
        data: Mapping[str, Any] | None = None,
    ) -> PipelineSpan:
        del name
        del data
        return self._span


NULL_PIPELINE_TRACER = NullPipelineTracer()


def resolve_pipeline_tracer(tracer: PipelineTracer | None) -> PipelineTracer:
    return tracer if tracer is not None else NULL_PIPELINE_TRACER


def build_transform_span_name(transform_name: str) -> str:
    return f"{TOKENTRIM_TRANSFORM_SPAN_NAME_PREFIX}{transform_name}"


def build_transform_span_data(
    *,
    transform_name: str,
    token_budget: int | None,
    input_items: int,
    input_tokens: int,
    output_items: int | None = None,
    output_tokens: int | None = None,
    changed: bool | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "kind": TOKENTRIM_TRANSFORM_SPAN_KIND,
        "transform_name": transform_name,
        "input_items": input_items,
        "input_tokens": input_tokens,
    }
    if token_budget is not None:
        data["token_budget"] = token_budget
    if output_items is not None:
        data["output_items"] = output_items
    if output_tokens is not None:
        data["output_tokens"] = output_tokens
    if changed is not None:
        data["changed"] = changed
    return deepcopy(data)
