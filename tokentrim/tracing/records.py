from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class TokentrimSpanRecord:
    span_id: str
    trace_id: str
    source: str
    kind: str
    name: str | None
    source_span_id: str | None
    parent_id: str | None
    started_at: str | None
    ended_at: str | None
    error: dict[str, Any] | None
    metrics: dict[str, int] | None
    data: dict[str, Any]
    raw_span: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TokentrimTraceRecord:
    trace_id: str
    source: str
    capture_mode: str
    source_trace_id: str | None
    user_id: str
    session_id: str
    workflow_name: str
    started_at: str | None
    ended_at: str | None
    group_id: str | None
    metadata: dict[str, Any] | None
    raw_trace: dict[str, Any]
    spans: tuple[TokentrimSpanRecord, ...] = ()


def build_canonical_id(*, source: str, source_id: str | None) -> tuple[str, str | None]:
    normalized_source_id = source_id if isinstance(source_id, str) and source_id else None
    if normalized_source_id is None:
        return f"{source}:{uuid4()}", None
    return f"{source}:{normalized_source_id}", normalized_source_id
