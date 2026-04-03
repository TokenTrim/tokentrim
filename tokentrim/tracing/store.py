from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace

from tokentrim.tracing.records import TokentrimSpanRecord, TokentrimTraceRecord


class TraceStore(ABC):
    @abstractmethod
    def list_traces(
        self,
        *,
        user_id: str,
        session_id: str,
        limit: int | None = None,
    ) -> tuple[TokentrimTraceRecord, ...]:
        """Return completed traces for the given user/session scope."""

    @abstractmethod
    def create_trace(
        self,
        *,
        user_id: str,
        session_id: str,
        trace: TokentrimTraceRecord,
    ) -> None:
        """Create a new active trace."""

    @abstractmethod
    def append_span(
        self,
        *,
        trace_id: str,
        span: TokentrimSpanRecord,
    ) -> None:
        """Append a completed span to an active trace."""

    @abstractmethod
    def complete_trace(
        self,
        *,
        trace_id: str,
    ) -> None:
        """Mark an active trace complete and index it for reads."""


@dataclass(slots=True)
class _ActiveTraceState:
    user_id: str
    session_id: str
    trace: TokentrimTraceRecord
    spans: list[TokentrimSpanRecord]


class InMemoryTraceStore(TraceStore):
    """Process-local storage for canonical Tokentrim trace history."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_traces: dict[str, _ActiveTraceState] = {}
        self._completed_traces: dict[str, TokentrimTraceRecord] = {}
        self._trace_index: dict[tuple[str, str], list[str]] = defaultdict(list)

    def list_traces(
        self,
        *,
        user_id: str,
        session_id: str,
        limit: int | None = None,
    ) -> tuple[TokentrimTraceRecord, ...]:
        with self._lock:
            trace_ids = list(self._trace_index.get((user_id, session_id), ()))
            if limit is not None and limit <= 0:
                return tuple()

            selected = reversed(trace_ids)
            if limit is not None:
                return tuple(self._completed_traces[trace_id] for _, trace_id in zip(range(limit), selected))
            return tuple(self._completed_traces[trace_id] for trace_id in selected)

    def create_trace(
        self,
        *,
        user_id: str,
        session_id: str,
        trace: TokentrimTraceRecord,
    ) -> None:
        with self._lock:
            self._active_traces[trace.trace_id] = _ActiveTraceState(
                user_id=user_id,
                session_id=session_id,
                trace=trace,
                spans=[],
            )

    def append_span(
        self,
        *,
        trace_id: str,
        span: TokentrimSpanRecord,
    ) -> None:
        with self._lock:
            active_trace = self._active_traces.get(trace_id)
            if active_trace is None:
                return
            active_trace.spans.append(span)

    def complete_trace(
        self,
        *,
        trace_id: str,
    ) -> None:
        with self._lock:
            active_trace = self._active_traces.pop(trace_id, None)
            if active_trace is None:
                return

            ordered_spans = tuple(
                sorted(
                    active_trace.spans,
                    key=lambda span: (
                        span.started_at is None,
                        span.started_at or "",
                        span.ended_at or "",
                        span.span_id,
                    ),
                )
            )
            started_at = next(
                (span.started_at for span in ordered_spans if span.started_at is not None),
                active_trace.trace.started_at,
            )
            end_candidates = [
                span.ended_at or span.started_at
                for span in ordered_spans
                if span.ended_at is not None or span.started_at is not None
            ]
            if active_trace.trace.ended_at is not None:
                end_candidates.append(active_trace.trace.ended_at)
            ended_at = max(end_candidates) if end_candidates else None
            completed_trace = replace(
                active_trace.trace,
                started_at=started_at,
                ended_at=ended_at,
                spans=ordered_spans,
            )
            self._completed_traces[trace_id] = completed_trace
            self._trace_index[(active_trace.user_id, active_trace.session_id)].append(trace_id)
