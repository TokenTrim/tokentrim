from __future__ import annotations

"""Trace store implementations for live capture and offline replay."""

import json
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import quote

from tokentrim.tracing.records import TokentrimSpanRecord, TokentrimTraceRecord


class TraceStore(ABC):
    """Persistence abstraction for Tokentrim trace history."""

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

            completed_trace = _finalize_trace(
                trace=active_trace.trace,
                spans=tuple(active_trace.spans),
            )
            self._completed_traces[trace_id] = completed_trace
            self._trace_index[(active_trace.user_id, active_trace.session_id)].append(trace_id)


class FilesystemTraceStore(TraceStore):
    """Filesystem-backed local-first trace storage for offline replay.

    Layout:
    - `active/<trace_id>.json` stores in-flight traces plus appended spans
    - `completed/<user_id>/<session_id>/<trace_id>.json` stores finalized traces
    """

    def __init__(self, *, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._active_dir.mkdir(parents=True, exist_ok=True)
        self._completed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _active_dir(self) -> Path:
        return self._root_dir / "active"

    @property
    def _completed_dir(self) -> Path:
        return self._root_dir / "completed"

    def list_traces(
        self,
        *,
        user_id: str,
        session_id: str,
        limit: int | None = None,
    ) -> tuple[TokentrimTraceRecord, ...]:
        directory = self._completed_dir / user_id / session_id
        if not directory.exists():
            return tuple()
        if limit is not None and limit <= 0:
            return tuple()
        traces = sorted(
            (self._read_completed_trace(path) for path in directory.glob("*.json")),
            key=lambda trace: (
                trace.ended_at or trace.started_at or "",
                trace.trace_id,
            ),
            reverse=True,
        )
        if limit is None:
            return tuple(traces)
        return tuple(traces[:limit])

    def create_trace(
        self,
        *,
        user_id: str,
        session_id: str,
        trace: TokentrimTraceRecord,
    ) -> None:
        self._write_json(
            self._active_path(trace.trace_id),
            _serialize_active_trace_payload(
                user_id=user_id,
                session_id=session_id,
                trace=trace,
                spans=(),
            ),
        )

    def append_span(
        self,
        *,
        trace_id: str,
        span: TokentrimSpanRecord,
    ) -> None:
        path = self._active_path(trace_id)
        if not path.exists():
            return
        payload = self._read_json(path)
        payload["spans"].append(_serialize_span(span))
        self._write_json(path, payload)

    def complete_trace(
        self,
        *,
        trace_id: str,
    ) -> None:
        active_path = self._active_path(trace_id)
        if not active_path.exists():
            return
        payload = self._read_json(active_path)
        trace = _deserialize_trace(payload["trace"])
        spans = tuple(
            _deserialize_span(span_payload)
            for span_payload in payload.get("spans", [])
            if isinstance(span_payload, dict)
        )
        completed_trace = _finalize_trace(trace=trace, spans=spans)
        user_id = str(payload["user_id"])
        session_id = str(payload["session_id"])
        completed_path = self._completed_path(user_id=user_id, session_id=session_id, trace_id=trace_id)
        completed_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(completed_path, _serialize_trace(completed_trace))
        active_path.unlink(missing_ok=True)

    def _active_path(self, trace_id: str) -> Path:
        return self._active_dir / f"{_quote_path_segment(trace_id)}.json"

    def _completed_path(self, *, user_id: str, session_id: str, trace_id: str) -> Path:
        return self._completed_dir / user_id / session_id / f"{_quote_path_segment(trace_id)}.json"

    def _read_completed_trace(self, path: Path) -> TokentrimTraceRecord:
        return _deserialize_trace(self._read_json(path))

    def _read_json(self, path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _serialize_active_trace_payload(
    *,
    user_id: str,
    session_id: str,
    trace: TokentrimTraceRecord,
    spans: tuple[TokentrimSpanRecord, ...],
) -> dict[str, object]:
    """Serialize one active trace document."""
    return {
        "user_id": user_id,
        "session_id": session_id,
        "trace": _serialize_trace(trace),
        "spans": [_serialize_span(span) for span in spans],
    }


def _finalize_trace(
    *,
    trace: TokentrimTraceRecord,
    spans: tuple[TokentrimSpanRecord, ...],
) -> TokentrimTraceRecord:
    """Normalize span ordering and derive stable start/end timestamps."""
    ordered_spans = tuple(
        sorted(
            spans,
            key=lambda span: (
                span.started_at is None,
                span.started_at or "",
                span.ended_at or "",
                span.span_id,
            ),
        )
    )
    started_at = next((span.started_at for span in ordered_spans if span.started_at is not None), trace.started_at)
    end_candidates = [
        span.ended_at or span.started_at
        for span in ordered_spans
        if span.ended_at is not None or span.started_at is not None
    ]
    if trace.ended_at is not None:
        end_candidates.append(trace.ended_at)
    return replace(
        trace,
        started_at=started_at,
        ended_at=max(end_candidates) if end_candidates else None,
        spans=ordered_spans,
    )


def _quote_path_segment(value: str) -> str:
    """Encode ids for safe filesystem storage without changing logical ids."""
    return quote(value, safe="")


def _serialize_trace(trace: TokentrimTraceRecord) -> dict[str, object]:
    """Serialize a finalized trace record."""
    return {
        "trace_id": trace.trace_id,
        "source": trace.source,
        "capture_mode": trace.capture_mode,
        "source_trace_id": trace.source_trace_id,
        "user_id": trace.user_id,
        "session_id": trace.session_id,
        "workflow_name": trace.workflow_name,
        "started_at": trace.started_at,
        "ended_at": trace.ended_at,
        "group_id": trace.group_id,
        "metadata": trace.metadata,
        "raw_trace": trace.raw_trace,
        "spans": [_serialize_span(span) for span in trace.spans],
    }


def _deserialize_trace(payload: dict[str, object]) -> TokentrimTraceRecord:
    """Deserialize a persisted trace record."""
    return TokentrimTraceRecord(
        trace_id=str(payload["trace_id"]),
        source=str(payload["source"]),
        capture_mode=str(payload["capture_mode"]),
        source_trace_id=payload.get("source_trace_id") if isinstance(payload.get("source_trace_id"), str) else None,
        user_id=str(payload["user_id"]),
        session_id=str(payload["session_id"]),
        workflow_name=str(payload["workflow_name"]),
        started_at=payload.get("started_at") if isinstance(payload.get("started_at"), str) else None,
        ended_at=payload.get("ended_at") if isinstance(payload.get("ended_at"), str) else None,
        group_id=payload.get("group_id") if isinstance(payload.get("group_id"), str) else None,
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
        raw_trace=payload.get("raw_trace") if isinstance(payload.get("raw_trace"), dict) else {},
        spans=tuple(
            _deserialize_span(span_payload)
            for span_payload in payload.get("spans", [])
            if isinstance(span_payload, dict)
        ),
    )


def _serialize_span(span: TokentrimSpanRecord) -> dict[str, object]:
    """Serialize one trace span."""
    return {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "source": span.source,
        "kind": span.kind,
        "name": span.name,
        "source_span_id": span.source_span_id,
        "parent_id": span.parent_id,
        "started_at": span.started_at,
        "ended_at": span.ended_at,
        "error": span.error,
        "metrics": span.metrics,
        "data": span.data,
        "raw_span": span.raw_span,
    }


def _deserialize_span(payload: dict[str, object]) -> TokentrimSpanRecord:
    """Deserialize one persisted span."""
    return TokentrimSpanRecord(
        span_id=str(payload["span_id"]),
        trace_id=str(payload["trace_id"]),
        source=str(payload["source"]),
        kind=str(payload["kind"]),
        name=payload.get("name") if isinstance(payload.get("name"), str) else None,
        source_span_id=payload.get("source_span_id") if isinstance(payload.get("source_span_id"), str) else None,
        parent_id=payload.get("parent_id") if isinstance(payload.get("parent_id"), str) else None,
        started_at=payload.get("started_at") if isinstance(payload.get("started_at"), str) else None,
        ended_at=payload.get("ended_at") if isinstance(payload.get("ended_at"), str) else None,
        error=payload.get("error") if isinstance(payload.get("error"), dict) else None,
        metrics=payload.get("metrics") if isinstance(payload.get("metrics"), dict) else None,
        data=payload.get("data") if isinstance(payload.get("data"), dict) else {},
        raw_span=payload.get("raw_span") if isinstance(payload.get("raw_span"), dict) else {},
    )
