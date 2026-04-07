from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from tokentrim.memory.types import sanitize_memory_segment
from tokentrim.tracing.atif import export_trace_to_atif, load_trace_from_atif
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


class _ActiveTraceStore(TraceStore, ABC):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_traces: dict[str, _ActiveTraceState] = {}

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
            completed_trace = _complete_active_trace(active_trace)
        self._persist_completed_trace(
            user_id=active_trace.user_id,
            session_id=active_trace.session_id,
            trace=completed_trace,
        )

    @abstractmethod
    def _persist_completed_trace(
        self,
        *,
        user_id: str,
        session_id: str,
        trace: TokentrimTraceRecord,
    ) -> None:
        """Persist a completed trace for future reads."""


class InMemoryTraceStore(_ActiveTraceStore):
    """Process-local storage for canonical Tokentrim trace history."""

    def __init__(self) -> None:
        super().__init__()
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

    def _persist_completed_trace(
        self,
        *,
        user_id: str,
        session_id: str,
        trace: TokentrimTraceRecord,
    ) -> None:
        with self._lock:
            self._completed_traces[trace.trace_id] = trace
            self._trace_index[(user_id, session_id)].append(trace.trace_id)


class FileSystemTraceStore(_ActiveTraceStore):
    """Filesystem-backed trace history persisted as ATIF JSON.

    Persisted files are ATIF-shaped for interoperability, but Tokentrim still
    treats the embedded canonical trace metadata as the authoritative
    round-trip source.
    """

    def __init__(self, *, root_dir: str | Path | None = None) -> None:
        super().__init__()
        self._root_dir = Path(root_dir) if root_dir is not None else Path.cwd() / ".tokentrim" / "traces"

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def list_traces(
        self,
        *,
        user_id: str,
        session_id: str,
        limit: int | None = None,
    ) -> tuple[TokentrimTraceRecord, ...]:
        session_dir = self._session_dir(user_id=user_id, session_id=session_id)
        if not session_dir.exists():
            return ()
        files = sorted(session_dir.glob("*.atif.json"), reverse=True)
        if limit is not None and limit <= 0:
            return ()
        if limit is not None:
            files = files[:limit]
        traces: list[TokentrimTraceRecord] = []
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            traces.append(load_trace_from_atif(payload))
        return tuple(traces)

    def _persist_completed_trace(
        self,
        *,
        user_id: str,
        session_id: str,
        trace: TokentrimTraceRecord,
    ) -> None:
        session_dir = self._session_dir(user_id=user_id, session_id=session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / self._trace_filename(trace)
        temp_path = path.with_name(f".{path.name}.tmp")
        temp_path.write_text(
            json.dumps(export_trace_to_atif(trace), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def _session_dir(self, *, user_id: str, session_id: str) -> Path:
        return (
            self._root_dir
            / "users"
            / sanitize_memory_segment(user_id)
            / "sessions"
            / sanitize_memory_segment(session_id)
        )

    def _trace_filename(self, trace: TokentrimTraceRecord) -> str:
        timestamp = (trace.ended_at or trace.started_at or "unknown").replace(":", "-").replace("+", "_")
        trace_id = sanitize_memory_segment(trace.trace_id)
        return f"{timestamp}_{trace_id}.atif.json"


def _complete_active_trace(active_trace: _ActiveTraceState) -> TokentrimTraceRecord:
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
    return replace(
        active_trace.trace,
        started_at=started_at,
        ended_at=ended_at,
        spans=ordered_spans,
    )
