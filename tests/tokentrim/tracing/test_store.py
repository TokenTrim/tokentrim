from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tokentrim.tracing import FilesystemTraceStore, InMemoryTraceStore, TokentrimSpanRecord, TokentrimTraceRecord


def _trace(trace_id: str, workflow_name: str = "Agent workflow") -> TokentrimTraceRecord:
    return TokentrimTraceRecord(
        trace_id=trace_id,
        source="openai_agents",
        capture_mode="identity",
        source_trace_id=trace_id.removeprefix("openai_agents:"),
        user_id="u1",
        session_id="s1",
        workflow_name=workflow_name,
        started_at=None,
        ended_at=None,
        group_id=None,
        metadata={"source": "test"},
        raw_trace={"id": trace_id.removeprefix("openai_agents:")},
    )


def _span(
    span_id: str,
    trace_id: str,
    *,
    parent_id: str | None = None,
    started_at: str,
    ended_at: str,
) -> TokentrimSpanRecord:
    return TokentrimSpanRecord(
        span_id=span_id,
        trace_id=trace_id,
        source="openai_agents",
        kind="custom",
        name=span_id,
        source_span_id=span_id.removeprefix("openai_agents:"),
        parent_id=parent_id,
        started_at=started_at,
        ended_at=ended_at,
        error=None,
        metrics=None,
        data={"name": span_id},
        raw_span={"id": span_id.removeprefix("openai_agents:")},
    )


def test_in_memory_trace_store_completes_trace_with_immutable_record() -> None:
    store = InMemoryTraceStore()
    trace = _trace("openai_agents:trace_1")

    store.create_trace(user_id="u1", session_id="s1", trace=trace)
    store.append_span(
        trace_id="openai_agents:trace_1",
        span=_span(
            "openai_agents:span_1",
            "openai_agents:trace_1",
            started_at="2026-04-03T10:00:00+00:00",
            ended_at="2026-04-03T10:00:01+00:00",
        ),
    )
    store.complete_trace(trace_id="openai_agents:trace_1")

    traces = store.list_traces(user_id="u1", session_id="s1")

    assert len(traces) == 1
    assert traces[0].trace_id == "openai_agents:trace_1"
    assert traces[0].source_trace_id == "trace_1"
    assert traces[0].spans[0].span_id == "openai_agents:span_1"
    assert traces[0].spans[0].source_span_id == "span_1"
    assert traces[0].started_at == "2026-04-03T10:00:00+00:00"
    assert traces[0].ended_at == "2026-04-03T10:00:01+00:00"
    assert traces[0].raw_trace == {"id": "trace_1"}
    assert traces[0].spans[0].raw_span == {"id": "span_1"}
    with pytest.raises((FrozenInstanceError, TypeError)):
        traces[0].trace_id = "other"


def test_in_memory_trace_store_isolates_user_session_scope() -> None:
    store = InMemoryTraceStore()

    store.create_trace(user_id="u1", session_id="s1", trace=_trace("openai_agents:trace_1"))
    store.complete_trace(trace_id="openai_agents:trace_1")
    store.create_trace(
        user_id="u2",
        session_id="s2",
        trace=TokentrimTraceRecord(
            trace_id="openai_agents:trace_2",
            source="openai_agents",
            capture_mode="identity",
            source_trace_id="trace_2",
            user_id="u2",
            session_id="s2",
            workflow_name="Agent workflow",
            started_at=None,
            ended_at=None,
            group_id=None,
            metadata={"source": "test"},
            raw_trace={"id": "trace_2"},
        ),
    )
    store.complete_trace(trace_id="openai_agents:trace_2")

    assert [trace.trace_id for trace in store.list_traces(user_id="u1", session_id="s1")] == [
        "openai_agents:trace_1"
    ]
    assert [trace.trace_id for trace in store.list_traces(user_id="u2", session_id="s2")] == [
        "openai_agents:trace_2"
    ]


def test_in_memory_trace_store_returns_newest_traces_and_chronological_spans() -> None:
    store = InMemoryTraceStore()

    store.create_trace(user_id="u1", session_id="s1", trace=_trace("openai_agents:trace_1"))
    store.append_span(
        trace_id="openai_agents:trace_1",
        span=_span(
            "openai_agents:span_late",
            "openai_agents:trace_1",
            started_at="2026-04-03T10:00:05+00:00",
            ended_at="2026-04-03T10:00:06+00:00",
        ),
    )
    store.append_span(
        trace_id="openai_agents:trace_1",
        span=_span(
            "openai_agents:span_early",
            "openai_agents:trace_1",
            started_at="2026-04-03T10:00:01+00:00",
            ended_at="2026-04-03T10:00:02+00:00",
        ),
    )
    store.complete_trace(trace_id="openai_agents:trace_1")

    store.create_trace(user_id="u1", session_id="s1", trace=_trace("openai_agents:trace_2"))
    store.complete_trace(trace_id="openai_agents:trace_2")

    traces = store.list_traces(user_id="u1", session_id="s1")

    assert [trace.trace_id for trace in traces] == [
        "openai_agents:trace_2",
        "openai_agents:trace_1",
    ]
    assert [span.span_id for span in traces[1].spans] == [
        "openai_agents:span_early",
        "openai_agents:span_late",
    ]
    assert traces[1].started_at == "2026-04-03T10:00:01+00:00"
    assert traces[1].ended_at == "2026-04-03T10:00:06+00:00"


def test_filesystem_trace_store_persists_completed_trace_round_trip(tmp_path: Path) -> None:
    store = FilesystemTraceStore(root_dir=tmp_path / "traces")
    trace = _trace("openai_agents:trace_1")

    store.create_trace(user_id="u1", session_id="s1", trace=trace)
    store.append_span(
        trace_id="openai_agents:trace_1",
        span=_span(
            "openai_agents:span_1",
            "openai_agents:trace_1",
            started_at="2026-04-03T10:00:00+00:00",
            ended_at="2026-04-03T10:00:01+00:00",
        ),
    )
    store.complete_trace(trace_id="openai_agents:trace_1")

    reloaded_store = FilesystemTraceStore(root_dir=tmp_path / "traces")
    traces = reloaded_store.list_traces(user_id="u1", session_id="s1")

    assert len(traces) == 1
    assert traces[0].trace_id == "openai_agents:trace_1"
    assert traces[0].spans[0].span_id == "openai_agents:span_1"
    assert traces[0].raw_trace == {"id": "trace_1"}


def test_filesystem_trace_store_persists_active_trace_across_reloads(tmp_path: Path) -> None:
    root_dir = tmp_path / "traces"
    store = FilesystemTraceStore(root_dir=root_dir)
    store.create_trace(user_id="u1", session_id="s1", trace=_trace("openai_agents:trace_1"))
    store.append_span(
        trace_id="openai_agents:trace_1",
        span=_span(
            "openai_agents:span_1",
            "openai_agents:trace_1",
            started_at="2026-04-03T10:00:00+00:00",
            ended_at="2026-04-03T10:00:01+00:00",
        ),
    )

    reloaded_store = FilesystemTraceStore(root_dir=root_dir)
    reloaded_store.complete_trace(trace_id="openai_agents:trace_1")
    traces = reloaded_store.list_traces(user_id="u1", session_id="s1")

    assert len(traces) == 1
    assert traces[0].spans[0].span_id == "openai_agents:span_1"
    assert list((root_dir / "active").glob("*.json")) == []
