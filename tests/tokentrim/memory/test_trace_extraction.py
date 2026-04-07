from __future__ import annotations

from tokentrim.memory import build_trace_memory_candidate
from tokentrim.tracing import InMemoryTraceStore, TokentrimSpanRecord, TokentrimTraceRecord


def _trace(trace_id: str, *, ended_at: str, spans: tuple[TokentrimSpanRecord, ...]) -> TokentrimTraceRecord:
    return TokentrimTraceRecord(
        trace_id=trace_id,
        source="openai_agents",
        capture_mode="identity",
        source_trace_id=trace_id.removeprefix("openai_agents:"),
        user_id="u1",
        session_id="s1",
        workflow_name="Agent workflow",
        started_at=None,
        ended_at=ended_at,
        group_id=None,
        metadata={"source": "test"},
        raw_trace={"id": trace_id.removeprefix("openai_agents:")},
        spans=spans,
    )


def _error_span(span_id: str, trace_id: str, *, message: str) -> TokentrimSpanRecord:
    return TokentrimSpanRecord(
        span_id=span_id,
        trace_id=trace_id,
        source="openai_agents",
        kind="tool",
        name="run command",
        source_span_id=span_id.removeprefix("openai_agents:"),
        parent_id=None,
        started_at="2026-04-03T10:00:00+00:00",
        ended_at="2026-04-03T10:00:01+00:00",
        error={"message": message},
        metrics=None,
        data={"command": "pytest ./tokentrim/README.md"},
        raw_span={"path": "./tokentrim/README.md"},
    )


def _ok_span(span_id: str, trace_id: str) -> TokentrimSpanRecord:
    return TokentrimSpanRecord(
        span_id=span_id,
        trace_id=trace_id,
        source="openai_agents",
        kind="tool",
        name="run command",
        source_span_id=span_id.removeprefix("openai_agents:"),
        parent_id=None,
        started_at="2026-04-03T10:00:02+00:00",
        ended_at="2026-04-03T10:00:03+00:00",
        error=None,
        metrics=None,
        data={"command": "pytest ./tokentrim/README.md"},
        raw_span={"path": "./tokentrim/README.md"},
    )


def test_build_trace_memory_candidate_detects_repeated_failure() -> None:
    traces = (
        _trace(
            "openai_agents:trace_2",
            ended_at="2026-04-03T10:00:03+00:00",
            spans=(
                _error_span(
                    "openai_agents:span_2",
                    "openai_agents:trace_2",
                    message="FileNotFoundError: missing fixture",
                ),
            ),
        ),
        _trace(
            "openai_agents:trace_1",
            ended_at="2026-04-03T10:00:01+00:00",
            spans=(
                _error_span(
                    "openai_agents:span_1",
                    "openai_agents:trace_1",
                    message="FileNotFoundError: missing fixture",
                ),
            ),
        ),
    )

    candidate = build_trace_memory_candidate(traces, task_hint="fix README pytest")

    assert candidate is not None
    assert candidate.metadata["kind"] == "trace_repeated_error"
    assert "Repeated failure observed across 2 traces" in candidate.content
    assert "./tokentrim/README.md" in candidate.content


def test_build_trace_memory_candidate_detects_resolved_repeated_failure() -> None:
    traces = (
        _trace(
            "openai_agents:trace_3",
            ended_at="2026-04-03T10:00:05+00:00",
            spans=(_ok_span("openai_agents:span_3", "openai_agents:trace_3"),),
        ),
        _trace(
            "openai_agents:trace_2",
            ended_at="2026-04-03T10:00:03+00:00",
            spans=(
                _error_span(
                    "openai_agents:span_2",
                    "openai_agents:trace_2",
                    message="FileNotFoundError: missing fixture",
                ),
            ),
        ),
        _trace(
            "openai_agents:trace_1",
            ended_at="2026-04-03T10:00:01+00:00",
            spans=(
                _error_span(
                    "openai_agents:span_1",
                    "openai_agents:trace_1",
                    message="FileNotFoundError: missing fixture",
                ),
            ),
        ),
    )

    candidate = build_trace_memory_candidate(traces, task_hint="fix README pytest")

    assert candidate is not None
    assert candidate.metadata["kind"] == "trace_resolution"
    assert "Resolved prior repeated failure" in candidate.content


def test_build_trace_memory_candidate_returns_none_without_repetition() -> None:
    traces = (
        _trace(
            "openai_agents:trace_1",
            ended_at="2026-04-03T10:00:01+00:00",
            spans=(
                _error_span(
                    "openai_agents:span_1",
                    "openai_agents:trace_1",
                    message="ValueError: one-off issue",
                ),
            ),
        ),
    )

    assert build_trace_memory_candidate(traces, task_hint="investigate") is None
