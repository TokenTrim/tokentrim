from __future__ import annotations

from tokentrim.consolidator import (
    ConsolidationInput,
    TRACE_FAILURE_RECOVERY_KIND,
    TRACE_WORKFLOW_PATTERN_KIND,
    synthesize_trace_memory_plan,
)
from tokentrim.memory import MemoryRecord
from tokentrim.tracing import TokentrimSpanRecord, TokentrimTraceRecord


def _trace(
    trace_id: str,
    *,
    workflow_name: str = "code_agent",
    spans: tuple[TokentrimSpanRecord, ...],
) -> TokentrimTraceRecord:
    return TokentrimTraceRecord(
        trace_id=trace_id,
        source="openai_agents",
        capture_mode="identity",
        source_trace_id=trace_id.removeprefix("openai_agents:"),
        user_id="user_1",
        session_id="session_1",
        workflow_name=workflow_name,
        started_at="2026-04-13T10:00:00Z",
        ended_at="2026-04-13T10:01:00Z",
        group_id=None,
        metadata={"source": "test"},
        raw_trace={"id": trace_id.removeprefix("openai_agents:")},
        spans=spans,
    )


def _span(
    span_id: str,
    *,
    kind: str,
    name: str,
    error: dict[str, object] | None = None,
    parent_id: str | None = None,
) -> TokentrimSpanRecord:
    return TokentrimSpanRecord(
        span_id=span_id,
        trace_id="openai_agents:trace_shared",
        source="openai_agents",
        kind=kind,
        name=name,
        source_span_id=span_id.removeprefix("openai_agents:"),
        parent_id=parent_id,
        started_at="2026-04-13T10:00:00Z",
        ended_at="2026-04-13T10:00:01Z",
        error=error,
        metrics=None,
        data={"name": name},
        raw_span={"id": span_id.removeprefix("openai_agents:")},
    )


def test_synthesize_trace_memory_plan_promotes_repeated_failure_recovery_to_org() -> None:
    failure_a = _span(
        "openai_agents:span_fail_1",
        kind="tool_call",
        name="run_tests",
        error={"type": "CommandError", "message": "pytest missing"},
    )
    recovery_a = _span(
        "openai_agents:span_fix_1",
        kind="tool_call",
        name="install_pytest_and_rerun",
    )
    failure_b = _span(
        "openai_agents:span_fail_2",
        kind="tool_call",
        name="run_tests",
        error={"type": "CommandError", "message": "pytest missing"},
    )
    recovery_b = _span(
        "openai_agents:span_fix_2",
        kind="tool_call",
        name="install_pytest_and_rerun",
    )

    plan = synthesize_trace_memory_plan(
        ConsolidationInput(
            session_id="session_1",
            user_id="user_1",
            org_id="org_1",
            traces=(
                _trace("openai_agents:trace_1", spans=(failure_a, recovery_a)),
                _trace("openai_agents:trace_2", spans=(failure_b, recovery_b)),
            ),
            session_memories=(),
            user_memories=(),
            org_memories=(),
        )
    )

    assert len(plan.org_upserts) == 1
    assert not plan.user_upserts
    org_write = plan.org_upserts[0].write
    assert org_write.kind == TRACE_FAILURE_RECOVERY_KIND
    assert org_write.dedupe_key == (
        "trace:org:code_agent|run_tests|CommandError: pytest missing|install_pytest_and_rerun"
    )
    assert "recovery path that succeeded was 'install_pytest_and_rerun'" in org_write.content
    assert set(org_write.source_refs) == {
        "openai_agents:trace_1",
        "openai_agents:trace_2",
        "openai_agents:span_fail_1",
        "openai_agents:span_fail_2",
        "openai_agents:span_fix_1",
        "openai_agents:span_fix_2",
    }
    assert any("failure/recovery pattern" in item for item in plan.rationale)


def test_synthesize_trace_memory_plan_promotes_single_failure_recovery_to_user() -> None:
    failure = _span(
        "openai_agents:span_fail",
        kind="tool_call",
        name="open_config",
        error={"type": "FileNotFoundError", "message": "settings.toml"},
    )
    recovery = _span(
        "openai_agents:span_fix",
        kind="tool_call",
        name="locate_correct_config",
    )

    plan = synthesize_trace_memory_plan(
        ConsolidationInput(
            session_id="session_1",
            user_id="user_1",
            org_id="org_1",
            traces=(_trace("openai_agents:trace_1", spans=(failure, recovery)),),
            session_memories=(),
            user_memories=(),
            org_memories=(),
        )
    )

    assert len(plan.user_upserts) == 1
    assert not plan.org_upserts
    user_write = plan.user_upserts[0].write
    assert user_write.kind == TRACE_FAILURE_RECOVERY_KIND
    assert "In this user's traces" in user_write.content
    assert user_write.metadata["evidence_count"] == 1


def test_synthesize_trace_memory_plan_skips_existing_dedupe_keys() -> None:
    failure = _span(
        "openai_agents:span_fail",
        kind="tool_call",
        name="run_tests",
        error={"type": "CommandError", "message": "pytest missing"},
    )
    recovery = _span(
        "openai_agents:span_fix",
        kind="tool_call",
        name="install_pytest_and_rerun",
    )
    existing = MemoryRecord(
        memory_id="mem_org_existing",
        scope="org",
        subject_id="org_1",
        kind=TRACE_FAILURE_RECOVERY_KIND,
        content="Existing pattern",
        dedupe_key="trace:org:code_agent|run_tests|CommandError: pytest missing|install_pytest_and_rerun",
    )

    plan = synthesize_trace_memory_plan(
        ConsolidationInput(
            session_id="session_1",
            user_id="user_1",
            org_id="org_1",
            traces=(
                _trace("openai_agents:trace_1", spans=(failure, recovery)),
                _trace("openai_agents:trace_2", spans=(failure, recovery)),
            ),
            session_memories=(),
            user_memories=(),
            org_memories=(existing,),
        )
    )

    assert not plan.org_upserts


def test_synthesize_trace_memory_plan_promotes_repeated_successful_workflow_to_org() -> None:
    trace_a = _trace(
        "openai_agents:trace_1",
        workflow_name="shipping",
        spans=(
            _span("openai_agents:span_1", kind="tool_call", name="run_tests"),
            _span("openai_agents:span_2", kind="tool_call", name="review_diff"),
            _span("openai_agents:span_3", kind="tool_call", name="commit_changes"),
        ),
    )
    trace_b = _trace(
        "openai_agents:trace_2",
        workflow_name="shipping",
        spans=(
            _span("openai_agents:span_4", kind="tool_call", name="run_tests"),
            _span("openai_agents:span_5", kind="tool_call", name="review_diff"),
            _span("openai_agents:span_6", kind="tool_call", name="commit_changes"),
        ),
    )

    plan = synthesize_trace_memory_plan(
        ConsolidationInput(
            session_id="session_1",
            user_id="user_1",
            org_id="org_1",
            traces=(trace_a, trace_b),
            session_memories=(),
            user_memories=(),
            org_memories=(),
        )
    )

    assert len(plan.org_upserts) == 1
    workflow_write = plan.org_upserts[0].write
    assert workflow_write.kind == TRACE_WORKFLOW_PATTERN_KIND
    assert "run_tests -> review_diff -> commit_changes" in workflow_write.content
    assert workflow_write.metadata["evidence_count"] == 2
