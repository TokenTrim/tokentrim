from __future__ import annotations

import json
from pathlib import Path

from tokentrim.cli import main
from tokentrim.memory import FilesystemMemoryStore
from tokentrim.tracing import FilesystemTraceStore, TokentrimSpanRecord, TokentrimTraceRecord


def _seed_trace(store: FilesystemTraceStore, *, trace_id: str, user_id: str = "user_1", session_id: str = "session_1") -> None:
    trace = TokentrimTraceRecord(
        trace_id=trace_id,
        source="openai_agents",
        capture_mode="identity",
        source_trace_id=trace_id.removeprefix("openai_agents:"),
        user_id=user_id,
        session_id=session_id,
        workflow_name="code_agent",
        started_at=None,
        ended_at=None,
        group_id=None,
        metadata={"source": "test"},
        raw_trace={"id": trace_id.removeprefix("openai_agents:")},
    )
    store.create_trace(user_id=user_id, session_id=session_id, trace=trace)
    store.append_span(
        trace_id=trace_id,
        span=TokentrimSpanRecord(
            span_id=f"{trace_id}:fail",
            trace_id=trace_id,
            source="openai_agents",
            kind="tool_call",
            name="run_tests",
            source_span_id="fail",
            parent_id=None,
            started_at="2026-04-13T10:00:00Z",
            ended_at="2026-04-13T10:00:01Z",
            error={"type": "CommandError", "message": "pytest missing"},
            metrics=None,
            data={"name": "run_tests"},
            raw_span={"id": "fail"},
        ),
    )
    store.append_span(
        trace_id=trace_id,
        span=TokentrimSpanRecord(
            span_id=f"{trace_id}:fix",
            trace_id=trace_id,
            source="openai_agents",
            kind="tool_call",
            name="install_pytest_and_rerun",
            source_span_id="fix",
            parent_id=None,
            started_at="2026-04-13T10:00:02Z",
            ended_at="2026-04-13T10:00:03Z",
            error=None,
            metrics=None,
            data={"name": "install_pytest_and_rerun"},
            raw_span={"id": "fix"},
        ),
    )
    store.complete_trace(trace_id=trace_id)


def test_cli_consolidate_applies_deterministic_plan(capsys, tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    trace_dir = tmp_path / "traces"
    trace_store = FilesystemTraceStore(root_dir=trace_dir)
    _seed_trace(trace_store, trace_id="openai_agents:trace_1")
    _seed_trace(trace_store, trace_id="openai_agents:trace_2")

    exit_code = main(
        [
            "consolidate",
            "--memory-dir",
            str(memory_dir),
            "--trace-dir",
            str(trace_dir),
            "--mode",
            "deterministic",
            "--scope",
            "all",
            "--user-id",
            "user_1",
            "--session-id",
            "session_1",
            "--org-id",
            "org_1",
            "--apply",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["apply_result"]["upserted"]
    stored = FilesystemMemoryStore(root_dir=memory_dir).list_memories(scope="org", subject_id="org_1")
    assert len(stored) == 1
    assert stored[0].kind == "failure_recovery"


def test_cli_consolidate_scope_user_does_not_write_org(capsys, tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    trace_dir = tmp_path / "traces"
    trace_store = FilesystemTraceStore(root_dir=trace_dir)
    _seed_trace(trace_store, trace_id="openai_agents:trace_1")
    _seed_trace(trace_store, trace_id="openai_agents:trace_2")

    exit_code = main(
        [
            "consolidate",
            "--memory-dir",
            str(memory_dir),
            "--trace-dir",
            str(trace_dir),
            "--mode",
            "deterministic",
            "--scope",
            "user",
            "--user-id",
            "user_1",
            "--session-id",
            "session_1",
            "--org-id",
            "org_1",
            "--apply",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["plan"]["org_upserts"] == 0
    stored = FilesystemMemoryStore(root_dir=memory_dir).list_memories(scope="org", subject_id="org_1")
    assert stored == ()


def test_cli_consolidate_batch_mode_scans_completed_traces(capsys, tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    trace_dir = tmp_path / "traces"
    trace_store = FilesystemTraceStore(root_dir=trace_dir)
    _seed_trace(trace_store, trace_id="openai_agents:trace_1", user_id="user_1", session_id="session_1")
    _seed_trace(trace_store, trace_id="openai_agents:trace_2", user_id="user_1", session_id="session_1")

    exit_code = main(
        [
            "consolidate",
            "--memory-dir",
            str(memory_dir),
            "--trace-dir",
            str(trace_dir),
            "--mode",
            "deterministic",
            "--org-id",
            "org_1",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["user_id"] == "user_1"
    assert output["session_id"] == "session_1"
    assert output["applied"] is False
