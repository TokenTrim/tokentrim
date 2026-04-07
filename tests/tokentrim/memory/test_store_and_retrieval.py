from __future__ import annotations

from pathlib import Path

import pytest

from tokentrim.memory import (
    DefaultMemoryWritePolicy,
    LocalDirectoryMemoryStore,
    MemoryIndexRecord,
    default_memory_root,
    load_index,
    read_memory_markdown,
    write_index,
    write_memory_markdown,
)
from tokentrim.memory.types import MemoryScope
from tokentrim.pipeline.requests import ContextRequest
from tokentrim.tracing import InMemoryTraceStore, TokentrimSpanRecord, TokentrimTraceRecord
from tokentrim.transforms.remember_durable_memory import RememberDurableMemory
from tokentrim.transforms.retrieve_durable_memory import RetrieveDurableMemory
from tokentrim.types.state import PipelineState


def _request(*, user_id: str | None, session_id: str | None, task_hint: str | None = None) -> ContextRequest:
    request = ContextRequest(
        messages=tuple(),
        user_id=user_id,
        session_id=session_id,
        token_budget=None,
        steps=(RetrieveDurableMemory(),),
    )
    object.__setattr__(request, "task_hint", task_hint)
    return request


def _request_with_trace_store(
    *,
    user_id: str,
    session_id: str,
    trace_store: InMemoryTraceStore,
    task_hint: str | None = None,
) -> ContextRequest:
    request = _request(user_id=user_id, session_id=session_id, task_hint=task_hint)
    object.__setattr__(request, "trace_store", trace_store)
    return request


def _trace(trace_id: str, *, ended_at: str, spans: tuple[TokentrimSpanRecord, ...]) -> TokentrimTraceRecord:
    return TokentrimTraceRecord(
        trace_id=trace_id,
        source="openai_agents",
        capture_mode="identity",
        source_trace_id=trace_id.removeprefix("openai_agents:"),
        user_id="user",
        session_id="session",
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


def test_local_directory_memory_store_persists_and_ranks_relevant_entries(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    store.remember(
        user_id="user",
        session_id="session",
        content="The active file is ./tokentrim/README.md and pytest fails on missing fixture.",
    )
    store.remember(
        user_id="user",
        session_id="session",
        content="Unrelated note about docker compose setup.",
    )

    entries = store.retrieve(
        user_id="user",
        session_id="session",
        query="fix ./tokentrim/README.md missing fixture",
    )

    assert entries
    assert "./tokentrim/README.md" in entries[0].content
    index_path = store.session_index_path(user_id="user", session_id="session")
    assert index_path == tmp_path / "users" / "user" / "sessions" / "session" / "index.jsonl"
    records = load_index(index_path)
    assert len(records) == 2
    assert records[0].path is not None
    markdown_path = index_path.parent / records[0].path
    assert markdown_path.exists()
    payload = read_memory_markdown(markdown_path)
    assert payload["content"] == records[0].content


def test_retrieve_durable_memory_inserts_after_leading_system_messages(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    store.remember(
        user_id="user",
        session_id="session",
        content="Keep using ./tokentrim/README.md for the docs update.",
    )
    step = RetrieveDurableMemory(memory_store=store)
    messages = [
        {"role": "system", "content": "Working state only.\nGoal: update docs"},
        {"role": "system", "content": "History only.\n\nGoal:\nupdate docs"},
        {"role": "user", "content": "Please update ./tokentrim/README.md."},
    ]

    result = step.run(
        PipelineState(context=messages, tools=[]),
        _request(user_id="user", session_id="session"),
    )

    assert result.context[2]["role"] == "system"
    assert result.context[2]["content"].startswith("Durable memory only.")
    assert "./tokentrim/README.md" in result.context[2]["content"]


def test_retrieve_durable_memory_uses_parsed_working_state_not_literal_prefix(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    store.remember(
        user_id="user",
        session_id="session",
        content="Checkpoint: rerun pytest for ./tokentrim/README.md after fixing docs.",
    )
    step = RetrieveDurableMemory(memory_store=store)
    messages = [
        {
            "role": "system",
            "content": (
                "Working state only.\n"
                "Goal: update docs\n"
                "Active Files: ./tokentrim/README.md\n"
                "Latest Command: pytest tests/tokentrim/test_client.py\n"
            ),
        },
        {"role": "user", "content": "continue"},
    ]

    result = step.run(
        PipelineState(context=messages, tools=[]),
        _request(user_id="user", session_id="session"),
    )

    assert result.context[1]["content"].startswith("Durable memory only.")
    assert "./tokentrim/README.md" in result.context[1]["content"]


def test_retrieve_durable_memory_is_noop_without_identifiers(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    step = RetrieveDurableMemory(memory_store=store)
    messages = [{"role": "user", "content": "hello"}]

    result = step.run(
        PipelineState(context=messages, tools=[]),
        _request(user_id=None, session_id="session"),
    )

    assert result.context == messages


def test_retrieve_durable_memory_uses_task_hint_for_query(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    store.remember(
        user_id="user",
        session_id="session",
        content="Task fact: target command is pytest tests/tokentrim/transforms/compaction/test_compaction_transform.py.",
    )
    step = RetrieveDurableMemory(memory_store=store)

    result = step.run(
        PipelineState(context=[{"role": "user", "content": "continue"}], tools=[]),
        _request(
            user_id="user",
            session_id="session",
            task_hint="rerun pytest tests/tokentrim/transforms/compaction/test_compaction_transform.py",
        ),
    )

    assert "pytest tests/tokentrim/transforms/compaction/test_compaction_transform.py" in result.context[0]["content"]


def test_local_directory_memory_store_prefers_active_error_artifact_over_recent_generic_note(
    tmp_path: Path,
) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    store.remember(
        user_id="user",
        session_id="session",
        content="Generic note about continuing work tomorrow.",
    )
    store.remember(
        user_id="user",
        session_id="session",
        content="FileNotFoundError: missing fixture in ./tokentrim/README.md after pytest.",
    )

    entries = store.retrieve(
        user_id="user",
        session_id="session",
        query="fix ./tokentrim/README.md missing fixture",
    )

    assert entries
    assert "FileNotFoundError: missing fixture" in entries[0].content


def test_default_memory_root_uses_hidden_tokentrim_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    assert default_memory_root() == tmp_path / ".tokentrim" / "memory"
    assert LocalDirectoryMemoryStore().root_dir == tmp_path / ".tokentrim" / "memory"


def test_index_round_trip_preserves_scope_and_keywords(tmp_path: Path) -> None:
    index_path = tmp_path / "users" / "user" / "sessions" / "session" / "index.jsonl"
    records = (
        MemoryIndexRecord(
            entry_id="mem_01",
            path="entries/mem_01.md",
            content="Remember ./tokentrim/README.md",
            created_at="2026-04-06T13:00:00+00:00",
            metadata={"kind": "checkpoint"},
            keywords=("./tokentrim/README.md", "pytest"),
            scope=MemoryScope.SESSION,
        ),
    )

    write_index(index_path, records)
    loaded = load_index(index_path)

    assert loaded == records


def test_local_directory_memory_store_supports_user_and_project_scopes(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    store.remember(
        user_id="user",
        session_id="session-a",
        content="User preference: prefer pytest -q.",
        scope=MemoryScope.USER,
    )
    store.remember(
        user_id="user",
        session_id="session-a",
        content="Project fact: docs live in ./tokentrim/README.md.",
        scope=MemoryScope.PROJECT,
        project_id="tokentrim-project",
    )

    user_records = load_index(store.user_index_path(user_id="user"))
    project_records = load_index(store.project_index_path(project_id="tokentrim-project"))

    assert user_records[0].scope is MemoryScope.USER
    assert project_records[0].scope is MemoryScope.PROJECT

    entries = store.retrieve(
        user_id="user",
        session_id="session-b",
        query="pytest README",
        scopes=(MemoryScope.USER, MemoryScope.PROJECT),
        project_id="tokentrim-project",
    )

    assert len(entries) == 2
    assert {entry.scope for entry in entries} == {MemoryScope.USER, MemoryScope.PROJECT}


def test_markdown_memory_round_trip_preserves_frontmatter_and_content(tmp_path: Path) -> None:
    path = tmp_path / "entries" / "mem_01.md"
    write_memory_markdown(
        path,
        entry_id="mem_01",
        scope=MemoryScope.SESSION,
        created_at="2026-04-06T13:00:00+00:00",
        keywords=("pytest", "./tokentrim/README.md"),
        metadata={"kind": "checkpoint"},
        content="Use ./tokentrim/README.md for the docs update.",
    )

    payload = read_memory_markdown(path)

    assert payload["id"] == "mem_01"
    assert payload["scope"] == "session"
    assert payload["keywords"] == ("pytest", "./tokentrim/README.md")
    assert payload["metadata_json"] == {"kind": "checkpoint"}
    assert payload["content"] == "Use ./tokentrim/README.md for the docs update."


def test_retrieve_skips_missing_markdown_file_for_stale_index_entry(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    index_path = store.session_index_path(user_id="user", session_id="session")
    write_index(
        index_path,
        (
            MemoryIndexRecord(
                entry_id="mem_missing",
                path="entries/mem_missing.md",
                content="stale content",
                created_at="2026-04-06T13:00:00+00:00",
                metadata={"kind": "checkpoint"},
                keywords=("pytest",),
                scope=MemoryScope.SESSION,
            ),
        ),
    )

    entries = store.retrieve(user_id="user", session_id="session", query="pytest")

    assert entries == ()


def test_retrieve_uses_edited_markdown_content_instead_of_stale_index_snapshot(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    store.remember(
        user_id="user",
        session_id="session",
        content="Old note about generic docs work.",
        metadata={"kind": "checkpoint"},
    )
    index_path = store.session_index_path(user_id="user", session_id="session")
    record = load_index(index_path)[0]
    markdown_path = index_path.parent / str(record.path)
    write_memory_markdown(
        markdown_path,
        entry_id=record.entry_id,
        scope=MemoryScope.SESSION,
        created_at=record.created_at,
        keywords=("pytest", "./tokentrim/README.md"),
        metadata={"kind": "checkpoint"},
        content="Updated note: fix pytest failure in ./tokentrim/README.md.",
    )

    entries = store.retrieve(
        user_id="user",
        session_id="session",
        query="fix ./tokentrim/README.md pytest",
    )

    assert entries
    assert entries[0].content == "Updated note: fix pytest failure in ./tokentrim/README.md."


def test_invalid_markdown_frontmatter_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "entries" / "broken.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-frontmatter\nbody\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_memory_markdown(path)


def test_remember_durable_memory_explicit_user_instruction_writes_memory(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    step = RememberDurableMemory(memory_store=store)
    messages = [
        {"role": "assistant", "content": "pytest fails on ./tokentrim/README.md due to a missing fixture."},
        {"role": "user", "content": "Remember this: pytest fails on ./tokentrim/README.md due to a missing fixture."},
    ]

    result = step.run(
        PipelineState(context=messages, tools=[]),
        _request(user_id="user", session_id="session"),
    )

    assert result.context == messages
    entries = store.retrieve(
        user_id="user",
        session_id="session",
        query="pytest ./tokentrim/README.md missing fixture",
    )
    assert entries
    assert entries[0].metadata["kind"] == "explicit"


def test_remember_durable_memory_writes_checkpoint_from_working_state(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    step = RememberDurableMemory(memory_store=store, policy=DefaultMemoryWritePolicy(include_active_errors=False))
    messages = [
        {
            "role": "system",
            "content": (
                "Working state only.\n"
                "Goal: Update README docs.\n"
                "Active Files: ./tokentrim/README.md\n"
                "Latest Command: pytest tests/tokentrim/test_client.py\n"
                "Next Step: Rerun pytest after the README edit."
            ),
        },
        {"role": "user", "content": "continue"},
    ]

    step.run(
        PipelineState(context=messages, tools=[]),
        _request(user_id="user", session_id="session", task_hint="docs update"),
    )

    entries = store.retrieve(user_id="user", session_id="session", query="README pytest docs update")
    assert entries
    assert entries[0].metadata["kind"] == "checkpoint"
    assert "Goal: Update README docs." in entries[0].content


def test_remember_durable_memory_prefers_active_error_memory(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    step = RememberDurableMemory(memory_store=store)
    messages = [
        {
            "role": "system",
            "content": (
                "Working state only.\n"
                "Goal: Fix failing test.\n"
                "Active Files: ./tokentrim/README.md\n"
                "Latest Command: pytest tests/tokentrim/test_client.py\n"
                "Active Error: FileNotFoundError: missing fixture\n"
                "Next Step: Update the fixture and rerun pytest."
            ),
        },
        {"role": "user", "content": "continue"},
    ]

    step.run(
        PipelineState(context=messages, tools=[]),
        _request(user_id="user", session_id="session", task_hint="fix failing pytest"),
    )

    entries = store.retrieve(user_id="user", session_id="session", query="missing fixture pytest")
    assert entries
    assert entries[0].metadata["kind"] == "active_error"
    assert "Active error: FileNotFoundError: missing fixture" in entries[0].content


def test_remember_durable_memory_is_noop_without_identifiers(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    step = RememberDurableMemory(memory_store=store)
    messages = [{"role": "user", "content": "Remember this: pytest fails on README."}]

    result = step.run(
        PipelineState(context=messages, tools=[]),
        _request(user_id=None, session_id="session"),
    )

    assert result.context == messages
    assert store.retrieve(user_id="user", session_id="session", query="pytest") == ()


def test_remember_durable_memory_dedupes_repeated_identical_entry(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    step = RememberDurableMemory(memory_store=store)
    messages = [{"role": "user", "content": "Remember this: pytest fails on ./tokentrim/README.md."}]
    request = _request(user_id="user", session_id="session")

    step.run(PipelineState(context=messages, tools=[]), request)
    step.run(PipelineState(context=messages, tools=[]), request)

    records = load_index(store.session_index_path(user_id="user", session_id="session"))
    assert len(records) == 1


def test_remember_durable_memory_skips_ordinary_chatter(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    step = RememberDurableMemory(memory_store=store)
    messages = [
        {"role": "assistant", "content": "I can help with that."},
        {"role": "user", "content": "continue"},
    ]

    step.run(
        PipelineState(context=messages, tools=[]),
        _request(user_id="user", session_id="session"),
    )

    records = load_index(store.session_index_path(user_id="user", session_id="session"))
    assert records == ()


def test_remember_durable_memory_can_extract_repeated_failure_from_trace_history(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    trace_store = InMemoryTraceStore()
    trace_store.create_trace(
        user_id="user",
        session_id="session",
        trace=_trace("openai_agents:trace_1", ended_at="2026-04-03T10:00:01+00:00", spans=()),
    )
    trace_store.append_span(
        trace_id="openai_agents:trace_1",
        span=_error_span(
            "openai_agents:span_1",
            "openai_agents:trace_1",
            message="FileNotFoundError: missing fixture",
        ),
    )
    trace_store.complete_trace(trace_id="openai_agents:trace_1")
    trace_store.create_trace(
        user_id="user",
        session_id="session",
        trace=_trace("openai_agents:trace_2", ended_at="2026-04-03T10:00:03+00:00", spans=()),
    )
    trace_store.append_span(
        trace_id="openai_agents:trace_2",
        span=_error_span(
            "openai_agents:span_2",
            "openai_agents:trace_2",
            message="FileNotFoundError: missing fixture",
        ),
    )
    trace_store.complete_trace(trace_id="openai_agents:trace_2")

    step = RememberDurableMemory(memory_store=store)
    step.run(
        PipelineState(context=[{"role": "user", "content": "continue"}], tools=[]),
        _request_with_trace_store(
            user_id="user",
            session_id="session",
            trace_store=trace_store,
            task_hint="fix README pytest",
        ),
    )

    entries = store.retrieve(user_id="user", session_id="session", query="missing fixture")
    assert entries
    assert entries[0].metadata["kind"] == "trace_repeated_error"


def test_remember_durable_memory_prefers_current_active_error_over_trace_extraction(tmp_path: Path) -> None:
    store = LocalDirectoryMemoryStore(root_dir=tmp_path)
    trace_store = InMemoryTraceStore()
    trace_store.create_trace(
        user_id="user",
        session_id="session",
        trace=_trace("openai_agents:trace_1", ended_at="2026-04-03T10:00:01+00:00", spans=()),
    )
    trace_store.append_span(
        trace_id="openai_agents:trace_1",
        span=_error_span(
            "openai_agents:span_1",
            "openai_agents:trace_1",
            message="Old repeated failure",
        ),
    )
    trace_store.complete_trace(trace_id="openai_agents:trace_1")
    trace_store.create_trace(
        user_id="user",
        session_id="session",
        trace=_trace("openai_agents:trace_2", ended_at="2026-04-03T10:00:03+00:00", spans=()),
    )
    trace_store.append_span(
        trace_id="openai_agents:trace_2",
        span=_error_span(
            "openai_agents:span_2",
            "openai_agents:trace_2",
            message="Old repeated failure",
        ),
    )
    trace_store.complete_trace(trace_id="openai_agents:trace_2")

    step = RememberDurableMemory(memory_store=store)
    messages = [
        {
            "role": "system",
            "content": (
                "Working state only.\n"
                "Goal: Fix current bug.\n"
                "Active Files: ./tokentrim/README.md\n"
                "Latest Command: pytest tests/tokentrim/test_client.py\n"
                "Active Error: CurrentError: broken README path\n"
                "Next Step: Update README path and rerun."
            ),
        },
        {"role": "user", "content": "continue"},
    ]

    step.run(
        PipelineState(context=messages, tools=[]),
        _request_with_trace_store(
            user_id="user",
            session_id="session",
            trace_store=trace_store,
            task_hint="fix current bug",
        ),
    )

    entries = store.retrieve(user_id="user", session_id="session", query="current bug README")
    assert entries
    assert entries[0].metadata["kind"] == "active_error"
    assert "CurrentError: broken README path" in entries[0].content
