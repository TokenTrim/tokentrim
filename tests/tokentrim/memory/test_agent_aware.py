from __future__ import annotations

import pytest

from tokentrim.memory import (
    FilesystemMemoryStore,
    InMemoryMemoryStore,
    SESSION_MEMORY_READ_TOOL_NAME,
    SESSION_MEMORY_WRITE_TOOL_NAME,
    SessionMemoryToolHandler,
    build_agent_aware_memory_prompt,
    build_session_memory_index_result,
    build_session_memory_read_tool,
    build_session_memory_record_result,
    build_session_memory_tools,
    build_session_memory_write_result,
    build_session_memory_write_tool,
    execute_session_memory_read_tool,
    execute_session_memory_write_tool,
)


def test_build_session_memory_tools_expose_read_and_write_contracts() -> None:
    read_tool, write_tool = build_session_memory_tools()

    assert read_tool["name"] == SESSION_MEMORY_READ_TOOL_NAME
    assert write_tool["name"] == SESSION_MEMORY_WRITE_TOOL_NAME
    assert write_tool["input_schema"]["required"] == ["title", "description", "content", "kind"]


def test_build_agent_aware_memory_prompt_mentions_real_directory(tmp_path) -> None:
    store = FilesystemMemoryStore(root_dir=tmp_path / "memory")
    prompt = build_agent_aware_memory_prompt(memory_store=store, session_id="sess_1")

    assert "file-backed session memory directory" in prompt
    assert "MEMORY.md" in prompt
    assert "write_session_memory" in prompt
    assert "read_session_memory" in prompt
    assert "When To Save" in prompt
    assert "What Not To Save" in prompt
    assert "Trust And Freshness" in prompt


def test_execute_session_memory_write_tool_writes_session_memory() -> None:
    store = InMemoryMemoryStore()

    record = execute_session_memory_write_tool(
        arguments={
            "title": "Repo Root",
            "description": "Use repository root for debugging commands",
            "content": "Run debugging commands from the repository root.",
            "kind": "active_state",
            "dedupe_key": "repo_root",
            "reason": "Current task depends on repo-root relative paths",
        },
        memory_store=store,
        session_id="sess_1",
    )

    assert record.scope == "session"
    assert record.kind == "active_state"
    assert record.metadata == {
        "title": "Repo Root",
        "description": "Use repository root for debugging commands",
        "file_name": "Repo Root",
        "reason": "Current task depends on repo-root relative paths",
        "canonical_key": "repo_root",
    }


def test_execute_session_memory_write_tool_rejects_invalid_kind() -> None:
    store = InMemoryMemoryStore()

    with pytest.raises(ValueError):
        execute_session_memory_write_tool(
            arguments={
                "title": "Bad",
                "description": "Bad",
                "content": "Remember this",
                "kind": "summary",
            },
            memory_store=store,
            session_id="sess_1",
        )


def test_execute_session_memory_read_tool_returns_index_and_record() -> None:
    store = InMemoryMemoryStore()
    record = execute_session_memory_write_tool(
        arguments={
            "title": "Repo Root",
            "description": "Use repository root for debugging commands",
            "content": "Run debugging commands from the repository root.",
            "kind": "active_state",
        },
        memory_store=store,
        session_id="sess_1",
    )

    index_result = execute_session_memory_read_tool(
        arguments={},
        memory_store=store,
        session_id="sess_1",
    )
    record_result = execute_session_memory_read_tool(
        arguments={"memory_id": record.memory_id},
        memory_store=store,
        session_id="sess_1",
    )

    assert index_result["mode"] == "index"
    assert "Repo Root" in str(index_result["content"])
    assert record_result["mode"] == "record"
    assert record_result["memory_id"] == record.memory_id


def test_session_memory_tool_handler_dispatches_read_and_write() -> None:
    store = InMemoryMemoryStore()
    handler = SessionMemoryToolHandler(memory_store=store, session_id="sess_1")

    write_result = handler.execute(
        tool_name="write_session_memory",
        arguments={
            "title": "Repo Root",
            "description": "Use repository root for debugging commands",
            "content": "Run debugging commands from the repository root.",
            "kind": "active_state",
            "dedupe_key": "repo_root",
        },
    )
    read_result = handler.execute(
        tool_name="read_session_memory",
        arguments={},
    )

    assert write_result["ok"] is True
    assert write_result["action"] == "write_session_memory"
    assert read_result["ok"] is True
    assert read_result["action"] == "read_session_memory"


def test_session_memory_tool_handler_rejects_unknown_tool_name() -> None:
    store = InMemoryMemoryStore()
    handler = SessionMemoryToolHandler(memory_store=store, session_id="sess_1")

    with pytest.raises(ValueError):
        handler.execute(tool_name="other_tool", arguments={})


def test_session_memory_result_builders_are_serializable() -> None:
    store = InMemoryMemoryStore()
    record = execute_session_memory_write_tool(
        arguments={
            "title": "Repo Root",
            "description": "Use repository root for debugging commands",
            "content": "Run debugging commands from the repository root.",
            "kind": "active_state",
        },
        memory_store=store,
        session_id="sess_1",
    )

    write_payload = build_session_memory_write_result(record)
    index_payload = build_session_memory_index_result(memory_store=store, session_id="sess_1")
    record_payload = build_session_memory_record_result(record)

    assert write_payload["memory_id"] == record.memory_id
    assert index_payload["mode"] == "index"
    assert record_payload["mode"] == "record"
