from __future__ import annotations

import pytest

from tokentrim.memory import (
    InMemoryMemoryStore,
    SESSION_MEMORY_TOOL_NAME,
    SessionMemoryToolHandler,
    build_agent_aware_memory_prompt,
    build_session_memory_tool_result,
    build_session_memory_tool,
    execute_session_memory_tool,
)


def test_build_session_memory_tool_exposes_standard_contract() -> None:
    tool = build_session_memory_tool()

    assert tool["name"] == SESSION_MEMORY_TOOL_NAME
    assert tool["input_schema"]["required"] == ["content", "kind"]
    assert "Session memory may be injected later" in tool["description"]


def test_build_agent_aware_memory_prompt_mentions_system_owned_injection() -> None:
    prompt = build_agent_aware_memory_prompt()

    assert "Session memory is available" in prompt
    assert "Memory injection is controlled by the system" in prompt


def test_execute_session_memory_tool_writes_session_memory() -> None:
    store = InMemoryMemoryStore()

    record = execute_session_memory_tool(
        arguments={
            "content": "Avoid destructive commands unless explicitly requested",
            "kind": "constraint",
            "dedupe_key": "avoid_destructive",
            "reason": "User stated this as a constraint",
        },
        memory_store=store,
        session_id="sess_1",
    )

    assert record.scope == "session"
    assert record.kind == "constraint"
    assert record.metadata == {"reason": "User stated this as a constraint"}


def test_execute_session_memory_tool_rejects_invalid_kind() -> None:
    store = InMemoryMemoryStore()

    with pytest.raises(ValueError):
        execute_session_memory_tool(
            arguments={"content": "Remember this", "kind": "summary"},
            memory_store=store,
            session_id="sess_1",
        )


def test_session_memory_tool_handler_dispatches_and_returns_standard_payload() -> None:
    store = InMemoryMemoryStore()
    handler = SessionMemoryToolHandler(memory_store=store, session_id="sess_1")

    result = handler.execute(
        tool_name="remember",
        arguments={
            "content": "Use repo root for command debugging",
            "kind": "active_state",
            "dedupe_key": "repo_root",
        },
    )

    assert result["ok"] is True
    assert result["scope"] == "session"
    assert result["kind"] == "active_state"
    assert result["content"] == "Use repo root for command debugging"


def test_session_memory_tool_handler_rejects_unknown_tool_name() -> None:
    store = InMemoryMemoryStore()
    handler = SessionMemoryToolHandler(memory_store=store, session_id="sess_1")

    with pytest.raises(ValueError):
        handler.execute(
            tool_name="other_tool",
            arguments={"content": "x", "kind": "constraint"},
        )


def test_build_session_memory_tool_result_returns_serializable_payload() -> None:
    store = InMemoryMemoryStore()
    record = execute_session_memory_tool(
        arguments={"content": "Prefer concise answers", "kind": "preference"},
        memory_store=store,
        session_id="sess_1",
    )

    payload = build_session_memory_tool_result(record)

    assert payload["ok"] is True
    assert payload["memory_id"] == record.memory_id
    assert payload["dedupe_key"] is None
