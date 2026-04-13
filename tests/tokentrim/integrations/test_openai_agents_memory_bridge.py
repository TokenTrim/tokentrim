from __future__ import annotations

import json

from tokentrim.integrations.openai_agents import (
    OpenAIAgentsSessionMemoryBridge,
    build_openai_agents_session_memory_bridge,
)
from tokentrim.integrations.openai_agents.agent_aware import (
    build_openai_agents_tool_result_message,
    parse_openai_agents_tool_call,
)
from tokentrim.memory import InMemoryMemoryStore


def test_parse_openai_agents_tool_call_supports_function_shape() -> None:
    parsed = parse_openai_agents_tool_call(
        {
            "id": "call_1",
            "function": {
                "name": "write_session_memory",
                "arguments": json.dumps(
                    {
                        "title": "Repo Root",
                        "description": "Use repository root for debugging commands",
                        "content": "Run debugging commands from the repository root.",
                        "kind": "active_state",
                    }
                ),
            },
        }
    )

    assert parsed["tool_call_id"] == "call_1"
    assert parsed["name"] == "write_session_memory"
    assert parsed["arguments"]["kind"] == "active_state"


def test_build_openai_agents_tool_result_message_uses_tool_role_shape() -> None:
    message = build_openai_agents_tool_result_message(
        tool_call_id="call_1",
        tool_name="write_session_memory",
        payload={"ok": True, "action": "write_session_memory"},
    )

    assert message == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "write_session_memory",
        "content": json.dumps({"action": "write_session_memory", "ok": True}, sort_keys=True),
    }


def test_openai_agents_session_memory_bridge_handles_write_and_read_end_to_end() -> None:
    store = InMemoryMemoryStore()
    bridge = build_openai_agents_session_memory_bridge(
        memory_store=store,
        session_id="sess_1",
    )

    write_message = bridge.handle_tool_call(
        {
            "id": "call_1",
            "function": {
                "name": "write_session_memory",
                "arguments": json.dumps(
                    {
                        "title": "Repo Root",
                        "description": "Use repository root for debugging commands",
                        "content": "Run debugging commands from the repository root.",
                        "kind": "active_state",
                        "dedupe_key": "repo_root",
                    }
                ),
            },
        }
    )
    read_message = bridge.handle_tool_call(
        {
            "id": "call_2",
            "function": {
                "name": "read_session_memory",
                "arguments": json.dumps({}),
            },
        }
    )

    assert write_message["name"] == "write_session_memory"
    assert read_message["name"] == "read_session_memory"
    stored = store.list_memories(scope="session", subject_id="sess_1")
    assert len(stored) == 1
    assert stored[0].content == "Run debugging commands from the repository root."


def test_openai_agents_session_memory_bridge_handles_only_session_memory_tools() -> None:
    store = InMemoryMemoryStore()
    bridge = OpenAIAgentsSessionMemoryBridge(
        handler=build_openai_agents_session_memory_bridge(
            memory_store=store,
            session_id="sess_1",
        ).handler
    )

    assert bridge.can_handle(
        {
            "id": "call_1",
            "function": {"name": "read_session_memory", "arguments": "{}"},
        }
    )
    assert bridge.can_handle(
        {
            "id": "call_2",
            "function": {"name": "write_session_memory", "arguments": "{}"},
        }
    )
    assert not bridge.can_handle(
        {
            "id": "call_3",
            "function": {"name": "other_tool", "arguments": "{}"},
        }
    )
