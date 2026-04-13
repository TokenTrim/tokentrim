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
                "name": "remember",
                "arguments": json.dumps(
                    {
                        "content": "Use repository root",
                        "kind": "active_state",
                    }
                ),
            },
        }
    )

    assert parsed["tool_call_id"] == "call_1"
    assert parsed["name"] == "remember"
    assert parsed["arguments"]["kind"] == "active_state"


def test_build_openai_agents_tool_result_message_uses_tool_role_shape() -> None:
    message = build_openai_agents_tool_result_message(
        tool_call_id="call_1",
        tool_name="remember",
        payload={"ok": True, "scope": "session"},
    )

    assert message == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "remember",
        "content": json.dumps({"ok": True, "scope": "session"}, sort_keys=True),
    }


def test_openai_agents_session_memory_bridge_handles_remember_tool_end_to_end() -> None:
    store = InMemoryMemoryStore()
    bridge = build_openai_agents_session_memory_bridge(
        memory_store=store,
        session_id="sess_1",
    )

    message = bridge.handle_tool_call(
        {
            "id": "call_1",
            "function": {
                "name": "remember",
                "arguments": json.dumps(
                    {
                        "content": "Avoid destructive commands unless explicitly requested",
                        "kind": "constraint",
                        "dedupe_key": "avoid_destructive",
                    }
                ),
            },
        }
    )

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_1"
    assert message["name"] == "remember"
    assert "avoid_destructive" in str(message["content"])
    stored = store.list_memories(scope="session", subject_id="sess_1")
    assert len(stored) == 1
    assert stored[0].content == "Avoid destructive commands unless explicitly requested"


def test_openai_agents_session_memory_bridge_can_handle_only_remember() -> None:
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
            "function": {"name": "remember", "arguments": "{}"},
        }
    )
    assert not bridge.can_handle(
        {
            "id": "call_2",
            "function": {"name": "other_tool", "arguments": "{}"},
        }
    )
