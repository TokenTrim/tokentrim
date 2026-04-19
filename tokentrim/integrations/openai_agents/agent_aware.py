from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from tokentrim.memory import SessionMemoryToolHandler
from tokentrim.memory.agent_aware import (
    SESSION_MEMORY_READ_TOOL_NAME,
    SESSION_MEMORY_WRITE_TOOL_NAME,
)
from tokentrim.types.message import Message


@dataclass(frozen=True, slots=True)
class OpenAIAgentsSessionMemoryBridge:
    """Adapter helper that turns Tokentrim session-memory tool calls into OpenAI-Agents-style tool results."""

    handler: SessionMemoryToolHandler

    def can_handle(self, tool_call: Mapping[str, Any]) -> bool:
        parsed = parse_openai_agents_tool_call(tool_call)
        return self.handler.can_handle(parsed["name"])

    def handle_tool_call(self, tool_call: Mapping[str, Any]) -> Message:
        parsed = parse_openai_agents_tool_call(tool_call)
        result = self.handler.execute(
            tool_name=parsed["name"],
            arguments=parsed["arguments"],
        )
        return build_openai_agents_tool_result_message(
            tool_call_id=parsed["tool_call_id"],
            tool_name=parsed["name"],
            payload=result,
        )


def parse_openai_agents_tool_call(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    tool_call_id = tool_call.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id.strip():
        raise ValueError("OpenAI Agents tool call must include a non-empty string id.")

    name: str | None = None
    raw_arguments: Any = None

    if isinstance(tool_call.get("name"), str):
        name = tool_call["name"]
        raw_arguments = tool_call.get("arguments")
    else:
        function = tool_call.get("function")
        if isinstance(function, Mapping):
            if isinstance(function.get("name"), str):
                name = function["name"]
            raw_arguments = function.get("arguments")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("OpenAI Agents tool call must include a tool name.")

    arguments = _normalize_arguments(raw_arguments)
    return {
        "tool_call_id": tool_call_id.strip(),
        "name": name.strip(),
        "arguments": arguments,
    }


def build_openai_agents_tool_result_message(
    *,
    tool_call_id: str,
    tool_name: str,
    payload: Mapping[str, Any],
) -> Message:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(dict(payload), sort_keys=True),
    }


def build_openai_agents_session_memory_bridge(
    *,
    memory_store: Any,
    session_id: str,
) -> OpenAIAgentsSessionMemoryBridge:
    return OpenAIAgentsSessionMemoryBridge(
        handler=SessionMemoryToolHandler(
            memory_store=memory_store,
            session_id=session_id,
            read_tool_name=SESSION_MEMORY_READ_TOOL_NAME,
            write_tool_name=SESSION_MEMORY_WRITE_TOOL_NAME,
        )
    )


def _normalize_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, Mapping):
        return dict(raw_arguments)
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI Agents tool call arguments must be valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI Agents tool call arguments must decode to an object.")
        return parsed
    raise ValueError("OpenAI Agents tool call arguments must be an object or JSON object string.")
