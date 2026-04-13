from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

from tokentrim.memory.records import MemoryRecord
from tokentrim.memory.store import MemoryStore
from tokentrim.memory.writer import write_session_memory
from tokentrim.types.tool import Tool

SESSION_MEMORY_TOOL_NAME = "remember"
SESSION_MEMORY_TOOL_DESCRIPTION = (
    "Write a short session-scoped memory when you identify a useful current-session fact, "
    "constraint, decision, or preference. Do not store turn-by-turn summaries, speculative "
    "thoughts, or durable memory. Session memory may be injected later by the system automatically."
)

SESSION_MEMORY_KINDS = (
    "constraint",
    "active_state",
    "task_fact",
    "decision",
    "preference",
)


def build_agent_aware_memory_prompt(*, tool_name: str = SESSION_MEMORY_TOOL_NAME) -> str:
    """Describe Tokentrim's session-memory capability to an agent."""
    return (
        "Session memory is available for this conversation.\n"
        f"If you identify a useful session-scoped fact, constraint, decision, or preference, you may "
        f"use the `{tool_name}` tool to save it.\n"
        "Use it sparingly. Do not save summaries of every turn, speculative thoughts, or durable memory.\n"
        "Memory injection is controlled by the system and happens automatically when relevant."
    )


def build_session_memory_tool(*, tool_name: str = SESSION_MEMORY_TOOL_NAME) -> Tool:
    """Build Tokentrim's standard remember tool contract."""
    return {
        "name": tool_name,
        "description": SESSION_MEMORY_TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Short session-scoped memory to save.",
                    "maxLength": 400,
                },
                "kind": {
                    "type": "string",
                    "enum": list(SESSION_MEMORY_KINDS),
                    "description": "The type of memory being saved.",
                },
                "salience": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Relative importance of the memory from 0 to 1.",
                },
                "dedupe_key": {
                    "type": "string",
                    "description": "Optional stable key for updating an existing session memory.",
                    "maxLength": 120,
                },
                "reason": {
                    "type": "string",
                    "description": "Short explanation of why this is worth remembering.",
                    "maxLength": 200,
                },
            },
            "required": ["content", "kind"],
        },
    }


def execute_session_memory_tool(
    *,
    arguments: Mapping[str, Any],
    memory_store: MemoryStore,
    session_id: str,
) -> MemoryRecord:
    """Validate and execute a standard remember tool call."""
    content = _require_non_empty_string(arguments.get("content"), field_name="remember.content")
    kind = _require_kind(arguments.get("kind"))
    salience = _coerce_salience(arguments.get("salience", 0.5))
    dedupe_key = _optional_stripped_string(arguments.get("dedupe_key"), field_name="remember.dedupe_key")
    metadata = _build_reason_metadata(arguments.get("reason"))

    return write_session_memory(
        memory_store=memory_store,
        session_id=session_id,
        content=content,
        kind=kind,
        salience=salience,
        dedupe_key=dedupe_key,
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class SessionMemoryToolHandler:
    """Integration-facing helper for dispatching Tokentrim's standard remember tool."""

    memory_store: MemoryStore
    session_id: str
    tool_name: str = SESSION_MEMORY_TOOL_NAME

    def can_handle(self, tool_name: str) -> bool:
        return tool_name == self.tool_name

    def execute(self, *, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, object]:
        if not self.can_handle(tool_name):
            raise ValueError(f"Unsupported tool name: {tool_name!r}.")

        record = execute_session_memory_tool(
            arguments=arguments,
            memory_store=self.memory_store,
            session_id=self.session_id,
        )
        return build_session_memory_tool_result(record)


def build_session_memory_tool_result(record: MemoryRecord) -> dict[str, object]:
    """Build a standard integration-facing result payload for remember tool calls."""

    return {
        "ok": True,
        "memory_id": record.memory_id,
        "scope": record.scope,
        "kind": record.kind,
        "content": record.content,
        "dedupe_key": record.dedupe_key,
    }


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_stripped_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name=field_name)


def _require_kind(value: object) -> str:
    kind = _require_non_empty_string(value, field_name="remember.kind")
    if kind not in SESSION_MEMORY_KINDS:
        raise ValueError(f"remember.kind must be one of {SESSION_MEMORY_KINDS!r}.")
    return kind


def _coerce_salience(value: object) -> float:
    if not isinstance(value, int | float):
        raise ValueError("remember.salience must be numeric when provided.")
    return float(value)


def _build_reason_metadata(value: object) -> dict[str, object] | None:
    reason = _optional_stripped_string(value, field_name="remember.reason")
    if reason is None:
        return None
    return cast(dict[str, object], {"reason": reason})
