from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from tokentrim.memory.policy import build_session_memory_policy
from tokentrim.memory.records import MemoryRecord
from tokentrim.memory.store import FilesystemMemoryStore, MemoryStore
from tokentrim.memory.writer import write_session_memory
from tokentrim.types.tool import Tool

SESSION_MEMORY_READ_TOOL_NAME = "read_session_memory"
SESSION_MEMORY_WRITE_TOOL_NAME = "write_session_memory"
SESSION_MEMORY_WRITE_TOOL_DESCRIPTION = (
    "Write or update one file-backed session memory. Use this for useful current-session facts, "
    "constraints, active state, decisions, or preferences that should persist across turns in the current session."
)
SESSION_MEMORY_READ_TOOL_DESCRIPTION = (
    "Read the session memory index or one specific session memory file for the current session."
)

SESSION_MEMORY_KINDS = (
    "constraint",
    "active_state",
    "task_fact",
    "decision",
    "preference",
)


def build_agent_aware_memory_prompt(
    *,
    memory_store: MemoryStore,
    session_id: str,
    write_tool_name: str = SESSION_MEMORY_WRITE_TOOL_NAME,
    read_tool_name: str = SESSION_MEMORY_READ_TOOL_NAME,
) -> str:
    """Describe Tokentrim's file-backed session-memory subsystem to the agent."""
    session_dir = _session_memory_dir(memory_store=memory_store, session_id=session_id)
    entrypoint_path = session_dir / "MEMORY.md"
    return build_session_memory_policy(
        session_dir=session_dir,
        entrypoint_path=entrypoint_path,
        write_tool_name=write_tool_name,
        read_tool_name=read_tool_name,
    )


def build_session_memory_tools() -> tuple[Tool, Tool]:
    return (
        build_session_memory_read_tool(),
        build_session_memory_write_tool(),
    )


def build_session_memory_read_tool(*, tool_name: str = SESSION_MEMORY_READ_TOOL_NAME) -> Tool:
    return {
        "name": tool_name,
        "description": SESSION_MEMORY_READ_TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "Optional specific memory id to read. If omitted, returns the MEMORY.md index.",
                    "maxLength": 200,
                }
            },
        },
    }


def build_session_memory_write_tool(*, tool_name: str = SESSION_MEMORY_WRITE_TOOL_NAME) -> Tool:
    return {
        "name": tool_name,
        "description": SESSION_MEMORY_WRITE_TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short semantic title for the memory file.",
                    "maxLength": 120,
                },
                "description": {
                    "type": "string",
                    "description": "One-line hook that should appear in MEMORY.md.",
                    "maxLength": 200,
                },
                "content": {
                    "type": "string",
                    "description": "Concrete session memory content to save.",
                    "maxLength": 1200,
                },
                "kind": {
                    "type": "string",
                    "enum": list(SESSION_MEMORY_KINDS),
                    "description": "The type of session memory being saved.",
                },
                "salience": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Relative importance of the memory from 0 to 1.",
                },
                "dedupe_key": {
                    "type": "string",
                    "description": "Stable key for updating an existing session memory instead of creating a duplicate.",
                    "maxLength": 200,
                },
                "reason": {
                    "type": "string",
                    "description": "Short explanation for why this memory matters in the current session.",
                    "maxLength": 200,
                },
            },
            "required": ["title", "description", "content", "kind"],
        },
    }


def execute_session_memory_write_tool(
    *,
    arguments: Mapping[str, Any],
    memory_store: MemoryStore,
    session_id: str,
) -> MemoryRecord:
    title = _require_non_empty_string(arguments.get("title"), field_name="write_session_memory.title")
    description = _require_non_empty_string(
        arguments.get("description"),
        field_name="write_session_memory.description",
    )
    content = _require_non_empty_string(arguments.get("content"), field_name="write_session_memory.content")
    kind = _require_kind(arguments.get("kind"))
    salience = _coerce_salience(arguments.get("salience", 0.5))
    dedupe_key = _optional_stripped_string(
        arguments.get("dedupe_key"),
        field_name="write_session_memory.dedupe_key",
    )
    metadata = _build_memory_metadata(
        title=title,
        description=description,
        reason=arguments.get("reason"),
        dedupe_key=dedupe_key,
    )
    return write_session_memory(
        memory_store=memory_store,
        session_id=session_id,
        content=content,
        kind=kind,
        salience=salience,
        dedupe_key=dedupe_key,
        metadata=metadata,
    )


def execute_session_memory_read_tool(
    *,
    arguments: Mapping[str, Any],
    memory_store: MemoryStore,
    session_id: str,
) -> dict[str, object]:
    requested_memory_id = _optional_stripped_string(
        arguments.get("memory_id"),
        field_name="read_session_memory.memory_id",
    )
    if requested_memory_id is None:
        return build_session_memory_index_result(memory_store=memory_store, session_id=session_id)

    record = next(
        (
            memory
            for memory in memory_store.list_memories(scope="session", subject_id=session_id)
            if memory.memory_id == requested_memory_id
        ),
        None,
    )
    if record is None:
        raise ValueError(f"Unknown session memory id: {requested_memory_id!r}.")
    return build_session_memory_record_result(record)


@dataclass(frozen=True, slots=True)
class SessionMemoryToolHandler:
    """Integration-facing helper for dispatching Tokentrim's standard session memory tools."""

    memory_store: MemoryStore
    session_id: str
    read_tool_name: str = SESSION_MEMORY_READ_TOOL_NAME
    write_tool_name: str = SESSION_MEMORY_WRITE_TOOL_NAME

    def can_handle(self, tool_name: str) -> bool:
        return tool_name in {self.read_tool_name, self.write_tool_name}

    def execute(self, *, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, object]:
        if tool_name == self.write_tool_name:
            record = execute_session_memory_write_tool(
                arguments=arguments,
                memory_store=self.memory_store,
                session_id=self.session_id,
            )
            return build_session_memory_write_result(record)
        if tool_name == self.read_tool_name:
            return execute_session_memory_read_tool(
                arguments=arguments,
                memory_store=self.memory_store,
                session_id=self.session_id,
            )
        raise ValueError(f"Unsupported tool name: {tool_name!r}.")


def build_session_memory_write_result(record: MemoryRecord) -> dict[str, object]:
    return {
        "ok": True,
        "action": "write_session_memory",
        "memory_id": record.memory_id,
        "kind": record.kind,
        "content": record.content,
        "title": _memory_title(record),
        "description": _memory_description(record),
    }


def build_session_memory_index_result(*, memory_store: MemoryStore, session_id: str) -> dict[str, object]:
    records = memory_store.list_memories(scope="session", subject_id=session_id)
    lines = ["# MEMORY", ""]
    if not records:
        lines.append("_No active memories yet._")
    else:
        for record in records:
            lines.append(f"- [{_memory_title(record)}] ({record.memory_id}) — {_memory_description(record)}")
    return {
        "ok": True,
        "action": "read_session_memory",
        "mode": "index",
        "content": "\n".join(lines),
        "memory_ids": [record.memory_id for record in records],
    }


def build_session_memory_record_result(record: MemoryRecord) -> dict[str, object]:
    return {
        "ok": True,
        "action": "read_session_memory",
        "mode": "record",
        "memory_id": record.memory_id,
        "title": _memory_title(record),
        "description": _memory_description(record),
        "kind": record.kind,
        "content": record.content,
        "source_refs": list(record.source_refs),
    }


def _session_memory_dir(*, memory_store: MemoryStore, session_id: str) -> Path:
    if isinstance(memory_store, FilesystemMemoryStore):
        return memory_store.scope_dir(scope="session", subject_id=session_id)
    return Path(f".tokentrim/memory/session/{session_id}")


def _memory_title(record: MemoryRecord) -> str:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    value = metadata.get("title")
    return value.strip() if isinstance(value, str) and value.strip() else record.kind.replace("_", " ").title()


def _memory_description(record: MemoryRecord) -> str:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    value = metadata.get("description")
    if isinstance(value, str) and value.strip():
        return value.strip()
    content = " ".join(record.content.split())
    return content[:117] + "..." if len(content) > 120 else content


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_stripped_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name=field_name)


def _require_kind(value: object) -> str:
    kind = _require_non_empty_string(value, field_name="write_session_memory.kind")
    if kind not in SESSION_MEMORY_KINDS:
        raise ValueError(f"write_session_memory.kind must be one of {SESSION_MEMORY_KINDS!r}.")
    return kind


def _coerce_salience(value: object) -> float:
    if not isinstance(value, int | float):
        raise ValueError("write_session_memory.salience must be numeric when provided.")
    return float(value)


def _build_memory_metadata(
    *,
    title: str,
    description: str,
    reason: object,
    dedupe_key: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "title": title,
        "description": description,
        "file_name": title,
    }
    normalized_reason = _optional_stripped_string(reason, field_name="write_session_memory.reason")
    if normalized_reason is not None:
        metadata["reason"] = normalized_reason
    if dedupe_key is not None:
        metadata["canonical_key"] = dedupe_key
    return cast(dict[str, object], metadata)
