from __future__ import annotations

import re
from typing import Final

from tokentrim.types.message import Message
from tokentrim.working_state import find_working_state

_QUERY_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_./-]{3,}")


def build_memory_query(*, messages: list[Message], task_hint: str | None) -> str:
    parts: list[str] = []
    if task_hint:
        parts.append(task_hint)

    for message in reversed(messages):
        content = message["content"].strip()
        if not content:
            continue
        if message["role"] == "user":
            parts.append(content)
            break

    working_state = find_working_state(messages)
    if working_state is not None:
        parts.extend(
            value
            for value in (
                working_state.goal,
                working_state.current_step,
                " ".join(working_state.active_files) if working_state.active_files else None,
                working_state.latest_command,
                working_state.active_error,
                " ".join(working_state.constraints) if working_state.constraints else None,
                working_state.next_step,
            )
            if value
        )

    normalized = " ".join(parts).strip()
    if not normalized:
        return ""

    tokens: list[str] = []
    seen: set[str] = set()
    for match in _QUERY_TOKEN_RE.finditer(normalized):
        token = match.group(0).lower()
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return " ".join(tokens)


def render_memory_message(entries: tuple[object, ...]) -> str:
    lines = ["Durable memory only."]
    for index, entry in enumerate(entries, start=1):
        content = getattr(entry, "content")
        created_at = getattr(entry, "created_at")
        lines.append(f"Memory {index} ({created_at}): {content}")
    return "\n".join(lines)


def insert_after_leading_system_messages(
    messages: list[Message],
    memory_message: Message,
) -> list[Message]:
    index = 0
    while index < len(messages) and messages[index]["role"] == "system":
        index += 1
    return [*messages[:index], memory_message, *messages[index:]]
