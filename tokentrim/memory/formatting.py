from __future__ import annotations

from collections.abc import Mapping

from tokentrim.core.token_counting import count_message_tokens
from tokentrim.memory.freshness import memory_age, memory_freshness_bucket, memory_freshness_note
from tokentrim.memory.records import MemoryRecord
from tokentrim.types.message import Message

MEMORY_MESSAGE_PREFIX = "Injected memory:\n"


def render_injected_memory_message(
    *,
    candidates: tuple[MemoryRecord, ...],
    current_messages: list[Message],
    token_budget: int | None,
    max_memory_tokens: int,
    tokenizer_model: str | None,
) -> str | None:
    lines = []
    for candidate in candidates:
        if not candidate.content.strip():
            continue
        lines.append(_format_memory_line(candidate))

    if not lines:
        return None

    while lines:
        full_content = f"{MEMORY_MESSAGE_PREFIX}{chr(10).join(lines)}"
        candidate_message = [{"role": "system", "content": full_content}]
        candidate_tokens = count_message_tokens(candidate_message, tokenizer_model)
        if candidate_tokens > max_memory_tokens:
            lines.pop()
            continue
        if token_budget is not None:
            existing_tokens = count_message_tokens(current_messages, tokenizer_model)
            if existing_tokens + candidate_tokens > token_budget:
                lines.pop()
                continue
        return full_content
    return None


def _format_memory_line(record: MemoryRecord) -> str:
    prefix = f"- [{record.scope}/{record.kind}]"
    title = _memory_title(record)
    description = _memory_description(record)
    age = memory_age(record.updated_at)
    freshness_bucket = memory_freshness_bucket(record.updated_at)
    content = " ".join(record.content.split())
    parts = [f"{prefix} {title}"]
    if description:
        parts.append(f"{description}.")
    parts.append(f"Updated {age}.")
    if record.status != "active":
        parts.append(f"Status: {record.status}.")
    if freshness_bucket == "aging":
        parts.append("Re-check against current project state before using.")
    elif freshness_bucket == "stale":
        parts.append("Likely stale unless confirmed by current evidence.")
    parts.append(content)
    freshness_note = memory_freshness_note(record.updated_at)
    if freshness_note:
        parts.append(freshness_note)
    return " ".join(part for part in parts if part)


def _memory_title(record: MemoryRecord) -> str:
    metadata = record.metadata if isinstance(record.metadata, Mapping) else {}
    value = metadata.get("title")
    return value.strip() if isinstance(value, str) and value.strip() else record.kind.replace("_", " ").title()


def _memory_description(record: MemoryRecord) -> str:
    metadata = record.metadata if isinstance(record.metadata, Mapping) else {}
    value = metadata.get("description")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""
