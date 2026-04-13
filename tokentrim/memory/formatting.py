from __future__ import annotations

from tokentrim.core.token_counting import count_message_tokens
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
        lines.append(f"- [{candidate.scope}/{candidate.kind}] {candidate.content.strip()}")

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
