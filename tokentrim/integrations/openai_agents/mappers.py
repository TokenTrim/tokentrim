from __future__ import annotations

from typing import TYPE_CHECKING, cast

from tokentrim.types.message import Message

if TYPE_CHECKING:
    from agents.items import TResponseInputItem

    from tokentrim.client import Tokentrim
    from tokentrim.integrations.openai_agents.options import OpenAIAgentsOptions


def trim_handoff_input_history(
    *,
    tokentrim: Tokentrim,
    input_history: str | tuple[TResponseInputItem, ...],
    options: OpenAIAgentsOptions,
) -> str | tuple[TResponseInputItem, ...]:
    if isinstance(input_history, str):
        trimmed = trim_input_items(
            tokentrim=tokentrim,
            items=[{"role": "user", "content": input_history}],
            options=options,
        )
        if (
            len(trimmed) == 1
            and trimmed[0].get("role") == "user"
            and isinstance(trimmed[0].get("content"), str)
        ):
            return cast(str, trimmed[0]["content"])
        return tuple(trimmed)

    trimmed = trim_input_items(
        tokentrim=tokentrim,
        items=list(input_history),
        options=options,
    )
    return tuple(trimmed)


def trim_input_items(
    *,
    tokentrim: Tokentrim,
    items: list[TResponseInputItem],
    options: OpenAIAgentsOptions,
) -> list[TResponseInputItem]:
    messages = input_items_to_messages(items)
    if messages is None:
        return list(items)

    result = tokentrim.compose(*options.steps).apply(
        messages,
        user_id=options.user_id,
        session_id=options.session_id,
        token_budget=options.token_budget,
        trace_store=options.trace_store,
        pipeline_tracer=options.pipeline_tracer,
    )
    return [message_to_input_item(message) for message in result.context]


def input_items_to_messages(items: list[TResponseInputItem]) -> list[Message] | None:
    messages: list[Message] = []
    for item in items:
        message = input_item_to_message(item)
        if message is None:
            return None
        messages.append(message)
    return messages


def input_item_to_message(item: TResponseInputItem) -> Message | None:
    if not isinstance(item, dict):
        return None

    role = item.get("role")
    if not isinstance(role, str):
        return None

    content = extract_text_content(item.get("content"))
    if content is None:
        return None

    return {
        "role": role,
        "content": content,
    }


def extract_text_content(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            return None

        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue

        refusal = block.get("refusal")
        if isinstance(refusal, str):
            parts.append(refusal)
            continue

        return None

    return "".join(parts)


def message_to_input_item(message: Message) -> TResponseInputItem:
    return {
        "role": message["role"],
        "content": message["content"],
    }
