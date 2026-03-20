from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from tokentrim.errors.base import TokentrimError
from tokentrim.types.message import Message

if TYPE_CHECKING:
    from agents import RunConfig
    from agents.handoffs import HandoffInputData
    from agents.items import TResponseInputItem
    from agents.run import CallModelData, ModelInputData

    from tokentrim.client import Tokentrim


@dataclass(frozen=True, slots=True)
class OpenAIAgentsOptions:
    """
    Configure how Tokentrim is applied to OpenAI Agents SDK runs.

    The current adapter only rewrites plain text message inputs. Rich Responses
    items such as tool calls, images, and file search results are preserved
    unchanged and bypass Tokentrim's context pipeline.
    """

    user_id: str | None = None
    session_id: str | None = None
    token_budget: int | None = None
    enable_compaction: bool = False
    enable_rlm: bool = False
    enable_filter: bool = False
    apply_to_session_history: bool = True
    apply_to_handoffs: bool = True


def wrap_run_config(
    tokentrim: Tokentrim,
    *,
    run_config: RunConfig | None = None,
    options: OpenAIAgentsOptions | None = None,
) -> RunConfig:
    """
    Wrap an Agents SDK RunConfig so Tokentrim trims plain-text message inputs.
    """

    RunConfig, _, _ = _load_agents_sdk()
    resolved_options = options or OpenAIAgentsOptions()
    effective_run_config = run_config or RunConfig()
    if not _requires_adapter(resolved_options):
        return effective_run_config

    return replace(
        effective_run_config,
        call_model_input_filter=_build_call_model_input_filter(
            tokentrim,
            existing_filter=effective_run_config.call_model_input_filter,
            options=resolved_options,
        ),
        session_input_callback=(
            _build_session_input_callback(
                tokentrim,
                existing_callback=effective_run_config.session_input_callback,
                options=resolved_options,
            )
            if resolved_options.apply_to_session_history
            else effective_run_config.session_input_callback
        ),
        handoff_input_filter=(
            _build_handoff_input_filter(
                tokentrim,
                existing_filter=effective_run_config.handoff_input_filter,
                options=resolved_options,
            )
            if resolved_options.apply_to_handoffs
            else effective_run_config.handoff_input_filter
        ),
    )


def _requires_adapter(options: OpenAIAgentsOptions) -> bool:
    return (
        options.token_budget is not None
        or options.enable_compaction
        or options.enable_rlm
        or options.enable_filter
    )


def _load_agents_sdk() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from agents import RunConfig
        from agents.handoffs import HandoffInputData
        from agents.run import ModelInputData
    except ImportError as exc:
        raise TokentrimError(
            "openai-agents is required for tokentrim.integrations.openai_agents."
        ) from exc

    return RunConfig, ModelInputData, HandoffInputData


async def _resolve_maybe_awaitable(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _build_call_model_input_filter(
    tokentrim: Tokentrim,
    *,
    existing_filter: Any,
    options: OpenAIAgentsOptions,
):
    _, ModelInputData, _ = _load_agents_sdk()

    async def _filter(call_model_data: CallModelData[Any]) -> ModelInputData:
        model_data = call_model_data.model_data
        if existing_filter is not None:
            model_data = await _resolve_maybe_awaitable(existing_filter(call_model_data))

        trimmed_input = _trim_input_items(
            tokentrim=tokentrim,
            items=model_data.input,
            options=options,
        )
        return ModelInputData(
            input=trimmed_input,
            instructions=model_data.instructions,
        )

    return _filter


def _build_session_input_callback(
    tokentrim: Tokentrim,
    *,
    existing_callback: Any,
    options: OpenAIAgentsOptions,
):
    async def _callback(
        history_items: list[TResponseInputItem],
        new_items: list[TResponseInputItem],
    ) -> list[TResponseInputItem]:
        if existing_callback is None:
            combined = [*history_items, *new_items]
        else:
            combined = await _resolve_maybe_awaitable(existing_callback(history_items, new_items))

        return _trim_input_items(
            tokentrim=tokentrim,
            items=combined,
            options=options,
        )

    return _callback


def _build_handoff_input_filter(
    tokentrim: Tokentrim,
    *,
    existing_filter: Any,
    options: OpenAIAgentsOptions,
):
    _, _, HandoffInputData = _load_agents_sdk()

    async def _filter(handoff_input_data: HandoffInputData) -> HandoffInputData:
        filtered = handoff_input_data
        if existing_filter is not None:
            filtered = await _resolve_maybe_awaitable(existing_filter(handoff_input_data))

        trimmed_history = _trim_handoff_input_history(
            tokentrim=tokentrim,
            input_history=filtered.input_history,
            options=options,
        )
        if trimmed_history == filtered.input_history:
            return filtered

        return HandoffInputData(
            input_history=trimmed_history,
            pre_handoff_items=filtered.pre_handoff_items,
            new_items=filtered.new_items,
            run_context=filtered.run_context,
        )

    return _filter


def _trim_handoff_input_history(
    *,
    tokentrim: Tokentrim,
    input_history: str | tuple[TResponseInputItem, ...],
    options: OpenAIAgentsOptions,
) -> str | tuple[TResponseInputItem, ...]:
    if isinstance(input_history, str):
        trimmed = _trim_input_items(
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

    trimmed = _trim_input_items(
        tokentrim=tokentrim,
        items=list(input_history),
        options=options,
    )
    return tuple(trimmed)


def _trim_input_items(
    *,
    tokentrim: Tokentrim,
    items: list[TResponseInputItem],
    options: OpenAIAgentsOptions,
) -> list[TResponseInputItem]:
    messages = _input_items_to_messages(items)
    if messages is None:
        return list(items)

    result = tokentrim.get_better_context(
        messages,
        user_id=options.user_id,
        session_id=options.session_id,
        token_budget=options.token_budget,
        enable_compaction=options.enable_compaction,
        enable_rlm=options.enable_rlm,
        enable_filter=options.enable_filter,
    )
    return [_message_to_input_item(message) for message in result.messages]


def _input_items_to_messages(items: list[TResponseInputItem]) -> list[Message] | None:
    messages: list[Message] = []
    for item in items:
        message = _input_item_to_message(item)
        if message is None:
            return None
        messages.append(message)
    return messages


def _input_item_to_message(item: TResponseInputItem) -> Message | None:
    if not isinstance(item, dict):
        return None

    role = item.get("role")
    if not isinstance(role, str):
        return None

    content = _extract_text_content(item.get("content"))
    if content is None:
        return None

    return {
        "role": role,
        "content": content,
    }


def _extract_text_content(content: object) -> str | None:
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


def _message_to_input_item(message: Message) -> TResponseInputItem:
    return {
        "role": message["role"],
        "content": message["content"],
    }


__all__ = [
    "OpenAIAgentsOptions",
    "wrap_run_config",
]
