from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tokentrim.integrations.openai_agents.mappers import (
    trim_handoff_input_history,
    trim_input_items,
)
from tokentrim.integrations.openai_agents.sdk import load_agents_sdk, resolve_maybe_awaitable

if TYPE_CHECKING:
    from agents.handoffs import HandoffInputData
    from agents.items import TResponseInputItem
    from agents.run import CallModelData, ModelInputData

    from tokentrim.client import Tokentrim
    from tokentrim.integrations.openai_agents.options import OpenAIAgentsOptions


def build_call_model_input_filter(
    tokentrim: Tokentrim,
    *,
    existing_filter: Any,
    options: OpenAIAgentsOptions,
):
    _, ModelInputData, _ = load_agents_sdk()

    async def _filter(call_model_data: CallModelData[Any]) -> ModelInputData:
        model_data = call_model_data.model_data
        if existing_filter is not None:
            model_data = await resolve_maybe_awaitable(existing_filter(call_model_data))

        trimmed_input = trim_input_items(
            tokentrim=tokentrim,
            items=model_data.input,
            options=options,
        )
        return ModelInputData(
            input=trimmed_input,
            instructions=model_data.instructions,
        )

    return _filter


def build_session_input_callback(
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
            combined = await resolve_maybe_awaitable(existing_callback(history_items, new_items))

        return trim_input_items(
            tokentrim=tokentrim,
            items=combined,
            options=options,
        )

    return _callback


def build_handoff_input_filter(
    tokentrim: Tokentrim,
    *,
    existing_filter: Any,
    options: OpenAIAgentsOptions,
):
    _, _, HandoffInputData = load_agents_sdk()

    async def _filter(handoff_input_data: HandoffInputData) -> HandoffInputData:
        filtered = handoff_input_data
        if existing_filter is not None:
            filtered = await resolve_maybe_awaitable(existing_filter(handoff_input_data))

        trimmed_history = trim_handoff_input_history(
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
