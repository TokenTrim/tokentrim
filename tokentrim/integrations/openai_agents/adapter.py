from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from tokentrim.integrations.base import IntegrationAdapter
from tokentrim.integrations.openai_agents.hooks import (
    build_call_model_input_filter,
    build_handoff_input_filter,
    build_session_input_callback,
)
from tokentrim.integrations.openai_agents.options import OpenAIAgentsOptions
from tokentrim.integrations.openai_agents.sdk import load_agents_sdk, requires_adapter

if TYPE_CHECKING:
    from agents import RunConfig

    from tokentrim.client import Tokentrim


class OpenAIAgentsAdapter(IntegrationAdapter["RunConfig"]):
    """Tokentrim adapter for the OpenAI Agents SDK."""

    def __init__(self, options: OpenAIAgentsOptions | None = None) -> None:
        self._options = options or OpenAIAgentsOptions()

    def wrap(self, tokentrim: Tokentrim, config: RunConfig | None = None) -> RunConfig:
        RunConfig, _, _ = load_agents_sdk()
        effective_run_config = config or RunConfig()
        if not requires_adapter(
            token_budget=self._options.token_budget,
            steps=self._options.steps,
        ):
            return effective_run_config

        return replace(
            effective_run_config,
            call_model_input_filter=build_call_model_input_filter(
                tokentrim,
                existing_filter=effective_run_config.call_model_input_filter,
                options=self._options,
            ),
            session_input_callback=(
                build_session_input_callback(
                    tokentrim,
                    existing_callback=effective_run_config.session_input_callback,
                    options=self._options,
                )
                if self._options.apply_to_session_history
                else effective_run_config.session_input_callback
            ),
            handoff_input_filter=(
                build_handoff_input_filter(
                    tokentrim,
                    existing_filter=effective_run_config.handoff_input_filter,
                    options=self._options,
                )
                if self._options.apply_to_handoffs
                else effective_run_config.handoff_input_filter
            ),
        )
