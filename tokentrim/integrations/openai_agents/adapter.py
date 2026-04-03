from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from tokentrim.errors.base import TokentrimError
from tokentrim.integrations.base import IntegrationAdapter
from tokentrim.integrations.openai_agents.hooks import (
    build_call_model_input_filter,
    build_handoff_input_filter,
    build_session_input_callback,
)
from tokentrim.integrations.openai_agents.options import OpenAIAgentsOptions
from tokentrim.integrations.openai_agents.pipeline_tracing import OpenAIAgentsPipelineTracer
from tokentrim.integrations.openai_agents.sdk import (
    load_agents_sdk,
    requires_adapter,
    requires_trim_hooks,
)
from tokentrim.integrations.openai_agents.tracing import (
    build_identity_trace_metadata,
    install_identity_processor,
)

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
            trace_store=self._options.trace_store,
        ):
            return effective_run_config

        should_trim = requires_trim_hooks(
            token_budget=self._options.token_budget,
            steps=self._options.steps,
        )
        effective_options = (
            replace(
                self._options,
                pipeline_tracer=self._options.pipeline_tracer or OpenAIAgentsPipelineTracer(),
            )
            if should_trim
            else self._options
        )
        trace_metadata = effective_run_config.trace_metadata
        if effective_options.trace_store is not None:
            if not effective_options.user_id or not effective_options.session_id:
                raise TokentrimError(
                    "`trace_store` requires both `user_id` and `session_id` in OpenAI Agents integration."
                )
            store_id = install_identity_processor(effective_options.trace_store)
            trace_metadata = build_identity_trace_metadata(
                effective_run_config.trace_metadata,
                store_id=store_id,
                user_id=effective_options.user_id,
                session_id=effective_options.session_id,
            )

        return replace(
            effective_run_config,
            call_model_input_filter=(
                build_call_model_input_filter(
                    tokentrim,
                    existing_filter=effective_run_config.call_model_input_filter,
                    options=effective_options,
                )
                if should_trim
                else effective_run_config.call_model_input_filter
            ),
            session_input_callback=(
                build_session_input_callback(
                    tokentrim,
                    existing_callback=effective_run_config.session_input_callback,
                    options=effective_options,
                )
                if should_trim and effective_options.apply_to_session_history
                else effective_run_config.session_input_callback
            ),
            handoff_input_filter=(
                build_handoff_input_filter(
                    tokentrim,
                    existing_filter=effective_run_config.handoff_input_filter,
                    options=effective_options,
                )
                if should_trim and effective_options.apply_to_handoffs
                else effective_run_config.handoff_input_filter
            ),
            trace_metadata=trace_metadata,
        )
