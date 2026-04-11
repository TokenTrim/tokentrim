from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from tokentrim.core.copy_utils import freeze_messages, freeze_tools
from tokentrim.errors.base import TokentrimError
from tokentrim.pipeline import PipelineRequest, UnifiedPipeline
from tokentrim.types.message import Message
from tokentrim.types.result import Result
from tokentrim.types.tool import Tool
from tokentrim.transforms.base import Transform

if TYPE_CHECKING:
    from agents import RunConfig

    from tokentrim.integrations.base import IntegrationAdapter
    from tokentrim.tracing import PipelineTracer, TraceStore


AdapterConfigT = TypeVar("AdapterConfigT")


class ComposedPipeline:
    """
    Unified compose-first pipeline.

    A composed pipeline runs all steps against a shared request state.
    """

    def __init__(
        self,
        *,
        owner: Tokentrim,
        steps: tuple[Transform, ...],
        pipeline: UnifiedPipeline,
        default_token_budget: int | None,
    ) -> None:
        self._owner = owner
        self._steps = steps
        self._pipeline = pipeline
        self._default_token_budget = default_token_budget

    def apply(
        self,
        payload: list[Message] | list[Tool] | None = None,
        *,
        context: list[Message] | None = None,
        tools: list[Tool] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        task_hint: str | None = None,
        token_budget: int | None = None,
        trace_store: TraceStore | None = None,
        pipeline_tracer: PipelineTracer | None = None,
    ) -> Result:
        """
        Apply composed steps to the provided payload or payloads.

        `payload` preserves the single-input API.
        `context=` and `tools=` allow mixed pipelines in one call.
        """
        effective_budget = (
            token_budget if token_budget is not None else self._default_token_budget
        )
        normalized_context, normalized_tools = self._normalize_payloads(
            payload=payload,
            context=context,
            tools=tools,
        )
        request = PipelineRequest(
            messages=freeze_messages(normalized_context),
            tools=freeze_tools(normalized_tools),
            user_id=user_id,
            session_id=session_id,
            task_hint=task_hint,
            token_budget=effective_budget,
            trace_store=trace_store,
            pipeline_tracer=pipeline_tracer,
            steps=self._steps,
        )
        return self._pipeline.run(request)

    def _normalize_payloads(
        self,
        *,
        payload: list[Message] | list[Tool] | None,
        context: list[Message] | None,
        tools: list[Tool] | None,
    ) -> tuple[list[Message], list[Tool]]:
        if payload is not None and (context is not None or tools is not None):
            raise TokentrimError(
                "compose(...).apply(...) accepts either `payload` or `context=`/`tools=`, not both."
            )

        if payload is not None:
            kind = self._infer_kind_from_payload(payload)
            if kind == "tools":
                return [], cast(list[Tool], payload)
            return cast(list[Message], payload), []

        if context is not None or tools is not None:
            return list(context or []), list(tools or [])

        raise TokentrimError(
            "compose(...).apply(...) requires `payload` or at least one of `context=`/`tools=`."
        )

    def _infer_kind_from_payload(self, payload: list[Message] | list[Tool]) -> str:
        if not isinstance(payload, list):
            raise TokentrimError("compose(...).apply(...) payload must be a list.")
        if not payload:
            raise TokentrimError(
                "compose(...).apply(...) cannot infer payload kind from an empty list."
            )
        saw_message = False
        saw_tool = False
        for entry in payload:
            if not isinstance(entry, dict):
                raise TokentrimError("compose(...).apply(...) payload entries must be dicts.")
            is_message = (
                isinstance(entry.get("role"), str) and isinstance(entry.get("content"), str)
            )
            is_tool = (
                isinstance(entry.get("name"), str)
                and isinstance(entry.get("description"), str)
                and isinstance(entry.get("input_schema"), dict)
            )
            if is_message and not is_tool:
                saw_message = True
                continue
            if is_tool and not is_message:
                saw_tool = True
                continue
            raise TokentrimError(
                "compose(...).apply(...) payload entries must all be message-shaped or tool-shaped dicts."
            )

        if saw_message and not saw_tool:
            return "context"
        if saw_tool and not saw_message:
            return "tools"
        raise TokentrimError(
            "compose(...).apply(...) payload entries must not mix messages and tools in one positional payload."
        )

    def to_openai_agents(
        self,
        *,
        token_budget: int | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        apply_to_session_history: bool = False,
        apply_to_handoffs: bool = False,
        trace_store: TraceStore | None = None,
        config: RunConfig | None = None,
    ) -> RunConfig:
        """
        Build an OpenAI Agents `RunConfig` from this composed pipeline.

        By default this only installs `call_model_input_filter`. Session and
        handoff hooks are opt-in to keep behavior predictable.
        """
        effective_budget = (
            token_budget if token_budget is not None else self._default_token_budget
        )
        from tokentrim.integrations.openai_agents import (
            OpenAIAgentsAdapter,
            OpenAIAgentsOptions,
        )

        adapter = OpenAIAgentsAdapter(
            options=OpenAIAgentsOptions(
                user_id=user_id,
                session_id=session_id,
                token_budget=effective_budget,
                trace_store=trace_store,
                steps=self._steps,
                apply_to_session_history=apply_to_session_history,
                apply_to_handoffs=apply_to_handoffs,
            )
        )
        return adapter.wrap(self._owner, config=config)


class Tokentrim:
    """
    Local context and tool optimisation for LLM agents.

    The SDK runs inside the caller's process. Model-backed features use LiteLLM
    directly and can be configured independently per feature.
    """

    def __init__(
        self,
        tokenizer: str | None = None,
        *,
        token_budget: int | None = None,
    ) -> None:
        self._default_token_budget = token_budget
        self._pipeline = UnifiedPipeline(
            tokenizer_model=tokenizer,
        )

    def compose(self, *steps: Transform) -> ComposedPipeline:
        """Create a composed pipeline and run it with `.apply(...)`."""
        return ComposedPipeline(
            owner=self,
            steps=tuple(steps),
            pipeline=self._pipeline,
            default_token_budget=self._default_token_budget,
        )

    def wrap_integration(
        self,
        adapter: IntegrationAdapter[AdapterConfigT],
        *,
        config: AdapterConfigT | None = None,
    ) -> AdapterConfigT:
        return adapter.wrap(self, config=config)

    def openai_agents_config(
        self,
        *,
        steps: tuple[Transform, ...] = (),
        token_budget: int | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        apply_to_session_history: bool = False,
        apply_to_handoffs: bool = False,
        trace_store: TraceStore | None = None,
        config: RunConfig | None = None,
    ) -> RunConfig:
        """
        Build an OpenAI Agents `RunConfig` with Tokentrim hooks pre-wired.
        """
        return self.compose(*steps).to_openai_agents(
            token_budget=token_budget,
            user_id=user_id,
            session_id=session_id,
            apply_to_session_history=apply_to_session_history,
            apply_to_handoffs=apply_to_handoffs,
            trace_store=trace_store,
            config=config,
        )
