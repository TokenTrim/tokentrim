from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar, cast

from tokentrim.core.copy_utils import freeze_messages, freeze_tools
from tokentrim.errors.base import TokentrimError
from tokentrim.pipeline import ContextRequest, ToolsRequest, UnifiedPipeline
from tokentrim.types.message import Message
from tokentrim.types.result import Result
from tokentrim.types.tool import Tool
from tokentrim.transforms.base import Transform

if TYPE_CHECKING:
    from agents import RunConfig

    from tokentrim.integrations.base import IntegrationAdapter
    from tokentrim.integrations.openai_agents import OpenAIAgentsAdapter, OpenAIAgentsOptions


AdapterConfigT = TypeVar("AdapterConfigT")
PayloadKind = Literal["context", "tools"]


class ComposedPipeline:
    """
    Unified compose-first pipeline.

    A composed pipeline contains either context steps or tool steps. `apply(...)`
    runs the correct internal pipeline based on step type (or payload shape when
    no steps are provided).
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
        self._kind: PayloadKind | None = self._resolve_kind_from_steps(steps)

    def apply(
        self,
        payload: list[Message] | list[Tool],
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        task_hint: str | None = None,
        token_budget: int | None = None,
    ) -> Result:
        """
        Apply composed steps to the provided payload.

        Context runs use `user_id` and `session_id`.
        Tool runs use `task_hint`.
        """
        effective_budget = (
            token_budget if token_budget is not None else self._default_token_budget
        )
        kind = self._kind or self._infer_kind_from_payload(payload)
        if kind == "context":
            request = ContextRequest(
                messages=freeze_messages(cast(list[Message], payload)),
                user_id=user_id,
                session_id=session_id,
                token_budget=effective_budget,
                steps=self._steps,
            )
            return self._pipeline.run(request)

        request = ToolsRequest(
            tools=freeze_tools(cast(list[Tool], payload)),
            task_hint=task_hint,
            token_budget=effective_budget,
            steps=self._steps,
        )
        return self._pipeline.run(request)

    def _resolve_kind_from_steps(
        self,
        steps: tuple[Transform, ...],
    ) -> PayloadKind | None:
        if not steps:
            return None

        first_kind = steps[0].kind
        if first_kind not in ("context", "tools"):
            raise TokentrimError("compose(...) received an unsupported step object.")
        if any(step.kind != first_kind for step in steps):
            raise TokentrimError("compose(...) cannot mix context and tool steps.")

        return cast(PayloadKind, first_kind)

    def _infer_kind_from_payload(self, payload: list[Message] | list[Tool]) -> PayloadKind:
        if not isinstance(payload, list):
            raise TokentrimError("compose(...).apply(...) payload must be a list.")
        if not payload:
            raise TokentrimError(
                "compose(...).apply(...) cannot infer payload kind from an empty list."
            )
        head = payload[0]
        if not isinstance(head, dict):
            raise TokentrimError("compose(...).apply(...) payload entries must be dicts.")

        is_message = (
            isinstance(head.get("role"), str) and isinstance(head.get("content"), str)
        )
        is_tool = (
            isinstance(head.get("name"), str)
            and isinstance(head.get("description"), str)
            and isinstance(head.get("input_schema"), dict)
        )
        if is_message and not is_tool:
            return "context"
        if is_tool and not is_message:
            return "tools"
        raise TokentrimError(
            "compose(...).apply(...) payload kind is ambiguous. Pass steps to disambiguate."
        )

    def to_openai_agents(
        self,
        *,
        token_budget: int | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        apply_to_session_history: bool = False,
        apply_to_handoffs: bool = False,
        config: RunConfig | None = None,
    ) -> RunConfig:
        """
        Build an OpenAI Agents `RunConfig` from this composed pipeline.

        By default this only installs `call_model_input_filter`. Session and
        handoff hooks are opt-in to keep behavior predictable.
        """
        if self._kind == "tools":
            raise TokentrimError(
                "compose(...).to_openai_agents(...) supports context transforms only."
            )

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
            config=config,
        )
