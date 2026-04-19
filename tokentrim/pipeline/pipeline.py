from __future__ import annotations

import uuid
from typing import Any

from tokentrim.core.copy_utils import clone_messages, clone_tools, freeze_messages, freeze_tools
from tokentrim.core.token_counting import count_message_tokens, count_tool_tokens
from tokentrim.errors.base import TokentrimError
from tokentrim.errors.budget import BudgetExceededError
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.tracing import (
    build_transform_span_data,
    build_transform_span_name,
    resolve_pipeline_tracer,
)
from tokentrim.types.result import Result
from tokentrim.types.state import PipelineState
from tokentrim.types.step_trace import StepTrace
from tokentrim.types.trace import Trace
from tokentrim.transforms.base import Transform


class UnifiedPipeline:
    """Single pipeline runtime for both context and tools transforms."""

    def __init__(
        self,
        *,
        tokenizer_model: str | None,
    ) -> None:
        self._tokenizer_model = tokenizer_model

    def run(self, request: PipelineRequest) -> Result:
        if not isinstance(request, PipelineRequest):
            raise TokentrimError("Unsupported request type for UnifiedPipeline.")

        trace, frozen_messages, frozen_tools = self._run_state(request=request)
        return Result(
            context=frozen_messages,
            tools=frozen_tools,
            trace=trace,
        )

    def _run_state(
        self,
        *,
        request: PipelineRequest,
    ) -> tuple[Trace, tuple[Any, ...], tuple[Any, ...]]:
        resolved_steps = self._resolve_steps(request.steps)
        effective_budget = self._resolve_token_budget(
            request.token_budget,
            resolved_steps=resolved_steps,
        )
        request = PipelineRequest(
            messages=request.messages,
            tools=request.tools,
            user_id=request.user_id,
            session_id=request.session_id,
            org_id=request.org_id,
            task_hint=request.task_hint,
            token_budget=effective_budget,
            memory_store=request.memory_store,
            injector_model=request.injector_model,
            agent_aware_memory=request.agent_aware_memory,
            trace_store=request.trace_store,
            pipeline_tracer=request.pipeline_tracer,
            steps=resolved_steps,
        )
        step_traces: list[StepTrace] = []
        state = PipelineState(
            context=clone_messages(request.messages),
            tools=clone_tools(request.tools),
        )
        input_tokens = self._count_total_tokens(state)
        pipeline_tracer = resolve_pipeline_tracer(request.pipeline_tracer)

        for step in request.steps:
            if not isinstance(step, Transform):
                raise TokentrimError("Pipeline steps must be transforms.")

            resolved_step = step
            before_state = PipelineState(
                context=clone_messages(state.context),
                tools=clone_tools(state.tools),
            )
            before_tokens = self._count_total_tokens(before_state)
            input_items = len(before_state.context) + len(before_state.tools)
            with pipeline_tracer.start_span(
                name=build_transform_span_name(resolved_step.name),
                data=build_transform_span_data(
                    transform_name=resolved_step.name,
                    token_budget=request.token_budget,
                    input_items=input_items,
                    input_tokens=before_tokens,
                ),
            ) as transform_span:
                try:
                    state = resolved_step.run(state, request)
                except Exception as exc:
                    transform_span.set_error(exc)
                    raise
                after_tokens = self._count_total_tokens(state)
                output_items = len(state.context) + len(state.tools)
                changed = state != before_state
                transform_span.set_data(
                    build_transform_span_data(
                        transform_name=resolved_step.name,
                        token_budget=request.token_budget,
                        input_items=input_items,
                        input_tokens=before_tokens,
                        output_items=output_items,
                        output_tokens=after_tokens,
                        changed=changed,
                    )
                )
            step_traces.append(
                StepTrace(
                    step_name=resolved_step.name,
                    input_items=input_items,
                    output_items=output_items,
                    input_tokens=before_tokens,
                    output_tokens=after_tokens,
                    changed=changed,
                )
            )

        output_tokens = self._count_total_tokens(state)
        if request.token_budget is not None and output_tokens > request.token_budget:
            raise BudgetExceededError(budget=request.token_budget, actual=output_tokens)

        return (
            Trace(
                id=str(uuid.uuid4()),
                token_budget=request.token_budget,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                steps=tuple(step_traces),
            ),
            freeze_messages(state.context),
            freeze_tools(state.tools),
        )

    def _resolve_steps(self, steps: tuple[Transform, ...]) -> tuple[Transform, ...]:
        resolved_steps: list[Transform] = []
        for step in steps:
            if not isinstance(step, Transform):
                raise TokentrimError("Pipeline steps must be transforms.")
            resolved_steps.append(step.resolve(tokenizer_model=self._tokenizer_model))
        return tuple(resolved_steps)

    def _resolve_token_budget(
        self,
        token_budget: int | None,
        *,
        resolved_steps: tuple[Transform, ...],
    ) -> int | None:
        effective_budget = token_budget
        for step in resolved_steps:
            effective_budget = step.resolve_token_budget(effective_budget)
        return effective_budget

    def _count_total_tokens(self, state: PipelineState) -> int:
        return count_message_tokens(state.context, self._tokenizer_model) + count_tool_tokens(
            state.tools,
            self._tokenizer_model,
        )
