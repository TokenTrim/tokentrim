from __future__ import annotations

import uuid
from typing import Any

from tokentrim.core.copy_utils import clone_messages, clone_tools, freeze_messages, freeze_tools
from tokentrim.core.token_counting import count_message_tokens, count_tool_tokens
from tokentrim.errors.base import TokentrimError
from tokentrim.errors.budget import BudgetExceededError
from tokentrim.pipeline.requests import PipelineRequest
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
        step_traces: list[StepTrace] = []
        state = PipelineState(
            context=clone_messages(request.messages),
            tools=clone_tools(request.tools),
        )
        input_tokens = self._count_total_tokens(state)

        for step in request.steps:
            if not isinstance(step, Transform):
                raise TokentrimError("Pipeline steps must be transforms.")

            resolved_step = step.resolve(tokenizer_model=self._tokenizer_model)
            before_state = PipelineState(
                context=clone_messages(state.context),
                tools=clone_tools(state.tools),
            )
            before_tokens = self._count_total_tokens(before_state)
            state = resolved_step.run(state, request)
            after_tokens = self._count_total_tokens(state)
            step_traces.append(
                StepTrace(
                    step_name=resolved_step.name,
                    input_items=len(before_state.context) + len(before_state.tools),
                    output_items=len(state.context) + len(state.tools),
                    input_tokens=before_tokens,
                    output_tokens=after_tokens,
                    changed=state != before_state,
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

    def _count_total_tokens(self, state: PipelineState) -> int:
        return count_message_tokens(state.context, self._tokenizer_model) + count_tool_tokens(
            state.tools,
            self._tokenizer_model,
        )
