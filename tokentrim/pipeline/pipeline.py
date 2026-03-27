from __future__ import annotations

import uuid
from typing import Any, cast

from tokentrim.core.copy_utils import clone_messages, clone_tools, freeze_messages, freeze_tools
from tokentrim.core.token_counting import count_message_tokens, count_tool_tokens
from tokentrim.errors.base import TokentrimError
from tokentrim.errors.budget import BudgetExceededError
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.types.result import Result
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
            context=cast(tuple, frozen_messages) if request.messages or self._has_context_steps(request) else None,
            tools=cast(tuple, frozen_tools) if request.tools or self._has_tools_steps(request) else None,
            trace=trace,
        )

    def _run_state(
        self,
        *,
        request: PipelineRequest,
    ) -> tuple[Trace, tuple[Any, ...], tuple[Any, ...]]:
        step_traces: list[StepTrace] = []
        messages = clone_messages(request.messages)
        tools = clone_tools(request.tools)
        input_tokens = self._count_total_tokens(messages=messages, tools=tools)

        for step in request.steps:
            if not isinstance(step, Transform) or step.kind not in ("context", "tools"):
                raise TokentrimError("Pipeline steps must be context or tools transforms.")

            resolved_step = step.resolve(tokenizer_model=self._tokenizer_model)
            before_messages = clone_messages(messages)
            before_tools = clone_tools(tools)
            before_tokens = self._count_total_tokens(messages=before_messages, tools=before_tools)

            if resolved_step.kind == "context":
                messages = cast(list, resolved_step.run(messages, request))
            else:
                tools = cast(list, resolved_step.run(tools, request))

            after_tokens = self._count_total_tokens(messages=messages, tools=tools)
            step_traces.append(
                StepTrace(
                    step_name=resolved_step.name,
                    input_items=len(before_messages) + len(before_tools),
                    output_items=len(messages) + len(tools),
                    input_tokens=before_tokens,
                    output_tokens=after_tokens,
                    changed=messages != before_messages or tools != before_tools,
                )
            )

        output_tokens = self._count_total_tokens(messages=messages, tools=tools)
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
            freeze_messages(messages),
            freeze_tools(tools),
        )

    def _count_total_tokens(self, *, messages: list[Any], tools: list[Any]) -> int:
        return count_message_tokens(messages, self._tokenizer_model) + count_tool_tokens(
            tools,
            self._tokenizer_model,
        )

    def _has_context_steps(self, request: PipelineRequest) -> bool:
        return any(step.kind == "context" for step in request.steps if isinstance(step, Transform))

    def _has_tools_steps(self, request: PipelineRequest) -> bool:
        return any(step.kind == "tools" for step in request.steps if isinstance(step, Transform))
