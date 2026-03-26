from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, cast

from tokentrim.core.copy_utils import clone_messages, clone_tools, freeze_messages, freeze_tools
from tokentrim.core.token_counting import count_message_tokens, count_tool_tokens
from tokentrim.errors.base import TokentrimError
from tokentrim.errors.budget import BudgetExceededError
from tokentrim.types.result import Result
from tokentrim.types.step_trace import StepTrace
from tokentrim.types.trace import Trace
from tokentrim.pipeline.requests import ContextRequest, ToolsRequest
from tokentrim.transforms.base import Transform


class UnifiedPipeline:
    """Single pipeline runtime for both context and tools transforms."""

    def __init__(
        self,
        *,
        tokenizer_model: str | None,
    ) -> None:
        self._tokenizer_model = tokenizer_model

    def run(self, request: ContextRequest | ToolsRequest) -> Result:
        if isinstance(request, ContextRequest):
            trace, frozen_payload = self._run_payload(
                request=request,
                payload=clone_messages(request.messages),
                expected_kind="context",
                invalid_step_message="Context steps must be context transforms.",
                clone_payload=clone_messages,
                freeze_payload=freeze_messages,
                count_tokens=count_message_tokens,
            )
            return Result(context=cast(tuple, frozen_payload), trace=trace)

        if isinstance(request, ToolsRequest):
            trace, frozen_payload = self._run_payload(
                request=request,
                payload=clone_tools(request.tools),
                expected_kind="tools",
                invalid_step_message="Tool steps must be tools transforms.",
                clone_payload=clone_tools,
                freeze_payload=freeze_tools,
                count_tokens=count_tool_tokens,
            )
            return Result(tools=cast(tuple, frozen_payload), trace=trace)

        raise TokentrimError("Unsupported request type for UnifiedPipeline.")

    def _run_payload(
        self,
        *,
        request: ContextRequest | ToolsRequest,
        payload: list[Any],
        expected_kind: str,
        invalid_step_message: str,
        clone_payload: Callable[[list[Any]], list[Any]],
        freeze_payload: Callable[[list[Any]], tuple[Any, ...]],
        count_tokens: Callable[[list[Any], str | None], int],
    ) -> tuple[Trace, tuple[Any, ...]]:
        step_traces: list[StepTrace] = []
        input_tokens = count_tokens(payload, self._tokenizer_model)

        for step in request.steps:
            if not isinstance(step, Transform) or step.kind != expected_kind:
                raise TokentrimError(invalid_step_message)

            resolved_step = step.resolve(tokenizer_model=self._tokenizer_model)
            before = clone_payload(payload)
            before_tokens = count_tokens(before, self._tokenizer_model)
            payload = resolved_step.run(payload, request)
            after_tokens = count_tokens(payload, self._tokenizer_model)
            step_traces.append(
                StepTrace(
                    step_name=resolved_step.name,
                    input_items=len(before),
                    output_items=len(payload),
                    input_tokens=before_tokens,
                    output_tokens=after_tokens,
                    changed=payload != before,
                )
            )

        output_tokens = count_tokens(payload, self._tokenizer_model)
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
            freeze_payload(payload),
        )
