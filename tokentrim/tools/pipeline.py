from __future__ import annotations

import uuid

from tokentrim._copy import clone_tools, freeze_tools
from tokentrim._tokens import count_tool_tokens
from tokentrim.errors.base import TokentrimError
from tokentrim.errors.budget import BudgetExceededError
from tokentrim.tools.base import ToolStep
from tokentrim.tools.request import ToolsRequest
from tokentrim.types.step_trace import StepTrace
from tokentrim.types.tool import Tool
from tokentrim.types.tools_result import ToolsResult


class ToolsPipeline:
    """
    Run enabled tool steps in a fixed order.
    """

    def __init__(
        self,
        *,
        tokenizer_model: str | None,
        tool_creation_model: str | None,
    ) -> None:
        self._tokenizer_model = tokenizer_model
        self._tool_creation_model = tool_creation_model

    def run(self, request: ToolsRequest) -> ToolsResult:
        tools: list[Tool] = clone_tools(request.tools)
        step_traces: list[StepTrace] = []

        for step in request.steps:
            if not isinstance(step, ToolStep):
                raise TokentrimError("Tool steps must be ToolStep objects.")
            resolved_step = step.resolve(
                tokenizer_model=self._tokenizer_model,
                tool_creation_model=self._tool_creation_model,
            )
            before = clone_tools(tools)
            tools = resolved_step.run(tools, request)
            step_traces.append(
                StepTrace(
                    step_name=resolved_step.name,
                    input_count=len(before),
                    output_count=len(tools),
                    changed=tools != before,
                )
            )

        token_count = count_tool_tokens(tools, self._tokenizer_model)
        if request.token_budget is not None and token_count > request.token_budget:
            raise BudgetExceededError(budget=request.token_budget, actual=token_count)

        return ToolsResult(
            tools=freeze_tools(tools),
            step_traces=tuple(step_traces),
            token_count=token_count,
            trace_id=str(uuid.uuid4()),
        )
