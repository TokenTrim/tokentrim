from __future__ import annotations

import uuid
from collections.abc import Sequence

from tokentrim._copy import clone_tools, freeze_tools
from tokentrim._tokens import count_tool_tokens
from tokentrim.errors.base import TokentrimError
from tokentrim.errors.budget import BudgetExceededError
from tokentrim.tools.base import ToolStep
from tokentrim.tools.bpe import ToolBPEStep
from tokentrim.tools.creator import ToolCreatorStep
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
        steps: Sequence[ToolStep] | None = None,
    ) -> None:
        self._tokenizer_model = tokenizer_model
        self._steps = tuple(
            steps
            or (
                ToolBPEStep(),
                ToolCreatorStep(model=tool_creation_model),
            )
        )

    def run(self, request: ToolsRequest) -> ToolsResult:
        tools: list[Tool] = clone_tools(request.tools)
        selected_steps = self._validate_requested_steps(request.steps)
        step_traces: list[StepTrace] = []

        for step in self._steps:
            if step.name not in selected_steps:
                continue
            before = clone_tools(tools)
            tools = step.run(tools, request)
            step_traces.append(
                StepTrace(
                    step_name=step.name,
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

    def _validate_requested_steps(self, step_names: tuple[str, ...]) -> set[str]:
        known_steps = {step.name for step in self._steps}
        selected_steps = set(step_names)
        unknown_steps = sorted(selected_steps - known_steps)
        if unknown_steps:
            raise TokentrimError(
                "Unknown tool steps requested: " + ", ".join(unknown_steps)
            )
        return selected_steps
