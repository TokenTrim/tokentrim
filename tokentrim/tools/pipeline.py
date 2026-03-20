from __future__ import annotations

import uuid

from tokentrim._copy import clone_tools, freeze_tools
from tokentrim._tokens import count_tool_tokens
from tokentrim.errors.budget import BudgetExceededError
from tokentrim.tools.bpe import ToolBPEStep
from tokentrim.tools.creator import ToolCreatorStep
from tokentrim.tools.request import ToolsRequest
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
        self._bpe = ToolBPEStep()
        self._creator = ToolCreatorStep(model=tool_creation_model)

    def run(self, request: ToolsRequest) -> ToolsResult:
        tools: list[Tool] = clone_tools(request.tools)
        created_tools: list[Tool] = []

        if request.enable_tool_bpe:
            tools = self._bpe.run(tools)

        if request.enable_tool_creation:
            created_tools = self._creator.run(tools, task_hint=request.task_hint)
            tools = [*tools, *created_tools]

        token_count = count_tool_tokens(tools, self._tokenizer_model)
        if request.token_budget is not None and token_count > request.token_budget:
            raise BudgetExceededError(budget=request.token_budget, actual=token_count)

        return ToolsResult(
            tools=freeze_tools(tools),
            created_tools=freeze_tools(created_tools),
            token_count=token_count,
            trace_id=str(uuid.uuid4()),
        )

