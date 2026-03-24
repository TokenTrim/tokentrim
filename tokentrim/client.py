from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from tokentrim._copy import freeze_messages, freeze_tools
from tokentrim.context.pipeline import ContextPipeline
from tokentrim.context.request import ContextRequest
from tokentrim.context.store import MemoryStore, NoOpMemoryStore
from tokentrim.tools.pipeline import ToolsPipeline
from tokentrim.tools.request import ToolsRequest
from tokentrim.types.context_result import ContextResult
from tokentrim.types.message import Message
from tokentrim.types.tool import Tool
from tokentrim.types.tools_result import ToolsResult

if TYPE_CHECKING:
    from tokentrim.integrations.base import IntegrationAdapter


AdapterConfigT = TypeVar("AdapterConfigT")


class Tokentrim:
    """
    Local context and tool optimisation for LLM agents.

    The SDK runs inside the caller's process. Model-backed features use LiteLLM
    directly and can be configured independently per feature.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        compaction_model: str | None = None,
        tool_creation_model: str | None = None,
        token_budget: int | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        resolved_memory_store = memory_store or NoOpMemoryStore()
        self._default_token_budget = token_budget
        self._context_pipeline = ContextPipeline(
            tokenizer_model=model,
            compaction_model=compaction_model or model,
            memory_store=resolved_memory_store,
        )
        self._tools_pipeline = ToolsPipeline(
            tokenizer_model=model,
            tool_creation_model=tool_creation_model or model,
        )

    def get_better_context(
        self,
        messages: list[Message],
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        token_budget: int | None = None,
        steps: Sequence[str] = (),
    ) -> ContextResult:
        request = ContextRequest(
            messages=freeze_messages(messages),
            user_id=user_id,
            session_id=session_id,
            token_budget=token_budget if token_budget is not None else self._default_token_budget,
            steps=tuple(steps),
        )
        return self._context_pipeline.run(request)

    def get_better_tools(
        self,
        tools: list[Tool],
        *,
        task_hint: str | None = None,
        token_budget: int | None = None,
        steps: Sequence[str] = (),
    ) -> ToolsResult:
        request = ToolsRequest(
            tools=freeze_tools(tools),
            task_hint=task_hint,
            token_budget=token_budget if token_budget is not None else self._default_token_budget,
            steps=tuple(steps),
        )
        return self._tools_pipeline.run(request)

    def wrap_integration(
        self,
        adapter: IntegrationAdapter[AdapterConfigT],
        *,
        config: AdapterConfigT | None = None,
    ) -> AdapterConfigT:
        return adapter.wrap(self, config=config)
