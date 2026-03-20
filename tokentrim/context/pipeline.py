from __future__ import annotations

import uuid

from tokentrim._copy import clone_messages, freeze_messages
from tokentrim._tokens import count_message_tokens
from tokentrim.context.compaction import CompactionStep
from tokentrim.context.filter import FilterStep
from tokentrim.context.request import ContextRequest
from tokentrim.context.rlm import RLMStep
from tokentrim.context.store import MemoryStore
from tokentrim.errors.budget import BudgetExceededError
from tokentrim.types.context_result import ContextResult
from tokentrim.types.message import Message


class ContextPipeline:
    """
    Run enabled context steps in a fixed order.
    """

    def __init__(
        self,
        *,
        tokenizer_model: str | None,
        compaction_model: str | None,
        memory_store: MemoryStore,
    ) -> None:
        self._tokenizer_model = tokenizer_model
        self._filter = FilterStep()
        self._compaction = CompactionStep(model=compaction_model)
        self._rlm = RLMStep(memory_store=memory_store)

    def run(self, request: ContextRequest) -> ContextResult:
        messages: list[Message] = clone_messages(request.messages)

        if request.enable_filter:
            messages = self._filter.run(messages)

        if request.enable_compaction:
            messages = self._compaction.run(
                messages,
                token_budget=request.token_budget,
                tokenizer_model=self._tokenizer_model,
            )

        if request.enable_rlm:
            messages = self._rlm.run(
                messages,
                user_id=request.user_id,
                session_id=request.session_id,
            )

        token_count = count_message_tokens(messages, self._tokenizer_model)
        if request.token_budget is not None and token_count > request.token_budget:
            raise BudgetExceededError(budget=request.token_budget, actual=token_count)

        return ContextResult(
            messages=freeze_messages(messages),
            token_count=token_count,
            trace_id=str(uuid.uuid4()),
        )

