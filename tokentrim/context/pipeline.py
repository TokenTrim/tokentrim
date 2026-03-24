from __future__ import annotations

import uuid

from tokentrim._copy import clone_messages, freeze_messages
from tokentrim._tokens import count_message_tokens
from tokentrim.context.base import ContextStep
from tokentrim.context.request import ContextRequest
from tokentrim.context.store import MemoryStore
from tokentrim.errors.base import TokentrimError
from tokentrim.errors.budget import BudgetExceededError
from tokentrim.types.context_result import ContextResult
from tokentrim.types.message import Message
from tokentrim.types.step_trace import StepTrace


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
        self._compaction_model = compaction_model
        self._memory_store = memory_store

    def run(self, request: ContextRequest) -> ContextResult:
        messages: list[Message] = clone_messages(request.messages)
        step_traces: list[StepTrace] = []

        for step in request.steps:
            if not isinstance(step, ContextStep):
                raise TokentrimError("Context steps must be ContextStep objects.")
            resolved_step = step.resolve(
                tokenizer_model=self._tokenizer_model,
                compaction_model=self._compaction_model,
                memory_store=self._memory_store,
            )
            before = clone_messages(messages)
            messages = resolved_step.run(messages, request)
            step_traces.append(
                StepTrace(
                    step_name=resolved_step.name,
                    input_count=len(before),
                    output_count=len(messages),
                    changed=messages != before,
                )
            )

        token_count = count_message_tokens(messages, self._tokenizer_model)
        if request.token_budget is not None and token_count > request.token_budget:
            raise BudgetExceededError(budget=request.token_budget, actual=token_count)

        return ContextResult(
            messages=freeze_messages(messages),
            step_traces=tuple(step_traces),
            token_count=token_count,
            trace_id=str(uuid.uuid4()),
        )
