from __future__ import annotations

import uuid
from collections.abc import Sequence

from tokentrim._copy import clone_messages, freeze_messages
from tokentrim._tokens import count_message_tokens
from tokentrim.context.base import ContextStep
from tokentrim.context.compaction import CompactionStep
from tokentrim.context.filter import FilterStep
from tokentrim.context.request import ContextRequest
from tokentrim.context.rlm import RLMStep
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
        steps: Sequence[ContextStep] | None = None,
    ) -> None:
        self._tokenizer_model = tokenizer_model
        self._steps = tuple(
            steps
            or (
                FilterStep(),
                CompactionStep(
                    model=compaction_model,
                    tokenizer_model=tokenizer_model,
                ),
                RLMStep(memory_store=memory_store),
            )
        )

    def run(self, request: ContextRequest) -> ContextResult:
        messages: list[Message] = clone_messages(request.messages)
        selected_steps = self._validate_requested_steps(request.steps)
        step_traces: list[StepTrace] = []

        for step in self._steps:
            if step.name not in selected_steps:
                continue
            before = clone_messages(messages)
            messages = step.run(messages, request)
            step_traces.append(
                StepTrace(
                    step_name=step.name,
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

    def _validate_requested_steps(self, step_names: tuple[str, ...]) -> set[str]:
        known_steps = {step.name for step in self._steps}
        selected_steps = set(step_names)
        unknown_steps = sorted(selected_steps - known_steps)
        if unknown_steps:
            raise TokentrimError(
                "Unknown context steps requested: " + ", ".join(unknown_steps)
            )
        return selected_steps
