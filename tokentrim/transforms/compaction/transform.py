from __future__ import annotations

from dataclasses import dataclass, replace

from tokentrim.core.llm_client import generate_text
from tokentrim.core.token_counting import count_message_tokens
from tokentrim.transforms.compaction.error import (
    CompactionConfigurationError,
    CompactionExecutionError,
)
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.types.message import Message
from tokentrim.types.state import PipelineState


@dataclass(frozen=True, slots=True)
class CompactConversation(Transform):
    """Summarise older messages into one system message when over budget."""

    model: str | None = None
    keep_last: int = 6
    reserve_tokens: int = 0
    tokenizer_model: str | None = None

    @property
    def name(self) -> str:
        return "compaction"

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
    ) -> Transform:
        return replace(
            self,
            tokenizer_model=(
                self.tokenizer_model if self.tokenizer_model is not None else tokenizer_model
            ),
        )

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        messages = state.context
        if self.keep_last < 1:
            raise CompactionConfigurationError("Compaction keep_last must be at least 1.")
        if self.reserve_tokens < 0:
            raise CompactionConfigurationError("Compaction reserve_tokens must be at least 0.")
        if request.token_budget is None:
            return state
        if len(messages) <= self.keep_last:
            return state
        target_budget = max(0, request.token_budget - self.reserve_tokens)
        if count_message_tokens(messages, self.tokenizer_model) <= target_budget:
            return state
        if not self.model:
            raise CompactionConfigurationError(
                "Compaction is enabled but no compaction model is configured."
            )

        to_compact = messages[: -self.keep_last]
        recent = messages[-self.keep_last :]
        summary = self._compress(to_compact)
        return PipelineState(
            context=[{"role": "system", "content": summary}, *recent],
            tools=state.tools,
        )

    def _compress(self, messages: list[Message]) -> str:
        history = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        prompt = [
            {
                "role": "system",
                "content": (
                    "Summarise the conversation history for future turns. "
                    "Keep the result concise, factual, and plaintext only."
                ),
            },
            {
                "role": "user",
                "content": history,
            },
        ]
        try:
            return generate_text(
                model=self.model,
                messages=prompt,
                temperature=0.0,
            )
        except CompactionConfigurationError:
            raise
        except Exception as exc:
            raise CompactionExecutionError("Compaction failed.") from exc
