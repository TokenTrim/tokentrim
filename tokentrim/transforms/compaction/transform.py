from __future__ import annotations

from dataclasses import dataclass, replace

from tokentrim.core.llm_client import generate_text
from tokentrim.core.token_counting import count_message_tokens
from tokentrim.transforms.compaction.error import (
    CompactionConfigurationError,
    CompactionExecutionError,
)
from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms.base import Transform
from tokentrim.types.message import Message


@dataclass(frozen=True, slots=True)
class CompactConversation(Transform):
    """Summarise older messages into one system message when over budget."""

    model: str | None = None
    keep_last: int = 6
    tokenizer_model: str | None = None

    @property
    def name(self) -> str:
        return "compaction"

    @property
    def kind(self) -> str:
        return "context"

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

    def run(self, messages: list[Message], request: ContextRequest) -> list[Message]:
        if self.keep_last < 1:
            raise CompactionConfigurationError("Compaction keep_last must be at least 1.")
        if request.token_budget is None:
            return list(messages)
        if len(messages) <= self.keep_last:
            return list(messages)
        if count_message_tokens(messages, self.tokenizer_model) <= request.token_budget:
            return list(messages)
        if not self.model:
            raise CompactionConfigurationError(
                "Compaction is enabled but no compaction model is configured."
            )

        to_compact = messages[: -self.keep_last]
        recent = messages[-self.keep_last :]
        summary = self._compress(to_compact)
        return [{"role": "system", "content": summary}, *recent]

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
