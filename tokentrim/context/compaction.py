from __future__ import annotations

from dataclasses import dataclass, replace

from tokentrim._llm import generate_text
from tokentrim._tokens import count_message_tokens
from tokentrim.context.base import ContextStep
from tokentrim.context.request import ContextRequest
from tokentrim.errors.base import TokentrimError
from tokentrim.types.message import Message


@dataclass(frozen=True, slots=True)
class CompactConversation(ContextStep):
    """
    Summarise older messages into one system message when over budget.
    """

    model: str | None = None
    keep_last: int = 6
    tokenizer_model: str | None = None

    @property
    def name(self) -> str:
        return "compaction"

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
        compaction_model: str | None = None,
        memory_store=None,
    ) -> ContextStep:
        del memory_store
        return replace(
            self,
            model=self.model if self.model is not None else compaction_model,
            tokenizer_model=(
                self.tokenizer_model if self.tokenizer_model is not None else tokenizer_model
            ),
        )

    def run(self, messages: list[Message], request: ContextRequest) -> list[Message]:
        if self.keep_last < 1:
            raise TokentrimError("Compaction keep_last must be at least 1.")
        if request.token_budget is None:
            return list(messages)
        if len(messages) <= self.keep_last:
            return list(messages)
        if count_message_tokens(messages, self.tokenizer_model) <= request.token_budget:
            return list(messages)
        if not self.model:
            raise TokentrimError(
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
        except TokentrimError:
            raise
        except Exception as exc:
            raise TokentrimError("Compaction failed.") from exc


CompactionStep = CompactConversation

__all__ = ["CompactConversation", "CompactionStep"]
