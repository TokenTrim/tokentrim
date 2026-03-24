from __future__ import annotations

from tokentrim._llm import generate_text
from tokentrim._tokens import count_message_tokens
from tokentrim.context.base import ContextStep
from tokentrim.context.request import ContextRequest
from tokentrim.errors.base import TokentrimError
from tokentrim.types.message import Message


class CompactionStep(ContextStep):
    """
    Summarise older messages into one system message when over budget.
    """

    _RECENT_MESSAGES_TO_KEEP = 6

    def __init__(self, model: str | None, tokenizer_model: str | None) -> None:
        self._model = model
        self._tokenizer_model = tokenizer_model

    @property
    def name(self) -> str:
        return "compaction"

    def run(self, messages: list[Message], request: ContextRequest) -> list[Message]:
        if request.token_budget is None:
            return list(messages)
        if len(messages) <= self._RECENT_MESSAGES_TO_KEEP:
            return list(messages)
        if count_message_tokens(messages, self._tokenizer_model) <= request.token_budget:
            return list(messages)
        if not self._model:
            raise TokentrimError(
                "Compaction is enabled but no compaction model is configured."
            )

        to_compact = messages[: -self._RECENT_MESSAGES_TO_KEEP]
        recent = messages[-self._RECENT_MESSAGES_TO_KEEP :]
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
                model=self._model,
                messages=prompt,
                temperature=0.0,
            )
        except TokentrimError:
            raise
        except Exception as exc:
            raise TokentrimError("Compaction failed.") from exc
