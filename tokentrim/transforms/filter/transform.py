from __future__ import annotations

from dataclasses import dataclass

from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms.base import Transform
from tokentrim.types.message import Message


@dataclass(frozen=True, slots=True)
class FilterMessages(Transform):
    """Remove low-signal messages before heavier context operations run."""

    @property
    def name(self) -> str:
        return "filter"

    @property
    def kind(self) -> str:
        return "context"

    def run(self, messages: list[Message], _request: ContextRequest) -> list[Message]:
        filtered = [message for message in messages if message["content"].strip()]
        if not filtered:
            return []

        result: list[Message] = []
        index = 0
        while index < len(filtered):
            current = filtered[index]
            run_length = 1

            while index + run_length < len(filtered):
                candidate = filtered[index + run_length]
                if (
                    candidate["role"] == current["role"]
                    and candidate["content"] == current["content"]
                ):
                    run_length += 1
                    continue
                break

            content = current["content"]
            if run_length > 1:
                content = f"{content} [repeated {run_length}x]"

            result.append({"role": current["role"], "content": content})
            index += run_length

        return result
