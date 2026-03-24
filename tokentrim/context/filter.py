from __future__ import annotations

from dataclasses import dataclass

from tokentrim.context.base import ContextStep
from tokentrim.context.request import ContextRequest
from tokentrim.types.message import Message


@dataclass(frozen=True, slots=True)
class FilterMessages(ContextStep):
    """
    Remove low-signal messages before heavier context operations run.
    """

    @property
    def name(self) -> str:
        return "filter"

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


FilterStep = FilterMessages

__all__ = ["FilterMessages", "FilterStep"]
