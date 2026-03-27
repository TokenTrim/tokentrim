from __future__ import annotations

from dataclasses import dataclass

from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.types.state import PipelineState


@dataclass(frozen=True, slots=True)
class FilterMessages(Transform):
    """Remove low-signal messages before heavier context operations run."""

    @property
    def name(self) -> str:
        return "filter"

    def run(self, state: PipelineState, _request: PipelineRequest) -> PipelineState:
        filtered = [message for message in state.context if message["content"].strip()]
        if not filtered:
            return PipelineState(context=[], tools=state.tools)

        result = []
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

        return PipelineState(context=result, tools=state.tools)
