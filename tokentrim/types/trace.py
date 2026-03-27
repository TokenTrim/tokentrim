from __future__ import annotations

from dataclasses import dataclass

from tokentrim.types.step_trace import StepTrace


@dataclass(frozen=True, slots=True)
class Trace:
    id: str
    token_budget: int | None
    input_tokens: int
    output_tokens: int
    steps: tuple[StepTrace, ...]
