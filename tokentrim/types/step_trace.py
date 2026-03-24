from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StepTrace:
    step_name: str
    input_count: int
    output_count: int
    changed: bool
