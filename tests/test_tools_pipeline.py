from __future__ import annotations

import pytest

from tokentrim.errors.base import TokentrimError
from tokentrim.errors.budget import BudgetExceededError
from tokentrim.tools.base import ToolStep
from tokentrim.tools.pipeline import ToolsPipeline
from tokentrim.tools.request import ToolsRequest


class RecorderStep(ToolStep):
    def __init__(self, name: str, description: str, calls: list[str]) -> None:
        self._name = name
        self._description = description
        self._calls = calls

    @property
    def name(self) -> str:
        return self._name

    def run(self, tools, request):
        del request
        self._calls.append(self._name)
        return [
            *tools,
            {
                "name": self._name,
                "description": self._description,
                "input_schema": {},
            },
        ]


class CreatorRecorder(ToolStep):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    @property
    def name(self) -> str:
        return "creator"

    def run(self, tools, request):
        del request
        self._calls.append("creator")
        return [
            *tools,
            {
                "name": "generated",
                "description": "generated description",
                "input_schema": {},
            }
        ]


def test_tools_pipeline_runs_steps_in_order() -> None:
    calls: list[str] = []
    pipeline = ToolsPipeline(
        tokenizer_model=None,
        tool_creation_model=None,
        steps=(
            RecorderStep("bpe", "bpe description", calls),
            CreatorRecorder(calls),
        ),
    )
    request = ToolsRequest(
        tools=(
            {
                "name": "base",
                "description": "base description",
                "input_schema": {},
            },
        ),
        task_hint="hint",
        token_budget=1000,
        steps=("bpe", "creator"),
    )

    result = pipeline.run(request)

    assert calls == ["bpe", "creator"]
    assert [tool["name"] for tool in result.tools] == ["base", "bpe", "generated"]
    assert [trace.step_name for trace in result.step_traces] == ["bpe", "creator"]
    assert [trace.output_count - trace.input_count for trace in result.step_traces] == [1, 1]
    assert isinstance(result.tools, tuple)
    assert result.trace_id
    assert result.token_count > 0


def test_tools_pipeline_does_not_mutate_nested_input_schema() -> None:
    pipeline = ToolsPipeline(tokenizer_model=None, tool_creation_model=None)
    original = [
        {
            "name": "search",
            "description": "docs",
            "input_schema": {"properties": {"query": {"type": "string"}}},
        }
    ]
    request = ToolsRequest(
        tools=tuple(original),
        task_hint=None,
        token_budget=None,
        steps=("bpe",),
    )

    result = pipeline.run(request)
    result.tools[0]["input_schema"]["properties"]["query"]["type"] = "number"

    assert original[0]["input_schema"]["properties"]["query"]["type"] == "string"


def test_tools_pipeline_raises_when_final_budget_is_exceeded() -> None:
    pipeline = ToolsPipeline(tokenizer_model=None, tool_creation_model=None)
    request = ToolsRequest(
        tools=(
            {
                "name": "tool",
                "description": "x" * 200,
                "input_schema": {},
            },
        ),
        task_hint=None,
        token_budget=5,
        steps=(),
    )

    with pytest.raises(BudgetExceededError):
        pipeline.run(request)


def test_tools_pipeline_raises_for_unknown_steps() -> None:
    pipeline = ToolsPipeline(tokenizer_model=None, tool_creation_model=None)
    request = ToolsRequest(
        tools=tuple(),
        task_hint=None,
        token_budget=None,
        steps=("unknown",),
    )

    with pytest.raises(TokentrimError) as exc_info:
        pipeline.run(request)

    assert "Unknown tool steps requested" in str(exc_info.value)
