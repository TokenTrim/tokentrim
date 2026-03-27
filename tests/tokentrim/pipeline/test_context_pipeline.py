from __future__ import annotations

import pytest

from tokentrim.transforms.base import Transform
from tokentrim.pipeline import PipelineRequest, UnifiedPipeline
from tokentrim.pipeline.requests import ContextRequest
from tokentrim.errors.base import TokentrimError
from tokentrim.errors.budget import BudgetExceededError


class RecorderStep(Transform):
    def __init__(self, name: str, marker: str, calls: list[str]) -> None:
        self._name = name
        self._marker = marker
        self._calls = calls

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> str:
        return "context"

    def run(self, messages, request):
        del request
        self._calls.append(self._name)
        return [*messages, {"role": "system", "content": self._marker}]


def test_pipeline_runs_steps_in_order() -> None:
    calls: list[str] = []
    pipeline = UnifiedPipeline(
        tokenizer_model=None,
    )
    request = ContextRequest(
        messages=({"role": "user", "content": "hello"},),
        user_id="user",
        session_id="session",
        token_budget=1000,
        steps=(
            RecorderStep("filter", "filter", calls),
            RecorderStep("compaction", "compaction", calls),
            RecorderStep("rlm", "rlm", calls),
        ),
    )

    result = pipeline.run(request)

    assert calls == ["filter", "compaction", "rlm"]
    assert [message["content"] for message in result.context] == [
        "hello",
        "filter",
        "compaction",
        "rlm",
    ]
    assert [trace.step_name for trace in result.trace.steps] == [
        "filter",
        "compaction",
        "rlm",
    ]
    assert isinstance(result.context, tuple)
    assert result.trace.id
    assert result.trace.output_tokens > 0


def test_pipeline_skips_disabled_steps() -> None:
    calls: list[str] = []
    pipeline = UnifiedPipeline(
        tokenizer_model=None,
    )
    request = ContextRequest(
        messages=({"role": "user", "content": "hello"},),
        user_id="user",
        session_id="session",
        token_budget=1000,
        steps=(RecorderStep("filter", "filter", calls),),
    )

    pipeline.run(request)

    assert calls == ["filter"]


def test_pipeline_does_not_mutate_input_messages() -> None:
    original = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
    snapshot = [dict(message) for message in original]
    pipeline = UnifiedPipeline(
        tokenizer_model=None,
    )
    request = ContextRequest(
        messages=tuple(original),
        user_id=None,
        session_id=None,
        token_budget=None,
        steps=(RecorderStep("filter", "filter", []),),
    )

    pipeline.run(request)

    assert original == snapshot


def test_pipeline_result_is_independent_from_input_dicts() -> None:
    original = [{"role": "user", "content": "hello"}]
    pipeline = UnifiedPipeline(
        tokenizer_model=None,
    )
    request = ContextRequest(
        messages=tuple(original),
        user_id=None,
        session_id=None,
        token_budget=None,
        steps=(),
    )

    result = pipeline.run(request)
    result.context[0]["content"] = "changed"

    assert original == [{"role": "user", "content": "hello"}]


def test_pipeline_raises_when_final_budget_is_exceeded() -> None:
    pipeline = UnifiedPipeline(
        tokenizer_model=None,
    )
    request = ContextRequest(
        messages=({"role": "user", "content": "x" * 80},),
        user_id=None,
        session_id=None,
        token_budget=5,
        steps=(),
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        pipeline.run(request)

    assert exc_info.value.actual > exc_info.value.budget


def test_pipeline_raises_for_unknown_steps() -> None:
    pipeline = UnifiedPipeline(
        tokenizer_model=None,
    )
    request = ContextRequest(
        messages=({"role": "user", "content": "hello"},),
        user_id=None,
        session_id=None,
        token_budget=None,
        steps=(object(),),
    )

    with pytest.raises(TokentrimError) as exc_info:
        pipeline.run(request)

    assert "Pipeline steps must be context or tools transforms." in str(exc_info.value)


class ToolRecorderStep(Transform):
    @property
    def name(self) -> str:
        return "tool-recorder"

    @property
    def kind(self) -> str:
        return "tools"

    def run(self, tools, request):
        del request
        return [
            *tools,
            {"name": "search", "description": "docs", "input_schema": {}},
        ]


def test_pipeline_runs_mixed_steps_against_shared_request() -> None:
    pipeline = UnifiedPipeline(tokenizer_model=None)
    request = PipelineRequest(
        messages=({"role": "user", "content": "hello"}, {"role": "assistant", "content": "   "}),
        tools=(),
        user_id="user",
        session_id="session",
        task_hint="investigate",
        token_budget=1000,
        steps=(RecorderStep("filter", "filter", []), ToolRecorderStep()),
    )

    result = pipeline.run(request)

    assert result.context == (
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "   "},
        {"role": "system", "content": "filter"},
    )
    assert result.tools == ({"name": "search", "description": "docs", "input_schema": {}},)
    assert [trace.step_name for trace in result.trace.steps] == ["filter", "tool-recorder"]
