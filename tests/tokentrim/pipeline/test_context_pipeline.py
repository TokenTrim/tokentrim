from __future__ import annotations

import pytest

from tokentrim.errors.base import TokentrimError
from tokentrim.errors.budget import BudgetExceededError
from tokentrim.pipeline import PipelineRequest, UnifiedPipeline
from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms import CompactConversation
from tokentrim.transforms.base import Transform
from tokentrim.types.state import PipelineState


class RecorderStep(Transform):
    def __init__(self, name: str, marker: str, calls: list[str]) -> None:
        self._name = name
        self._marker = marker
        self._calls = calls

    @property
    def name(self) -> str:
        return self._name

    def run(self, state, request):
        del request
        self._calls.append(self._name)
        return PipelineState(
            context=[*state.context, {"role": "system", "content": self._marker}],
            tools=state.tools,
        )


def test_pipeline_runs_steps_in_order() -> None:
    calls: list[str] = []
    pipeline = UnifiedPipeline(
        tokenizer_model=None,
    )
    request = ContextRequest(
        messages=({"role": "user", "content": "hello"},),
        user_id="user",
        session_id="session",
        org_id=None,
        token_budget=1000,
        steps=(
            RecorderStep("filter", "filter", calls),
            RecorderStep("compaction", "compaction", calls),
            RecorderStep("memory_review", "memory_review", calls),
        ),
    )

    result = pipeline.run(request)

    assert calls == ["filter", "compaction", "memory_review"]
    assert [message["content"] for message in result.context] == [
        "hello",
        "filter",
        "compaction",
        "memory_review",
    ]
    assert [trace.step_name for trace in result.trace.steps] == [
        "filter",
        "compaction",
        "memory_review",
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
        org_id=None,
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
        org_id=None,
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
        org_id=None,
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
        org_id=None,
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
        org_id=None,
        token_budget=None,
        steps=(object(),),
    )

    with pytest.raises(TokentrimError) as exc_info:
        pipeline.run(request)

    assert "Pipeline steps must be transforms." in str(exc_info.value)


class ToolRecorderStep(Transform):
    @property
    def name(self) -> str:
        return "tool-recorder"

    def run(self, state, request):
        del request
        return PipelineState(
            context=state.context,
            tools=[*state.tools, {"name": "search", "description": "docs", "input_schema": {}}],
        )


def test_pipeline_runs_mixed_steps_against_shared_request() -> None:
    pipeline = UnifiedPipeline(tokenizer_model=None)
    request = PipelineRequest(
        messages=({"role": "user", "content": "hello"}, {"role": "assistant", "content": "   "}),
        tools=(),
        user_id="user",
        session_id="session",
        org_id="org",
        task_hint="investigate",
        token_budget=1000,
        memory_store=None,
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


def test_pipeline_uses_auto_budget_from_compaction_step(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = UnifiedPipeline(tokenizer_model=None)
    request = ContextRequest(
        messages=tuple({"role": "user", "content": "x" * 80} for _ in range(6)),
        user_id=None,
        session_id=None,
        org_id=None,
        token_budget=None,
        steps=(
            CompactConversation(
                model="gpt-4o-mini",
                keep_last=0,
                context_window=120,
            ),
        ),
    )

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )

    result = pipeline.run(request)

    assert result.trace.token_budget == 68
    assert result.context[0]["role"] == "system"


def test_pipeline_auto_budget_can_raise_budget_exceeded_without_explicit_budget() -> None:
    pipeline = UnifiedPipeline(tokenizer_model=None)
    request = ContextRequest(
        messages=({"role": "user", "content": "x" * 400},),
        user_id=None,
        session_id=None,
        org_id=None,
        token_budget=None,
        steps=(
            CompactConversation(
                model="gpt-4o-mini",
                auto_budget=True,
                context_window=60,
            ),
        ),
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        pipeline.run(request)

    assert exc_info.value.budget == 34
