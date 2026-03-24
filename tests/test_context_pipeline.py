from __future__ import annotations

import pytest

from tokentrim.context.base import ContextStep
from tokentrim.context.pipeline import ContextPipeline
from tokentrim.context.request import ContextRequest
from tokentrim.context.store import NoOpMemoryStore
from tokentrim.errors.base import TokentrimError
from tokentrim.errors.budget import BudgetExceededError


class RecorderStep(ContextStep):
    def __init__(self, name: str, marker: str, calls: list[str]) -> None:
        self._name = name
        self._marker = marker
        self._calls = calls

    @property
    def name(self) -> str:
        return self._name

    def run(self, messages, request):
        del request
        self._calls.append(self._name)
        return [*messages, {"role": "system", "content": self._marker}]


def test_pipeline_runs_steps_in_order() -> None:
    calls: list[str] = []
    pipeline = ContextPipeline(
        tokenizer_model=None,
        compaction_model=None,
        memory_store=NoOpMemoryStore(),
        steps=(
            RecorderStep("filter", "filter", calls),
            RecorderStep("compaction", "compaction", calls),
            RecorderStep("rlm", "rlm", calls),
        ),
    )
    request = ContextRequest(
        messages=({"role": "user", "content": "hello"},),
        user_id="user",
        session_id="session",
        token_budget=1000,
        steps=("filter", "compaction", "rlm"),
    )

    result = pipeline.run(request)

    assert calls == ["filter", "compaction", "rlm"]
    assert [message["content"] for message in result.messages] == [
        "hello",
        "filter",
        "compaction",
        "rlm",
    ]
    assert [trace.step_name for trace in result.step_traces] == [
        "filter",
        "compaction",
        "rlm",
    ]
    assert isinstance(result.messages, tuple)
    assert result.trace_id
    assert result.token_count > 0


def test_pipeline_skips_disabled_steps() -> None:
    calls: list[str] = []
    pipeline = ContextPipeline(
        tokenizer_model=None,
        compaction_model=None,
        memory_store=NoOpMemoryStore(),
        steps=(
            RecorderStep("filter", "filter", calls),
            RecorderStep("compaction", "compaction", calls),
            RecorderStep("rlm", "rlm", calls),
        ),
    )
    request = ContextRequest(
        messages=({"role": "user", "content": "hello"},),
        user_id="user",
        session_id="session",
        token_budget=1000,
        steps=("filter",),
    )

    pipeline.run(request)

    assert calls == ["filter"]


def test_pipeline_does_not_mutate_input_messages() -> None:
    original = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
    snapshot = [dict(message) for message in original]
    pipeline = ContextPipeline(
        tokenizer_model=None,
        compaction_model=None,
        memory_store=NoOpMemoryStore(),
    )
    request = ContextRequest(
        messages=tuple(original),
        user_id=None,
        session_id=None,
        token_budget=None,
        steps=("filter",),
    )

    pipeline.run(request)

    assert original == snapshot


def test_pipeline_result_is_independent_from_input_dicts() -> None:
    original = [{"role": "user", "content": "hello"}]
    pipeline = ContextPipeline(
        tokenizer_model=None,
        compaction_model=None,
        memory_store=NoOpMemoryStore(),
    )
    request = ContextRequest(
        messages=tuple(original),
        user_id=None,
        session_id=None,
        token_budget=None,
        steps=(),
    )

    result = pipeline.run(request)
    result.messages[0]["content"] = "changed"

    assert original == [{"role": "user", "content": "hello"}]


def test_pipeline_raises_when_final_budget_is_exceeded() -> None:
    pipeline = ContextPipeline(
        tokenizer_model=None,
        compaction_model=None,
        memory_store=NoOpMemoryStore(),
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
    pipeline = ContextPipeline(
        tokenizer_model=None,
        compaction_model=None,
        memory_store=NoOpMemoryStore(),
    )
    request = ContextRequest(
        messages=({"role": "user", "content": "hello"},),
        user_id=None,
        session_id=None,
        token_budget=None,
        steps=("unknown",),
    )

    with pytest.raises(TokentrimError) as exc_info:
        pipeline.run(request)

    assert "Unknown context steps requested" in str(exc_info.value)
