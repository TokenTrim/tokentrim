from __future__ import annotations

from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms.filter import FilterMessages
from tokentrim.types.state import PipelineState


def _request() -> ContextRequest:
    return ContextRequest(
        messages=tuple(),
        user_id=None,
        session_id=None,
        token_budget=None,
        steps=(FilterMessages(),),
    )


def test_filter_removes_empty_messages() -> None:
    step = FilterMessages()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": ""},
    ]

    result = step.run(PipelineState(context=messages, tools=[]), _request())

    assert result.context == [{"role": "user", "content": "hello"}]


def test_filter_collapses_consecutive_duplicates() -> None:
    step = FilterMessages()
    messages = [
        {"role": "user", "content": "click"},
        {"role": "user", "content": "click"},
        {"role": "user", "content": "click"},
    ]

    result = step.run(PipelineState(context=messages, tools=[]), _request())

    assert result.context == [{"role": "user", "content": "click [repeated 3x]"}]


def test_filter_preserves_non_consecutive_duplicates() -> None:
    step = FilterMessages()
    messages = [
        {"role": "user", "content": "click"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "click"},
    ]

    result = step.run(PipelineState(context=messages, tools=[]), _request())

    assert result.context == messages
