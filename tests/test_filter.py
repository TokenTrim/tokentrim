from __future__ import annotations

from tokentrim.context.request import ContextRequest
from tokentrim.context.filter import FilterStep


def _request() -> ContextRequest:
    return ContextRequest(
        messages=tuple(),
        user_id=None,
        session_id=None,
        token_budget=None,
        steps=("filter",),
    )


def test_filter_removes_empty_messages() -> None:
    step = FilterStep()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": ""},
    ]

    result = step.run(messages, _request())

    assert result == [{"role": "user", "content": "hello"}]


def test_filter_collapses_consecutive_duplicates() -> None:
    step = FilterStep()
    messages = [
        {"role": "user", "content": "click"},
        {"role": "user", "content": "click"},
        {"role": "user", "content": "click"},
    ]

    result = step.run(messages, _request())

    assert result == [{"role": "user", "content": "click [repeated 3x]"}]


def test_filter_preserves_non_consecutive_duplicates() -> None:
    step = FilterStep()
    messages = [
        {"role": "user", "content": "click"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "click"},
    ]

    result = step.run(messages, _request())

    assert result == messages
