from __future__ import annotations

import pytest

from tokentrim.context.compaction import CompactionStep
from tokentrim.context.request import ContextRequest
from tokentrim.errors.base import TokentrimError


def _messages(count: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"message {index} " + ("x" * 40)} for index in range(count)]


def _request(*, token_budget: int | None) -> ContextRequest:
    return ContextRequest(
        messages=tuple(),
        user_id=None,
        session_id=None,
        token_budget=token_budget,
        steps=("compaction",),
    )


def test_compaction_is_noop_when_under_budget() -> None:
    step = CompactionStep(model="compact-model", tokenizer_model=None)
    messages = _messages(7)

    result = step.run(messages, _request(token_budget=1_000))

    assert result == messages


def test_compaction_is_noop_when_budget_is_none() -> None:
    step = CompactionStep(model="compact-model", tokenizer_model=None)
    messages = _messages(8)

    result = step.run(messages, _request(token_budget=None))

    assert result == messages


def test_compaction_is_noop_when_message_count_is_at_threshold() -> None:
    step = CompactionStep(model="compact-model", tokenizer_model=None)
    messages = _messages(6)

    result = step.run(messages, _request(token_budget=5))

    assert result == messages


def test_compaction_preserves_recent_messages_and_injects_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactionStep(model="compact-model", tokenizer_model=None)
    messages = _messages(10)

    monkeypatch.setattr(
        "tokentrim.context.compaction.generate_text",
        lambda **kwargs: "summary",
    )
    result = step.run(messages, _request(token_budget=5))

    assert result[0] == {"role": "system", "content": "summary"}
    assert result[1:] == messages[-6:]


def test_compaction_raises_when_model_is_missing_and_over_budget() -> None:
    step = CompactionStep(model=None, tokenizer_model=None)
    messages = _messages(8)

    with pytest.raises(TokentrimError) as exc_info:
        step.run(messages, _request(token_budget=5))

    assert "no compaction model" in str(exc_info.value)


def test_compaction_only_sends_older_messages_to_summarizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactionStep(model="compact-model", tokenizer_model=None)
    messages = _messages(10)
    captured = {}

    def fake_generate_text(**kwargs):
        captured["messages"] = kwargs["messages"]
        return "summary"

    monkeypatch.setattr("tokentrim.context.compaction.generate_text", fake_generate_text)

    step.run(messages, _request(token_budget=5))

    prompt = captured["messages"]
    assert "message 0" in prompt[1]["content"]
    assert "message 3" in prompt[1]["content"]
    assert "message 4" not in prompt[1]["content"]
    assert "message 9" not in prompt[1]["content"]


def test_compaction_wraps_unexpected_generation_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactionStep(model="compact-model", tokenizer_model=None)
    messages = _messages(8)

    def explode(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("tokentrim.context.compaction.generate_text", explode)

    with pytest.raises(TokentrimError) as exc_info:
        step.run(messages, _request(token_budget=5))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
