from __future__ import annotations

import pytest

from tokentrim.errors.base import TokentrimError
from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms.compaction import CompactConversation
from tokentrim.types.state import PipelineState


def _messages(count: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"message {index} " + ("x" * 40)} for index in range(count)]


def _request(*, token_budget: int | None) -> ContextRequest:
    return ContextRequest(
        messages=tuple(),
        user_id=None,
        session_id=None,
        token_budget=token_budget,
        steps=(CompactConversation(),),
    )


def test_compaction_is_noop_when_under_budget() -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = _messages(7)

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=1_000))

    assert result.context == messages


def test_compaction_is_noop_when_budget_is_none() -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = _messages(8)

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=None))

    assert result.context == messages


def test_compaction_is_noop_when_message_count_is_at_threshold() -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = _messages(6)

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=101))

    assert result.context == messages


def test_compaction_preserves_recent_messages_and_injects_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = _messages(10)

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )
    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=130))

    assert result.context[0]["role"] == "system"
    assert "History only." in result.context[0]["content"]
    assert result.context[0]["content"].endswith("summary")
    assert result.context[1:] == messages[-6:]


def test_compaction_respects_custom_keep_last(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model", keep_last=8, tokenizer_model=None)
    messages = _messages(12)

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )
    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=130))

    assert result.context[0]["role"] == "system"
    assert result.context[0]["content"].endswith("summary")
    assert result.context[1:] == messages[-8:]


def test_compaction_raises_when_model_is_missing_and_over_budget() -> None:
    step = CompactConversation(model=None, tokenizer_model=None)
    messages = _messages(8)

    with pytest.raises(TokentrimError) as exc_info:
        step.run(PipelineState(context=messages, tools=[]), _request(token_budget=5))

    assert "no compaction model" in str(exc_info.value)


def test_compaction_only_sends_older_messages_to_summarizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = _messages(10)
    captured = {}

    def fake_generate_text(**kwargs):
        captured["messages"] = kwargs["messages"]
        return "summary"

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)

    step.run(PipelineState(context=messages, tools=[]), _request(token_budget=101))

    prompt = captured["messages"]
    assert "message 0" in prompt[1]["content"]
    assert "message 3" in prompt[1]["content"]
    assert "message 4" not in prompt[1]["content"]
    assert "message 9" not in prompt[1]["content"]


def test_compaction_retries_with_second_prompt_family_when_first_output_drops_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = [
        {"role": "user", "content": "Run `pytest tests/tokentrim/transforms` from ./tokentrim."},
        {"role": "assistant", "content": "It failed with FileNotFoundError: missing fixture."},
        *_messages(8),
    ]
    prompts: list[str] = []
    outputs = iter(
        [
            "Goal: debug the failure.",
            "Goal: debug the failure. Facts: use pytest tests/tokentrim/transforms from ./tokentrim. "
            "Risk: FileNotFoundError: missing fixture.",
        ]
    )

    def fake_generate_text(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return next(outputs)

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=132))

    assert len(prompts) == 2
    assert "Summarise the conversation history" in prompts[0]
    assert "Rewrite the conversation as terse handoff notes" in prompts[1]
    assert "pytest tests/tokentrim/transforms" in result.context[0]["content"]
    assert "FileNotFoundError: missing fixture" in result.context[0]["content"]


def test_compaction_uses_local_fallback_when_model_outputs_fail_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = [
        {"role": "user", "content": "Check /tmp/build.log and run `git status`."},
        {"role": "assistant", "content": "error: build failed"},
        *_messages(8),
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "short summary without the important details",
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=73))

    assert "Preserve exactly:" in result.context[0]["content"]
    assert "/tmp/build.log" in result.context[0]["content"]
    assert "git status" in result.context[0]["content"]
    assert "error: build failed" in result.context[0]["content"]


def test_compaction_forwards_model_options(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(
        model="openai/mercury-2",
        keep_last=4,
        tokenizer_model="gpt-4o-mini",
        model_options={
            "api_base": "https://api.inceptionlabs.ai/v1",
            "api_key": "ice-key",
        },
    )
    messages = _messages(8)
    captured: dict[str, object] = {}

    def fake_generate_text(**kwargs):
        captured.update(kwargs)
        return "summary"

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=74))

    assert result.context[1:] == messages[-4:]
    assert captured["model"] == "openai/mercury-2"
    assert captured["completion_options"] == {
        "api_base": "https://api.inceptionlabs.ai/v1",
        "api_key": "ice-key",
    }


def test_compaction_wraps_unexpected_generation_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = _messages(8)

    def explode(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", explode)

    with pytest.raises(TokentrimError) as exc_info:
        step.run(PipelineState(context=messages, tools=[]), _request(token_budget=5))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
