from __future__ import annotations

import pytest

from tokentrim.errors.base import TokentrimError
from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms.compaction import CompactConversation
from tokentrim.types.state import PipelineState
from tokentrim.working_state import WorkingState, parse_working_state_message, render_working_state_message


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


def test_working_state_message_round_trips_through_typed_parser() -> None:
    state = WorkingState(
        goal="Update docs",
        current_step="Inspect failing test",
        active_files=("./tokentrim/README.md", "./tokentrim/tests/test_client.py"),
        latest_command="pytest tests/tokentrim/test_client.py",
        active_error="FileNotFoundError: missing fixture",
        constraints=("avoid scope changes", "do not remove tests"),
        next_step="Update fixture and rerun pytest.",
    )

    message = render_working_state_message(state)

    assert message is not None
    assert parse_working_state_message(message) == state


def test_compaction_is_noop_when_budget_is_none() -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None, auto_budget=False)
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
    original_message_count = len(messages)

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: (
            1_000 if len(counted_messages) == original_message_count else 10
        ),
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
    original_message_count = len(messages)

    def fake_generate_text(**kwargs):
        captured["messages"] = kwargs["messages"]
        return "summary"

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: (
            1_000 if len(counted_messages) == original_message_count else 10
        ),
    )

    step.run(PipelineState(context=messages, tools=[]), _request(token_budget=101))

    prompt = captured["messages"]
    assert "message 0" in prompt[1]["content"]
    assert "message 3" in prompt[1]["content"]
    assert "message 4" not in prompt[1]["content"]
    assert "message 9" not in prompt[1]["content"]


def test_compaction_uses_structured_prompt_family_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = _messages(10)
    captured = {}

    def fake_generate_text(**kwargs):
        captured["messages"] = kwargs["messages"]
        return (
            "Goal: keep debugging.\n"
            "Active State:\nin progress.\n"
            "Critical Artifacts:\npreserved.\n"
            "Open Risks:\nnone.\n"
            "Next Step:\ncontinue.\n"
            "Older Context:\nolder details."
        )

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)

    step.run(PipelineState(context=messages, tools=[]), _request(token_budget=101))

    prompt = captured["messages"]
    assert "compact engineering handoff" in prompt[0]["content"]
    assert "Goal:" in prompt[1]["content"]
    assert "Active State:" in prompt[1]["content"]
    assert "Older Context:" in prompt[1]["content"]


def test_compaction_can_trigger_automatically_from_known_model_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(
        model="gpt-4o-mini",
        tokenizer_model=None,
        context_window=220,
        reserved_output_tokens=50,
        auto_compact_buffer_tokens=50,
    )
    messages = _messages(10)

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=None))

    assert result.context[0]["role"] == "system"
    assert result.context[0]["content"].endswith("summary")


def test_explicit_token_budget_overrides_auto_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(
        model="gpt-4o-mini",
        tokenizer_model=None,
        context_window=600,
        reserved_output_tokens=100,
        auto_compact_buffer_tokens=50,
    )
    messages = _messages(7)

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=1_000))

    assert result.context == messages


def test_compaction_infers_auto_budget_from_model_name() -> None:
    step = CompactConversation(
        model="openai/mercury-2",
        tokenizer_model=None,
        reserved_output_tokens=8_000,
        auto_compact_buffer_tokens=4_000,
    )

    assert step._resolve_effective_token_budget(None) == 188_000


def test_compaction_microcompacts_bulky_terminal_output_before_summarization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    bulky_output = "\n".join(
        [
            "[COMMAND]",
            "$ pytest tests/tokentrim/transforms/compaction/test_compaction_transform.py",
            "[TERMINAL]",
            "Traceback (most recent call last):",
            "FileNotFoundError: missing fixture",
            "stderr: build failed",
            "stdout: rerun with --verbose",
            "[exit_code] 1",
            "extra line " * 30,
        ]
    )
    messages = [
        {"role": "user", "content": bulky_output},
        *_messages(9),
    ]
    captured_calls: list[list[dict[str, str]]] = []

    def fake_generate_text(**kwargs):
        captured_calls.append(kwargs["messages"])
        return (
            "Goal: debug tests.\n"
            "Current State: investigating.\n"
            "Important Facts: preserved.\n"
            "Commands / Paths / Identifiers: pytest tests/tokentrim/transforms/compaction/test_compaction_transform.py.\n"
            "Errors / Risks: FileNotFoundError: missing fixture.\n"
            "Next Steps: rerun."
        )

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)

    step.run(PipelineState(context=messages, tools=[]), _request(token_budget=101))

    prompt_content = captured_calls[0][1]["content"]
    assert "[microcompact]" in prompt_content
    assert "commands=pytest tests/tokentrim/transforms/compaction/test_compaction_transform.py" in prompt_content
    assert "errors=FileNotFoundError: missing fixture | stderr: build failed" in prompt_content
    assert "FileNotFoundError: missing fixture" in prompt_content
    assert "extra line extra line extra line" not in prompt_content


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
            "Goal:\ndebug the failure.\n"
            "Active State:\nin progress.\n"
            "Critical Artifacts:\n- none\n"
            "Open Risks:\n- none\n"
            "Next Step:\ncontinue debugging.\n"
            "Older Context:\nolder details.",
            "Goal:\ndebug the failure.\n"
            "Active State:\nin progress.\n"
            "Critical Artifacts:\npytest tests/tokentrim/transforms | ./tokentrim. | "
            "It failed with FileNotFoundError: missing fixture.\n"
            "Open Risks:\nIt failed with FileNotFoundError: missing fixture.\n"
            "Next Step:\nrerun pytest.\n"
            "Older Context:\nolder details.",
            "Goal:\ndebug the failure.\n"
            "Active State:\nin progress.\n"
            "Critical Artifacts:\npytest tests/tokentrim/transforms | ./tokentrim. | "
            "It failed with FileNotFoundError: missing fixture.\n"
            "Open Risks:\nIt failed with FileNotFoundError: missing fixture.\n"
            "Next Step:\nrerun pytest.\n"
            "Older Context:\nolder details.",
        ]
    )

    def fake_generate_text(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return next(outputs)

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)
    original_message_count = len(messages)
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: (
            1_000 if len(counted_messages) == original_message_count else 10
        ),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=200))

    assert len(prompts) >= 2
    assert "compact engineering handoff" in prompts[0]
    assert "Summarise the conversation history" in prompts[1]
    assert "pytest tests/tokentrim/transforms" in result.context[1]["content"]
    assert "FileNotFoundError: missing fixture" in result.context[1]["content"]


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

    assert "Critical Artifacts:" in result.context[1]["content"]
    assert "/tmp/build.log" in result.context[1]["content"]
    assert "git status" in result.context[1]["content"]
    assert "error: build failed" in result.context[1]["content"]
    assert "Older Context:" in result.context[1]["content"]


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
    original_message_count = len(messages)

    def fake_generate_text(**kwargs):
        captured.update(kwargs)
        return "summary"

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: (
            1_000 if len(counted_messages) == original_message_count else 10
        ),
    )

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


def test_compaction_injects_working_state_before_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = [
        {
            "role": "user",
            "content": (
                "Update ./tokentrim/README.md and rerun "
                "`pytest tests/tokentrim/transforms/compaction`."
            ),
        },
        {
            "role": "assistant",
            "content": "I'll update the docs and rerun the compaction tests next.",
        },
        *_messages(8),
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=130))

    assert result.context[0]["role"] == "system"
    assert "Working state only." in result.context[0]["content"]
    assert "Goal: Update ./tokentrim/README.md and rerun" in result.context[0]["content"]
    assert result.context[1]["role"] == "system"
    assert "History only." in result.context[1]["content"]


def test_working_state_preserves_latest_command_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = [
        {
            "role": "user",
            "content": "Run `pytest tests/tokentrim/transforms/compaction/test_compaction_transform.py`.",
        },
        {
            "role": "assistant",
            "content": "$ pytest tests/tokentrim/transforms/compaction/test_compaction_transform.py",
        },
        {
            "role": "user",
            "content": "Traceback (most recent call last):\nFileNotFoundError: missing fixture",
        },
        *_messages(7),
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=100))

    assert (
        "Latest Command: pytest tests/tokentrim/transforms/compaction/test_compaction_transform.py"
        in result.context[0]["content"]
    )
    assert "Active Error: FileNotFoundError: missing fixture" in result.context[0]["content"]


def test_working_state_tracks_active_files(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = [
        {
            "role": "user",
            "content": (
                "Check ./tokentrim/README.md and "
                "./tokentrim/tests/tokentrim/transforms/compaction/test_compaction_transform.py."
            ),
        },
        {
            "role": "assistant",
            "content": "I will inspect ./tokentrim/tokentrim/transforms/compaction/transform.py next.",
        },
        *_messages(8),
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=122))

    content = result.context[0]["content"]
    assert "Active Files:" in content
    assert "./tokentrim/tokentrim/transforms/compaction/transform.py" in content
    assert (
        "./tokentrim/tests/tokentrim/transforms/compaction/test_compaction_transform.py"
        in content
    )
    assert "./tokentrim/README.md" in content


def test_working_state_replaces_stale_error_with_newer_active_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = [
        {"role": "user", "content": "Run the compaction tests."},
        {"role": "assistant", "content": "FileNotFoundError: old missing fixture"},
        {"role": "assistant", "content": "Fixed that; tests passed locally."},
        {"role": "assistant", "content": "Permission denied: .pytest_cache"},
        *_messages(7),
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=100))

    content = result.context[0]["content"]
    assert "Active Error: Permission denied: .pytest_cache" in content
    assert "FileNotFoundError: old missing fixture" not in content


def test_working_state_omits_empty_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = [
        {"role": "user", "content": "Please review the proposal and avoid changing scope."},
        {"role": "assistant", "content": "I'll review the proposal next."},
        *_messages(8),
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=130))

    content = result.context[0]["content"]
    assert "Goal: Please review the proposal and avoid changing scope." in content
    assert "Constraints: Please review the proposal and avoid changing scope." in content
    assert "Latest Command:" not in content
    assert "Active Files:" not in content
    assert "Active Error:" not in content


def test_compaction_context_edit_drops_resolved_terminal_noise_before_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = [
        {
            "role": "assistant",
            "content": "$ pytest tests/unit.py\nTraceback (most recent call last):\nFileNotFoundError: old fixture",
        },
        {"role": "user", "content": "[exit_code] 1\nstderr: failed"},
        {"role": "assistant", "content": "Fixed that; tests passed locally."},
        {"role": "user", "content": "Rerun the targeted tests and update ./tokentrim/README.md."},
        *_messages(7),
    ]
    captured: dict[str, object] = {}

    def fake_generate_text(**kwargs):
        captured["messages"] = kwargs["messages"]
        return "summary"

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)

    step.run(PipelineState(context=messages, tools=[]), _request(token_budget=100))

    prompt = captured["messages"][1]["content"]
    assert "FileNotFoundError: old fixture" not in prompt
    assert "[exit_code] 1" not in prompt
    assert "Rerun the targeted tests and update ./tokentrim/README.md." in prompt


def test_compaction_normalizes_unstructured_model_output_to_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model", tokenizer_model=None)
    messages = _messages(10)
    original_message_count = len(messages)

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "plain checkpoint without headings",
    )
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: (
            1_000 if len(counted_messages) == original_message_count else 10
        ),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=130))

    content = result.context[0]["content"]
    assert "Goal:" in content
    assert "Active State:" in content
    assert "Critical Artifacts:" in content
    assert "Open Risks:" in content
    assert "Next Step:" in content
    assert "Older Context:" in content
    assert "plain checkpoint without headings" in content
