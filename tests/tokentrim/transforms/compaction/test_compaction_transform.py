from __future__ import annotations

import pytest

from tokentrim.errors.base import TokentrimError
from tokentrim.transforms.compaction.error import CompactionConfigurationError
from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms.compaction import CompactConversation
from tokentrim.types.state import PipelineState
from tokentrim.working_state import WorkingState, parse_working_state_message, render_working_state_message


def _messages(count: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"message {index} " + ("x" * 40)} for index in range(count)]


def _valid_summary() -> str:
    return (
        "Goal:\nfix issue\n\n"
        "Active State:\nin progress\n\n"
        "Critical Artifacts:\n./tokentrim/file.py\n\n"
        "Open Risks:\n- none\n\n"
        "Next Step:\nrerun pytest\n\n"
        "Older Context:\nolder details"
    )


def _summary_with_artifacts(*artifacts: str) -> str:
    body = " | ".join(artifacts) if artifacts else "./tokentrim/file.py"
    return (
        "Goal:\nfix\n\n"
        "Active State:\nprogress\n\n"
        f"Critical Artifacts:\n{body}\n\n"
        "Open Risks:\n- none\n\n"
        "Next Step:\nrerun\n\n"
        "Older Context:\nold"
    )


def _request(*, token_budget: int | None) -> ContextRequest:
    return ContextRequest(
        messages=tuple(),
        user_id=None,
        session_id=None,
        org_id=None,
        token_budget=token_budget,
        steps=(CompactConversation(),),
    )


def test_compaction_is_noop_when_under_budget() -> None:
    step = CompactConversation(model="compact-model")
    messages = _messages(7)

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=1_000))

    assert result.context == messages


def test_compaction_exposes_name_and_resolve_helpers() -> None:
    step = CompactConversation(model="compact-model")

    resolved = step.resolve(tokenizer_model="gpt-4o-mini")

    assert step.name == "compaction"
    assert isinstance(resolved, CompactConversation)
    assert resolved._tokenizer_model == "gpt-4o-mini"
    assert resolved.resolve_token_budget(123) == 123


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
    step = CompactConversation(model="compact-model", auto_budget=False)
    messages = _messages(8)

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=None))

    assert result.context == messages


def test_compaction_is_noop_when_message_count_is_at_threshold() -> None:
    step = CompactConversation(model="compact-model")
    messages = _messages(6)

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=101))

    assert result.context == messages


def test_compaction_preserves_recent_messages_and_injects_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model")
    messages = _messages(10)
    original_message_count = len(messages)

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: _valid_summary(),
    )
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: (
            1_000 if len(counted_messages) == original_message_count else 10
        ),
    )
    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=130))

    assert result.context[0]["role"] == "system"
    assert "History only." in result.context[0]["content"]
    assert "Goal:\nfix issue" in result.context[0]["content"]
    assert result.context[1:] == messages[-6:]


def test_compaction_respects_custom_keep_last(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model", keep_last=8)
    messages = _messages(12)
    original_message_count = len(messages)

    monkeypatch.setattr(
        CompactConversation,
        "_generate_summary",
        lambda self, messages: _summary_with_artifacts(
            "./tokentrim/README.md",
            "pytest tests/tokentrim/transforms/compaction",
        ),
    )
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: (
            1_000 if len(counted_messages) == original_message_count else 10
        ),
    )
    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=130))

    assert result.context[0]["role"] == "system"
    assert "Goal:\nfix" in result.context[0]["content"]
    assert result.context[1:] == messages[-8:]


def test_compaction_automatic_budget_reserves_output_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(
        model="gpt-4o-mini",
        context_window=100,
    )
    messages = _messages(8)

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: 90 if len(counted_messages) == len(messages) else 10,
    )
    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=None))

    assert result.context[0]["role"] == "system"
    assert "History only." in result.context[0]["content"]
    assert "summary" in result.context[0]["content"]
    assert result.context[1:] == messages[-6:]


def test_compaction_raises_when_model_is_missing_and_over_budget() -> None:
    step = CompactConversation(model=None)
    messages = _messages(8)

    with pytest.raises(TokentrimError) as exc_info:
        step.run(PipelineState(context=messages, tools=[]), _request(token_budget=5))

    assert "no compaction model" in str(exc_info.value)


def test_compaction_raises_for_invalid_configuration_before_running() -> None:
    state = PipelineState(context=_messages(8), tools=[])
    request = _request(token_budget=5)

    with pytest.raises(CompactionConfigurationError):
        CompactConversation(model="compact-model", keep_last=-1).run(state, request)

    with pytest.raises(CompactionConfigurationError):
        CompactConversation(model="compact-model", strategy="invalid").run(state, request)  # type: ignore[arg-type]


def test_compaction_resolve_token_budget_uses_output_and_buffer_headroom() -> None:
    step = CompactConversation(model="gpt-4o-mini", context_window=100)

    assert step.resolve_token_budget(None) == 57


def test_compaction_only_sends_older_messages_to_summarizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model")
    messages = _messages(10)
    captured = {}
    original_message_count = len(messages)

    def fake_generate_text(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _valid_summary()

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


def test_compaction_uses_structured_prompt_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model")
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


def test_compaction_allows_custom_instructions_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(
        model="compact-model",
        instructions="Preserve exact commands, file paths, and unresolved errors.",
    )
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
    assert prompt[0]["content"] == "Preserve exact commands, file paths, and unresolved errors."
    assert "Goal:" in prompt[1]["content"]
    assert "Critical Artifacts:" in prompt[1]["content"]


def test_compaction_can_trigger_automatically_from_known_model_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(
        model="gpt-4o-mini",
        context_window=220,
    )
    messages = _messages(10)

    monkeypatch.setattr(
        CompactConversation,
        "_generate_summary",
        lambda self, messages: _valid_summary(),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=None))

    assert result.context[0]["role"] == "system"
    assert "Goal:\nfix issue" in result.context[0]["content"]


def test_explicit_token_budget_overrides_auto_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(
        model="gpt-4o-mini",
        context_window=600,
    )
    messages = _messages(7)

    monkeypatch.setattr(
        CompactConversation,
        "_generate_summary",
        lambda self, messages: _summary_with_artifacts(
            "./tokentrim/tokentrim/transforms/compaction/transform.py",
            "./tokentrim/tests/tokentrim/transforms/compaction/test_compaction_transform.py",
            "./tokentrim/README.md",
        ),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=1_000))

    assert result.context == messages


def test_compaction_infers_auto_budget_from_model_name() -> None:
    step = CompactConversation(
        model="openai/mercury-2",
    )

    assert step._resolve_effective_token_budget(None) == 116_000


def test_compaction_chunks_oversized_summary_input(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="openai/mercury-2")
    messages = [{"role": "user", "content": "x" * 400} for _ in range(10)]
    original_message_count = len(messages)
    generated_chunks: list[int] = []

    def fake_generate(self, *, messages, template, preserved_artifacts):
        del self, template, preserved_artifacts
        generated_chunks.append(len(messages))
        return _valid_summary()

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.CompactionLLM.generate",
        fake_generate,
    )
    monkeypatch.setattr(
        CompactConversation,
        "_resolve_compactor_prompt_budget",
        lambda self: 500,
    )
    monkeypatch.setattr(
        CompactConversation,
        "_estimate_compactor_prompt_tokens",
        lambda self, messages, template: sum(
            len(str(message["content"])) for message in messages
        ),
    )
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: (
            1_000 if len(counted_messages) == original_message_count else 10
        ),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=130))

    assert generated_chunks
    assert 4 not in generated_chunks
    assert result.context[0]["role"] == "system"
    assert "Goal:\nfix issue" in result.context[0]["content"]


def test_compaction_returns_none_when_auto_budget_cannot_be_inferred() -> None:
    step = CompactConversation(model="unknown-model")

    assert step._infer_context_window_from_model() is None
    assert step._resolve_effective_context_window() is None
    assert step._resolve_effective_token_budget(None) is None


def test_compaction_microcompacts_bulky_terminal_output_before_summarization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model")
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
        return _valid_summary()

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)

    step.run(PipelineState(context=messages, tools=[]), _request(token_budget=101))

    prompt_content = captured_calls[0][1]["content"]
    assert "[microcompact]" in prompt_content
    assert "commands=pytest tests/tokentrim/transforms/compaction/test_compaction_transform.py" in prompt_content
    assert "errors=FileNotFoundError: missing fixture | stderr: build failed" in prompt_content
    assert "FileNotFoundError: missing fixture" in prompt_content
    assert "extra line extra line extra line" not in prompt_content


def test_compaction_returns_model_output_even_when_artifacts_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(
        model="compact-model",
    )
    messages = [
        {"role": "user", "content": "Run `pytest tests/tokentrim/transforms` from ./tokentrim."},
        {"role": "assistant", "content": "It failed with FileNotFoundError: missing fixture."},
        *_messages(8),
    ]
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: (
            "Goal:\ndebug the failure.\n"
            "Active State:\nin progress.\n"
            "Critical Artifacts:\n- none\n"
            "Open Risks:\n- none\n"
            "Next Step:\ncontinue debugging.\n"
            "Older Context:\nolder details."
        ),
    )
    original_message_count = len(messages)
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: (
            1_000 if len(counted_messages) == original_message_count else 10
        ),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=200))
    assert "Critical Artifacts:\n- none" in result.context[1]["content"]


def test_compaction_custom_instructions_use_single_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(
        model="compact-model",
        instructions="Focus on preserving shell commands, file paths, and unresolved errors.",
    )
    messages = [
        {"role": "user", "content": "Run `pytest tests/tokentrim/transforms` from ./tokentrim."},
        {"role": "assistant", "content": "It failed with FileNotFoundError: missing fixture."},
        *_messages(8),
    ]
    prompts: list[str] = []

    def fake_generate_text(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return (
            "Goal:\ndebug the failure.\n"
            "Active State:\nin progress.\n"
            "Critical Artifacts:\n- none\n"
            "Open Risks:\n- none\n"
            "Next Step:\ncontinue debugging.\n"
            "Older Context:\nolder details."
        )

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)
    original_message_count = len(messages)
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: (
            1_000 if len(counted_messages) == original_message_count else 10
        ),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=200))

    assert prompts == ["Focus on preserving shell commands, file paths, and unresolved errors."]
    assert "Goal:\ndebug the failure." in result.context[1]["content"]


def test_compaction_returns_raw_model_output_without_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model")
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
    assert "short summary without the important details" in result.context[1]["content"]


def test_compaction_returns_placeholder_when_model_output_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model")
    messages = [{"role": "user", "content": "alpha"}]

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "",
    )
    assert step._generate_summary(messages) == "- no summary returned"


def test_compaction_forwards_model_options(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(
        model="openai/mercury-2",
        keep_last=4,
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
        return _valid_summary()

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
    step = CompactConversation(model="compact-model")
    messages = _messages(8)

    def explode(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", explode)

    with pytest.raises(TokentrimError) as exc_info:
        step.run(PipelineState(context=messages, tools=[]), _request(token_budget=5))

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_compaction_reraises_configuration_errors_from_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model")
    messages = _messages(8)

    def explode(**kwargs):
        raise CompactionConfigurationError("bad setup")

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", explode)

    with pytest.raises(CompactionConfigurationError):
        step.run(PipelineState(context=messages, tools=[]), _request(token_budget=5))


def test_compaction_injects_working_state_before_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model")
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
        CompactConversation,
        "_generate_summary",
        lambda self, messages: _summary_with_artifacts(
            "./tokentrim/file_live.py",
            "pytest tests/case_live.py",
            "FileNotFoundError: fixture_live",
        ),
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
    step = CompactConversation(model="compact-model")
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
        CompactConversation,
        "_generate_summary",
        lambda self, messages: _summary_with_artifacts(
            "./tokentrim/file_16.py",
            "./tokentrim/file_17.py",
        ),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=100))

    assert (
        "Latest Command: pytest tests/tokentrim/transforms/compaction/test_compaction_transform.py"
        in result.context[0]["content"]
    )
    assert "Active Error: FileNotFoundError: missing fixture" in result.context[0]["content"]


def test_working_state_tracks_active_files(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model")
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
        CompactConversation,
        "_generate_summary",
        lambda self, messages: _summary_with_artifacts("Permission denied: .pytest_cache"),
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


def test_working_state_handles_multimodal_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model")
    messages = [
        {
            "role": "user",
            "content": "Do not widen scope.",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "$ pytest tests/case_live.py\n"
                        "Traceback (most recent call last):\n"
                        "FileNotFoundError: fixture_live\n"
                        "Inspect ./tokentrim/file_live.py\n"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/debug_live.png"},
                },
            ],
        },
        *_messages(7),
    ]

    monkeypatch.setattr(
        CompactConversation,
        "_generate_summary",
        lambda self, messages: _summary_with_artifacts(
            "./tokentrim/file_16.py",
            "./tokentrim/file_17.py",
        ),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=140))

    content = result.context[0]["content"]
    assert "Active Files: ./tokentrim/file_live.py" in content
    assert "Latest Command: pytest tests/case_live.py" in content
    assert "Active Error: FileNotFoundError: fixture_live" in content


def test_working_state_normalizes_duplicate_paths_with_trailing_punctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model")
    messages = [
        {"role": "user", "content": "Continue and preserve ./tokentrim/file_16.py."},
        {"role": "assistant", "content": "I will inspect ./tokentrim/file_16.py next."},
        {"role": "user", "content": "Then preserve ./tokentrim/file_17.py."},
        {"role": "assistant", "content": "Inspect ./tokentrim/file_17.py and rerun pytest."},
        *_messages(7),
    ]

    monkeypatch.setattr(
        CompactConversation,
        "_generate_summary",
        lambda self, messages: _summary_with_artifacts("Permission denied: .pytest_cache"),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=120))

    content = result.context[0]["content"]
    active_files_line = next(line for line in content.splitlines() if line.startswith("Active Files:"))
    assert "./tokentrim/file_17.py." not in active_files_line
    assert "./tokentrim/file_16.py." not in active_files_line
    assert active_files_line.count("./tokentrim/file_17.py") == 1
    assert active_files_line.count("./tokentrim/file_16.py") == 1


def test_working_state_replaces_stale_error_with_newer_active_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model")
    messages = [
        {"role": "user", "content": "Run the compaction tests."},
        {"role": "assistant", "content": "FileNotFoundError: old missing fixture"},
        {"role": "assistant", "content": "Fixed that; tests passed locally."},
        {"role": "assistant", "content": "Permission denied: .pytest_cache"},
        *_messages(7),
    ]

    monkeypatch.setattr(
        CompactConversation,
        "_generate_summary",
        lambda self, messages: _summary_with_artifacts("Permission denied: .pytest_cache"),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=100))

    content = result.context[0]["content"]
    assert "Active Error: Permission denied: .pytest_cache" in content
    assert "FileNotFoundError: old missing fixture" not in content


def test_working_state_omits_empty_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CompactConversation(model="compact-model")
    messages = [
        {"role": "user", "content": "Please review the proposal and avoid changing scope."},
        {"role": "assistant", "content": "I'll review the proposal next."},
        *_messages(8),
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: _valid_summary(),
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
    step = CompactConversation(model="compact-model")
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
        return _summary_with_artifacts("./tokentrim/README.md")

    monkeypatch.setattr("tokentrim.transforms.compaction.transform.generate_text", fake_generate_text)

    step.run(PipelineState(context=messages, tools=[]), _request(token_budget=100))

    prompt = captured["messages"][1]["content"]
    assert "FileNotFoundError: old fixture" not in prompt
    assert "[exit_code] 1" not in prompt
    assert "Rerun the targeted tests and update ./tokentrim/README.md." in prompt


def test_compaction_minimal_strategy_keeps_resolved_terminal_noise() -> None:
    balanced_step = CompactConversation(model="compact-model")
    minimal_step = CompactConversation(model="compact-model", strategy="minimal")
    messages = [
        {
            "role": "assistant",
            "content": "$ pytest tests/unit.py\nTraceback (most recent call last):\nFileNotFoundError: old fixture",
        },
        {"role": "user", "content": "[exit_code] 1\nstderr: failed"},
        {"role": "assistant", "content": "Fixed that; tests passed locally."},
        {"role": "user", "content": "Rerun the targeted tests and update ./tokentrim/README.md."},
    ]

    balanced = balanced_step._apply_context_edit(messages)
    minimal = minimal_step._apply_context_edit(messages)

    assert not any("FileNotFoundError: old fixture" in message["content"] for message in balanced)
    assert any("FileNotFoundError: old fixture" in message["content"] for message in minimal)


def test_compaction_aggressive_strategy_microcompacts_more_than_minimal() -> None:
    aggressive_step = CompactConversation(model="compact-model", strategy="aggressive")
    minimal_step = CompactConversation(model="compact-model", strategy="minimal")
    messages = [
        {
            "role": "assistant",
            "content": "$ pytest tests/unit.py\nFileNotFoundError: fixture\n" + ("log\n" * 20),
        },
        {"role": "user", "content": "[exit_code] 1\nstderr: failed"},
        {"role": "user", "content": "recent request " * 4},
        {"role": "assistant", "content": "recent response " * 4},
    ]

    aggressive = aggressive_step._apply_microcompact(messages, pressure="high")
    minimal = minimal_step._apply_microcompact(messages, pressure="high")

    assert aggressive[0]["role"] == "system"
    assert "[microcompact]" in aggressive[0]["content"]
    assert minimal == messages


def test_compaction_preserves_unstructured_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CompactConversation(model="compact-model")
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
    assert "plain checkpoint without headings" in content


def test_compaction_helper_branches() -> None:
    step = CompactConversation(model="compact-model")
    llm = step._compaction_llm()
    assert llm.model == "compact-model"
    assert step._is_shell_command("python3 script.py")
    assert not step._is_shell_command("just some text")
    assert "python3 script.py" in step._extract_artifacts_from_content("python3 script.py\n")
    assert step._finalize_summary_output("  hi\r\n") == "hi"
    assert step._finalize_summary_output("  ") == "- no summary returned"
