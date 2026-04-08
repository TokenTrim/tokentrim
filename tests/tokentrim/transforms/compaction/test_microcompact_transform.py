from __future__ import annotations

from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms.compaction.config import (
    BALANCED_CONTEXT_EDIT_CONFIG,
    BALANCED_MICROCOMPACT_CONFIG,
    AGGRESSIVE_CONTEXT_EDIT_CONFIG,
    get_context_edit_config,
    get_microcompact_config,
)
from tokentrim.transforms.compaction.microcompact import (
    MicrocompactConfig,
    MicrocompactMessages,
    MicrocompactOrchestrator,
)
from tokentrim.transforms.compaction.types import CompactionMetrics, MessageGroup, MicrocompactPlan
from tokentrim.types.state import PipelineState


def _request() -> ContextRequest:
    return ContextRequest(
        messages=tuple(),
        user_id=None,
        session_id=None,
        token_budget=None,
        steps=(MicrocompactMessages(),),
    )


def test_microcompact_exposes_standalone_transform() -> None:
    step = MicrocompactMessages()
    result = step.run(
        PipelineState(context=[{"role": "user", "content": "short"}], tools=[]),
        _request(),
    )

    assert result.context == [{"role": "user", "content": "short"}]


def test_microcompact_uses_age_based_policy_to_keep_recent_groups_verbose() -> None:
    orchestrator = MicrocompactOrchestrator(
        config=MicrocompactConfig(
            min_content_chars=40,
            recent_groups_to_keep=1,
            mature_group_age=3,
            use_salience_scoring=False,  # Test pure age-based policy
        )
    )
    messages = [
        {
            "role": "assistant",
            "content": "$ pytest tests/unit.py\nTraceback (most recent call last):\nFileNotFoundError: x\n"
            + ("older log\n" * 10),
        },
        {"role": "user", "content": "[exit_code] 1\nstderr: failed"},
        {"role": "user", "content": "normal discussion " * 8},
        {"role": "assistant", "content": "response " * 8},
        {"role": "user", "content": "recent request " * 8},
        {"role": "assistant", "content": "recent response " * 8},
    ]

    result = orchestrator.apply(messages)

    assert result[0]["role"] == "system"
    assert "[microcompact]" in result[0]["content"]
    assert "age=mature" in result[0]["content"]
    assert result[-2:] == messages[-2:]


def test_microcompact_groups_tool_rounds_and_preserves_command_output_link() -> None:
    orchestrator = MicrocompactOrchestrator(
        config=MicrocompactConfig(
            min_content_chars=20,
            recent_groups_to_keep=0,
            use_salience_scoring=False,  # Test pure age-based policy
        )
    )
    messages = [
        {"role": "assistant", "content": "I ran:\n$ uv run pytest tests/test_cli.py"},
        {
            "role": "user",
            "content": "[TERMINAL]\nTraceback (most recent call last):\nValueError: boom\n[exit_code] 2",
        },
    ]

    result = orchestrator.apply(messages)

    assert result == [
        {
            "role": "system",
            "content": result[0]["content"],
        }
    ]
    assert "kind=tool_round" in result[0]["content"]
    assert "commands=uv run pytest tests/test_cli.py" in result[0]["content"]
    assert "exit_codes=2" in result[0]["content"]
    assert "ValueError: boom" in result[0]["content"]


def test_microcompact_collapses_old_dialogue_rounds_but_not_recent_ones() -> None:
    step = MicrocompactMessages(
        config=MicrocompactConfig(
            min_content_chars=40,
            recent_groups_to_keep=1,
            mature_group_age=2,
        )
    )
    messages = [
        {"role": "user", "content": "old request " * 10},
        {"role": "assistant", "content": "old response " * 10},
        {"role": "user", "content": "recent request " * 10},
        {"role": "assistant", "content": "recent response " * 10},
    ]

    result = step.run(PipelineState(context=messages, tools=[]), _request())

    assert result.context[0]["role"] == "system"
    assert "kind=dialogue_round" in result.context[0]["content"]
    assert result.context[-2:] == messages[-2:]


def test_microcompact_preserves_existing_system_and_summary_messages() -> None:
    orchestrator = MicrocompactOrchestrator(
        config=MicrocompactConfig(
            min_content_chars=20,
            recent_groups_to_keep=0,
            use_salience_scoring=False,  # Test pure age-based policy
        )
    )
    messages = [
        {"role": "system", "content": "System instruction"},
        {"role": "system", "content": "History only. prior summary"},
        {"role": "assistant", "content": "$ pytest tests/unit.py\n" + ("log line\n" * 12)},
        {"role": "user", "content": "[exit_code] 1\nValueError: boom\nstderr: failed"},
    ]

    result = orchestrator.apply(messages)

    assert result[0] == messages[0]
    assert result[1] == messages[1]
    assert result[2]["role"] == "system"
    assert "[microcompact]" in result[2]["content"]


def test_microcompact_becomes_more_aggressive_under_budget_pressure() -> None:
    orchestrator = MicrocompactOrchestrator(
        config=MicrocompactConfig(
            min_content_chars=400,
            aggressive_min_content_chars=40,
            recent_groups_to_keep=2,
            aggressive_recent_groups_to_keep=0,
        )
    )
    messages = [
        {
            "role": "assistant",
            "content": "$ pytest tests/unit.py\nFileNotFoundError: fixture\n" + ("log\n" * 20),
        },
        {"role": "user", "content": "[exit_code] 1"},
        {"role": "user", "content": "recent request " * 4},
        {"role": "assistant", "content": "recent response " * 4},
    ]

    normal = orchestrator.apply(messages, token_budget=10_000)
    pressured = orchestrator.apply(messages, token_budget=10)

    assert normal == messages
    assert pressured[0]["role"] == "system"
    assert "kind=tool_round" in pressured[0]["content"]


def test_microcompact_salience_scoring_preserves_high_value_content() -> None:
    """Salience scoring should protect messages with errors/constraints from compaction."""
    orchestrator_with_salience = MicrocompactOrchestrator(
        config=MicrocompactConfig(
            min_content_chars=20,
            recent_groups_to_keep=0,
            mature_group_age=1,
            use_salience_scoring=True,
            min_salience_to_protect=20,
        )
    )
    orchestrator_without_salience = MicrocompactOrchestrator(
        config=MicrocompactConfig(
            min_content_chars=20,
            recent_groups_to_keep=0,
            mature_group_age=1,
            use_salience_scoring=False,
        )
    )
    messages = [
        {
            "role": "assistant",
            "content": (
                "$ pytest tests/unit.py\n"
                "Traceback (most recent call last):\n"
                "FileNotFoundError: missing fixture\n"
                "Do not modify unrelated files\n"
                + ("padding\n" * 20)
            ),
        },
        {
            "role": "user",
            "content": (
                "[exit_code] 1\n"
                "stderr: failed\n"
                "stdout: rerun with --verbose\n"
                + ("padding\n" * 10)
            ),
        },
    ]

    result_with_salience = orchestrator_with_salience.apply(messages)
    result_without_salience = orchestrator_without_salience.apply(messages)

    assert result_with_salience == messages, "Salience should preserve high-value content"
    assert result_without_salience[0]["role"] == "system", "Without salience, old content should compact"
    assert "[microcompact]" in result_without_salience[0]["content"]


def test_microcompact_helper_branches_and_types() -> None:
    orchestrator = MicrocompactOrchestrator(
        config=MicrocompactConfig(
            min_content_chars=20,
            recent_groups_to_keep=0,
            recent_tool_groups_to_keep=1,
            mature_group_age=2,
            use_salience_scoring=False,
        )
    ).with_tokenizer("gpt-4o-mini")
    assert orchestrator.tokenizer_model == "gpt-4o-mini"
    assert orchestrator._resolve_pressure([], token_budget=None, pressure=None) == "normal"
    assert orchestrator._resolve_pressure([{"role": "user", "content": "x" * 20}], token_budget=1, pressure=None) == "high"
    assert orchestrator._resolve_pressure([], token_budget=1, pressure="normal") == "normal"

    assert not orchestrator._should_pair(
        {"role": "user", "content": "x"},
        {"role": "system", "content": "system"},
    )
    assert orchestrator._should_pair(
        {"role": "user", "content": "[exit_code] 1"},
        {"role": "assistant", "content": "reply"},
    )
    assert orchestrator._should_pair(
        {"role": "tool", "content": "a", "tool_call_id": "1"},
        {"role": "tool", "content": "b", "tool_call_id": "2"},
    )

    groups = orchestrator._build_groups(
        [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_12345678", "function": {"name": "search"}}]},
            {"role": "tool", "content": "", "tool_call_id": "call_12345678", "name": "search"},
            {"role": "user", "content": "older"},
            {"role": "assistant", "content": "reply"},
        ],
        pressure="normal",
    )
    assert groups[0].kind == "protected"
    assert groups[1].kind == "tool_round"


def test_microcompact_preserves_full_tool_call_ids() -> None:
    orchestrator = MicrocompactOrchestrator(
        config=MicrocompactConfig(
            min_content_chars=40,
            recent_groups_to_keep=0,
            mature_group_age=1,
            min_tokens_saved=0,
            use_salience_scoring=False,
        )
    )
    messages = [
        {
            "role": "assistant",
            "content": "$ pytest tests/unit.py\nFileNotFoundError: fixture\n" + ("log\n" * 20),
            "tool_calls": [{"id": "call_00001234", "function": {"name": "search"}}],
        },
        {
            "role": "user",
            "content": "[exit_code] 1\nstderr: failed\nstdout: rerun with --verbose",
            "tool_call_id": "call_00001234",
            "name": "search",
        },
    ]

    result = orchestrator.apply(messages)

    assert "search(id=call_00001234)" in result[0]["content"]
    assert "search→result(id=call_00001234)" in result[0]["content"]

    protected_group = MessageGroup(messages=({"role": "system", "content": "sys"},), kind="protected", age_band="old", token_count=1)
    assert not orchestrator._should_compact_group(protected_group, pressure="normal")

    salient_group = MessageGroup(
        messages=(
            {"role": "assistant", "content": "$ pytest\nFileNotFoundError: fixture\nDo not edit unrelated files"},
            {"role": "user", "content": "[exit_code] 1"},
        ),
        kind="tool_round",
        age_band="mature",
        token_count=50,
    )
    salient_orchestrator = MicrocompactOrchestrator(
        config=MicrocompactConfig(
            min_content_chars=20,
            recent_groups_to_keep=0,
            mature_group_age=1,
            use_salience_scoring=True,
            min_salience_to_protect=20,
        )
    )
    assert not salient_orchestrator._should_compact_group(salient_group, pressure="normal")

    compacted = orchestrator._compact_group(
        MessageGroup(
            messages=(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "$ pytest tests/unit.py\nTraceback (most recent call last):"},
                        {"type": "image_url", "image_url": {"url": "https://cdn.example.com/debug.png"}},
                    ],
                    "tool_calls": [{"id": "call_abcdef123456", "function": {"name": "search"}}],
                },
                {
                    "role": "tool",
                    "content": "[exit_code] 1\nstderr: failed",
                    "tool_call_id": "call_abcdef123456",
                    "name": "search",
                },
            ),
            kind="tool_round",
            age_band="mature",
            token_count=100,
        )
    )
    assert "images=" in compacted["content"]
    assert "tools=" in compacted["content"]

    assert orchestrator._extract_tool_call_info(
        (
            {"role": "assistant", "content": "", "tool_calls": [{"id": "", "function": {"name": "search"}}]},
            {"role": "tool", "content": "", "tool_call_id": "", "name": "search"},
        )
    ) == ["search", "search→result"]

    assert orchestrator._extract_text_snippet(
        (
            {"role": "user", "content": "\n[meta]\nactual line\n"},
        )
    ) == "actual line"
    assert orchestrator._extract_text_snippet(({"role": "user", "content": "\n[meta]\n$ cmd\n"},)) == ""
    assert orchestrator._dedupe(["", "a", "a", "b"]) == ["a", "b"]
    assert orchestrator._normalize_line("  value., ") == "value"
    assert not orchestrator._looks_like_tool_or_terminal_content("plain text")
    assert orchestrator._is_protected_message({"role": "assistant", "content": "[microcompact] prior"})
    assert orchestrator._is_protected_message({"role": "assistant", "content": "History only. prior"})

    step = MicrocompactMessages()
    assert step.name == "microcompact"
    resolved = step.resolve(tokenizer_model="gpt-4o-mini")
    assert isinstance(resolved, MicrocompactMessages)
    assert resolved.tokenizer_model == "gpt-4o-mini"

    plan = MicrocompactPlan(messages=[], original_tokens=10, compacted_tokens=4, groups_seen=1, groups_compacted=1)
    assert plan.tokens_saved == 6

    metrics = CompactionMetrics(original_tokens=20, compacted_tokens=5, original_messages=4, compacted_messages=2)
    assert metrics.tokens_saved == 15
    assert metrics.compression_ratio == 0.75
    assert metrics.messages_removed == 2
    assert CompactionMetrics(original_tokens=0, compacted_tokens=0, original_messages=0, compacted_messages=0).compression_ratio == 0.0

    assert get_microcompact_config("balanced") is BALANCED_MICROCOMPACT_CONFIG
    assert get_context_edit_config("balanced") is BALANCED_CONTEXT_EDIT_CONFIG
    assert get_context_edit_config("aggressive") is AGGRESSIVE_CONTEXT_EDIT_CONFIG
