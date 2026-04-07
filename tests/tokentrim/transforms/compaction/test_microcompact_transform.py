from __future__ import annotations

from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms.compaction.microcompact import (
    MicrocompactConfig,
    MicrocompactMessages,
    MicrocompactOrchestrator,
)
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
        config=MicrocompactConfig(min_content_chars=20, recent_groups_to_keep=0)
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
        config=MicrocompactConfig(min_content_chars=20, recent_groups_to_keep=0)
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
