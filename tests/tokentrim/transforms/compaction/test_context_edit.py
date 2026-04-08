from __future__ import annotations

from tokentrim.transforms.compaction.config import ContextEditConfig
from tokentrim.transforms.compaction.context_edit import ContextEditor
from tokentrim.transforms.compaction.types import ContextEditMessageGroup


def test_context_editor_drops_old_resolved_tool_rounds() -> None:
    editor = ContextEditor()
    messages = [
        {
            "role": "assistant",
            "content": "$ pytest tests/unit.py\nTraceback (most recent call last):\nFileNotFoundError: old fixture",
        },
        {"role": "user", "content": "[exit_code] 1\nstderr: failed"},
        {"role": "assistant", "content": "Fixed that; tests passed locally."},
        {"role": "user", "content": "Continue with the next check."},
    ]

    result = editor.apply(messages)

    content = "\n".join(message["content"] for message in result)
    assert "FileNotFoundError: old fixture" not in content
    assert "[exit_code] 1" not in content
    assert "Fixed that; tests passed locally." in content


def test_context_editor_keeps_latest_unresolved_tool_round() -> None:
    editor = ContextEditor()
    messages = [
        {"role": "assistant", "content": "$ rg TODO tokentrim\n[exit_code] 0"},
        {"role": "user", "content": "stdout: found matches"},
        {
            "role": "assistant",
            "content": "$ pytest tests/unit.py\nTraceback (most recent call last):\nPermission denied: cache",
        },
        {"role": "user", "content": "[exit_code] 1"},
    ]

    result = editor.apply(messages)

    content = "\n".join(message["content"] for message in result)
    assert "$ pytest tests/unit.py" in content
    assert "Permission denied: cache" in content
    assert "$ rg TODO tokentrim" not in content


def test_context_editor_collapses_older_redundant_assistant_plans() -> None:
    editor = ContextEditor()
    messages = [
        {"role": "assistant", "content": "I'll inspect the failure first."},
        {"role": "user", "content": "Do not use heredocs."},
        {"role": "assistant", "content": "Next I'll update the tests."},
        {"role": "assistant", "content": "I'll rerun pytest after that."},
    ]

    result = editor.apply(messages)

    assistant_messages = [message["content"] for message in result if message["role"] == "assistant"]
    assert "I'll inspect the failure first." not in assistant_messages
    assert "Next I'll update the tests." not in assistant_messages
    assert "I'll rerun pytest after that." in assistant_messages


def test_context_editor_preserves_explicit_constraints() -> None:
    editor = ContextEditor()
    messages = [
        {"role": "user", "content": "Do not use heredocs."},
        {"role": "assistant", "content": "I'll inspect the repo next."},
        {"role": "assistant", "content": "I'll update README after that."},
    ]

    result = editor.apply(messages)

    assert any(message["content"] == "Do not use heredocs." for message in result)


def test_context_editor_keeps_old_unresolved_error_when_recent_turn_is_fluff() -> None:
    editor = ContextEditor()
    messages = [
        {
            "role": "assistant",
            "content": "$ pytest tests/unit.py\nTraceback (most recent call last):\nFileNotFoundError: missing fixture",
        },
        {"role": "user", "content": "[exit_code] 1"},
        {"role": "assistant", "content": "I am still thinking about the next step."},
        {"role": "user", "content": "continue"},
    ]

    result = editor.apply(messages)
    content = "\n".join(message["content"] for message in result)

    assert "FileNotFoundError: missing fixture" in content


def test_context_editor_private_branches_for_protection_pairing_and_artifacts() -> None:
    editor = ContextEditor(config=ContextEditConfig(collapse_repeated_assistant_plans=True))

    protected_group = ContextEditMessageGroup(
        messages=({"role": "system", "content": "system"},),
        kind="protected",
    )
    assert editor._drop_reason(
        protected_group,
        group_index=0,
        latest_tool_index=None,
        saw_later_success=False,
        kept_recent_plan=False,
    ) is None

    tool_call_group = ContextEditMessageGroup(
        messages=(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}],
            },
        ),
        kind="message",
    )
    assert editor._drop_reason(
        tool_call_group,
        group_index=0,
        latest_tool_index=None,
        saw_later_success=False,
        kept_recent_plan=False,
    ) is None

    plan_group = ContextEditMessageGroup(
        messages=({"role": "assistant", "content": "I'll inspect this next."},),
        kind="assistant_plan",
    )
    assert editor._drop_reason(
        plan_group,
        group_index=0,
        latest_tool_index=None,
        saw_later_success=False,
        kept_recent_plan=True,
    ) == "assistant_plan"

    assert editor._build_groups(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "continue"},
        ]
    )[0].kind == "protected"

    assert editor._should_pair(
        {"role": "user", "content": "[exit_code] 1"},
        {"role": "assistant", "content": "reply"},
    )
    assert not editor._should_pair(
        {"role": "assistant", "content": "plan"},
        {"role": "system", "content": "system"},
    )
