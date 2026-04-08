from __future__ import annotations

from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms.compaction import CompactConversation
from tokentrim.transforms.compaction.context_edit import ContextEditor
from tokentrim.transforms.compaction.microcompact import MicrocompactOrchestrator
from tokentrim.types.state import PipelineState


def _request(*, token_budget: int | None) -> ContextRequest:
    return ContextRequest(
        messages=tuple(),
        user_id=None,
        session_id=None,
        token_budget=token_budget,
        steps=(CompactConversation(),),
    )


def test_context_edit_handles_long_messy_conversation_without_losing_constraints() -> None:
    editor = ContextEditor()
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "System instruction: stay scoped to compaction."},
        {"role": "user", "content": "Do not remove tests or widen scope."},
    ]

    for index in range(40):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": (
                        f"$ pytest tests/unit_{index}.py\n"
                        "Traceback (most recent call last):\n"
                        f"FileNotFoundError: fixture_{index}\n"
                    ),
                },
                {"role": "user", "content": "[exit_code] 1\nstderr: failed"},
                {"role": "assistant", "content": f"Fixed that; tests passed locally for batch {index}."},
                {"role": "assistant", "content": f"I'll inspect batch {index + 1} next."},
            ]
        )

    messages.extend(
        [
            {
                "role": "assistant",
                "content": "$ pytest tests/final.py\nTraceback (most recent call last):\nPermission denied: .pytest_cache",
            },
            {"role": "user", "content": "[exit_code] 1\nstderr: still failing"},
            {"role": "assistant", "content": "I am investigating the remaining failure."},
        ]
    )

    result = editor.edit(messages)
    content = "\n".join(str(message["content"]) for message in result.messages)

    assert "Do not remove tests or widen scope." in content
    assert "Permission denied: .pytest_cache" in content
    assert "FileNotFoundError: fixture_0" not in content
    assert result.stats.removed_messages > 0
    assert result.stats.removed_resolved_errors > 0


def test_microcompact_handles_large_mixed_conversation_and_preserves_key_refs() -> None:
    orchestrator = MicrocompactOrchestrator()
    messages: list[dict[str, object]] = [{"role": "system", "content": "System anchor"}]

    for index in range(24):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"$ pytest tests/case_{index}.py\n"
                                "Traceback (most recent call last):\n"
                                f"FileNotFoundError: missing_{index}\n"
                                + ("verbose log line\n" * 12)
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"https://example.com/debug_{index}.png"},
                        },
                    ],
                    "tool_calls": [
                        {
                            "id": f"call_{index:04d}",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "[exit_code] 1\nstderr: failed\nstdout: rerun with --verbose",
                    "tool_call_id": f"call_{index:04d}",
                    "name": "search",
                },
            ]
        )

    plan = orchestrator.plan(messages, token_budget=200)
    compacted_content = "\n".join(str(message["content"]) for message in plan.messages)

    assert plan.groups_compacted > 0
    assert plan.tokens_saved > 0
    assert "[image: debug_0.png]" in compacted_content
    assert "tools=search(id=call_000)" in compacted_content or "tools=search(id=call_0000)" in compacted_content
    assert "FileNotFoundError: missing_0" in compacted_content


def test_compaction_transform_handles_very_long_difficult_conversation(
    monkeypatch,
) -> None:
    step = CompactConversation(model="compact-model", keep_last=8, strategy="balanced")
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "System instruction: stay scoped to compaction."},
        {"role": "user", "content": "Do not remove tests. Preserve commands, paths, and unresolved errors."},
    ]

    for index in range(30):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": (
                        f"$ pytest tests/case_{index}.py\n"
                        "Traceback (most recent call last):\n"
                        f"FileNotFoundError: fixture_{index}\n"
                        f"Inspect ./tokentrim/module_{index}.py\n"
                        + ("very noisy output\n" * 10)
                    ),
                },
                {"role": "user", "content": "[exit_code] 1\nstderr: failed"},
                {"role": "assistant", "content": f"I'll fix fixture_{index} and then rerun pytest."},
                {"role": "user", "content": f"Continue with case {index} after preserving ./tokentrim/module_{index}.py."},
            ]
        )

    messages.extend(
        [
            {"role": "assistant", "content": "Latest status: most cases are resolved."},
            {"role": "user", "content": "Keep the last few turns verbatim."},
        ]
    )

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: (
            "Goal:\nFinish the compaction work.\n\n"
            "Active State:\nInvestigating the long-running test fixes.\n\n"
            "Critical Artifacts:\npytest | ./tokentrim/module_29.py | FileNotFoundError: fixture_29\n\n"
            "Open Risks:\nPermission denied: .pytest_cache\n\n"
            "Next Step:\nRerun the targeted tests.\n\n"
            "Older Context:\nEarlier noisy tool rounds were compacted."
        ),
    )

    result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=2_400))

    assert result.context[0]["role"] == "system"
    assert result.context[1]["role"] == "system"
    assert "History only." in str(result.context[1]["content"])
    assert "Critical Artifacts:" in str(result.context[1]["content"])
    assert result.context[-8:] == messages[-8:]
