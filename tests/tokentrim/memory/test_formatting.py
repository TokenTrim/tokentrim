from __future__ import annotations

from tokentrim.memory import MemoryRecord, render_injected_memory_message


def test_render_injected_memory_message_formats_scope_and_kind() -> None:
    content = render_injected_memory_message(
        candidates=(
            MemoryRecord(
                memory_id="mem_1",
                scope="session",
                subject_id="sess_1",
                kind="constraint",
                content="Avoid destructive commands",
            ),
        ),
        current_messages=[{"role": "user", "content": "debug this"}],
        token_budget=500,
        max_memory_tokens=200,
        tokenizer_model=None,
    )

    assert content is not None
    assert "Injected memory:" in content
    assert "[session/constraint]" in content


def test_render_injected_memory_message_returns_none_when_budget_too_small() -> None:
    content = render_injected_memory_message(
        candidates=(
            MemoryRecord(
                memory_id="mem_1",
                scope="session",
                subject_id="sess_1",
                kind="task_fact",
                content="x" * 500,
            ),
        ),
        current_messages=[{"role": "user", "content": "debug this"}],
        token_budget=10,
        max_memory_tokens=10,
        tokenizer_model=None,
    )

    assert content is None
