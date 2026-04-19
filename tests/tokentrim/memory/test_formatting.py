from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
                metadata={"title": "Safety rule", "description": "Current session safety constraint"},
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
    assert "Safety rule" in content
    assert "Updated today." in content


def test_render_injected_memory_message_includes_freshness_warning_for_old_memory() -> None:
    old_timestamp = (datetime.now(UTC) - timedelta(days=5)).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    content = render_injected_memory_message(
        candidates=(
            MemoryRecord(
                memory_id="mem_1",
                scope="user",
                subject_id="user_1",
                kind="task_fact",
                content="The deploy flag used to be ENABLE_V2=true.",
                updated_at=old_timestamp,
                metadata={"title": "Old deploy flag", "description": "Historical deployment detail"},
            ),
        ),
        current_messages=[{"role": "user", "content": "how do I deploy?"}],
        token_budget=600,
        max_memory_tokens=300,
        tokenizer_model=None,
    )

    assert content is not None
    assert "Updated 5 days ago." in content
    assert "Re-check against current project state before using." in content
    assert "Verify code, files, flags, and current project state before relying on it." in content


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
