from __future__ import annotations

from tokentrim.memory import InMemoryMemoryStore, SessionMemoryWriter, write_session_memory


def test_session_memory_writer_creates_session_records() -> None:
    store = InMemoryMemoryStore()
    writer = SessionMemoryWriter(memory_store=store, session_id="sess_1")

    record = writer.write(
        content="Avoid destructive commands",
        kind="constraint",
        dedupe_key="avoid_destructive",
    )

    assert record.scope == "session"
    assert record.subject_id == "sess_1"
    assert record.kind == "constraint"


def test_write_session_memory_helper_uses_store_dedupe() -> None:
    store = InMemoryMemoryStore()

    first = write_session_memory(
        memory_store=store,
        session_id="sess_1",
        content="Use repo root",
        kind="active_state",
        dedupe_key="repo_root",
    )
    second = write_session_memory(
        memory_store=store,
        session_id="sess_1",
        content="Use repository root",
        kind="active_state",
        dedupe_key="repo_root",
    )

    assert first.memory_id == second.memory_id
    assert second.content == "Use repository root"
