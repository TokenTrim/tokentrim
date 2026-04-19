from __future__ import annotations

from pathlib import Path

import pytest

from tokentrim.memory import (
    FilesystemMemoryStore,
    InMemoryMemoryStore,
    MemoryQuery,
    MemoryRecord,
    MemoryWrite,
)


def test_memory_record_validates_required_fields() -> None:
    with pytest.raises(ValueError):
        MemoryRecord(
            memory_id="mem_1",
            scope="session",
            subject_id="sess_1",
            kind="constraint",
            content=" ",
        )


def test_memory_query_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        MemoryQuery(session_id="sess_1", k=0)


def test_in_memory_store_isolates_scope_and_subject() -> None:
    store = InMemoryMemoryStore()
    store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(content="Use repo root", kind="active_state", dedupe_key="repo"),
    )
    store.upsert_memory(
        MemoryRecord(
            memory_id="mem_user_1",
            scope="user",
            subject_id="user_1",
            kind="preference",
            content="Prefer concise answers",
        )
    )

    assert len(store.list_memories(scope="session", subject_id="sess_1")) == 1
    assert store.list_memories(scope="session", subject_id="sess_2") == ()
    assert len(store.list_memories(scope="user", subject_id="user_1")) == 1


def test_in_memory_store_dedupes_session_write_by_key() -> None:
    store = InMemoryMemoryStore()
    first = store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(
            content="Avoid destructive commands",
            kind="constraint",
            dedupe_key="avoid_destructive",
            salience=0.8,
        ),
    )
    second = store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(
            content="Avoid destructive commands unless requested",
            kind="constraint",
            dedupe_key="avoid_destructive",
            salience=0.9,
        ),
    )

    assert second.memory_id == first.memory_id
    assert second.content == "Avoid destructive commands unless requested"
    assert len(store.list_memories(scope="session", subject_id="sess_1")) == 1


def test_in_memory_query_respects_scope_weights() -> None:
    store = InMemoryMemoryStore()
    store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(content="Local session fact", kind="task_fact", salience=0.2),
    )
    store.upsert_memory(
        MemoryRecord(
            memory_id="mem_user_1",
            scope="user",
            subject_id="user_1",
            kind="task_fact",
            content="User fact",
            salience=0.9,
        )
    )

    result = store.query_memories(
        MemoryQuery(
            session_id="sess_1",
            user_id="user_1",
            k=2,
            scope_weights={"session": 5.0, "user": 1.0, "org": 0.1},
        )
    )

    assert [record.scope for record in result] == ["session", "user"]


def test_filesystem_store_round_trips_markdown_records(tmp_path: Path) -> None:
    store = FilesystemMemoryStore(root_dir=tmp_path / "memory")
    created = store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(
            content="Persist this fact",
            kind="task_fact",
            dedupe_key="fact_1",
            metadata={"source": "test", "title": "Persisted Fact", "description": "Useful task fact"},
        ),
    )

    listed = store.list_memories(scope="session", subject_id="sess_1")

    assert listed == (created,)
    assert (tmp_path / "memory" / "session" / "sess_1" / "persisted-fact.md").exists()
    entrypoint = tmp_path / "memory" / "session" / "sess_1" / "MEMORY.md"
    assert entrypoint.exists()
    entrypoint_text = entrypoint.read_text(encoding="utf-8")
    assert "Persisted Fact" in entrypoint_text
    assert "## Task Fact" in entrypoint_text


def test_filesystem_store_archives_and_deletes_records(tmp_path: Path) -> None:
    store = FilesystemMemoryStore(root_dir=tmp_path / "memory")
    created = store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(content="Persist this fact", kind="task_fact"),
    )

    store.archive_memory(memory_id=created.memory_id)
    archived = store.list_memories(scope="session", subject_id="sess_1")[0]
    assert archived.status == "archived"

    store.delete_memory(memory_id=created.memory_id)
    assert store.list_memories(scope="session", subject_id="sess_1") == ()


def test_filesystem_store_finds_memory_by_id_even_with_semantic_filename(tmp_path: Path) -> None:
    store = FilesystemMemoryStore(root_dir=tmp_path / "memory")
    created = store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(
            content="Persist this fact",
            kind="task_fact",
            dedupe_key="fact_1",
            metadata={"title": "Semantic file", "description": "Uses semantic filename"},
        ),
    )

    store.archive_memory(memory_id=created.memory_id)
    archived = store.list_memories(scope="session", subject_id="sess_1")[0]

    assert archived.memory_id == created.memory_id
    assert archived.status == "archived"
