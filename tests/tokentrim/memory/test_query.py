from __future__ import annotations

from tokentrim.memory import (
    MemoryQuery,
    MemoryRecord,
    memory_canonical_key,
    score_memory_record,
    select_memories,
    tokenize_text,
)


def test_tokenize_text_extracts_search_tokens() -> None:
    tokens = tokenize_text("Debug repo command failure in ./src")

    assert {"debug", "repo", "command", "failure", "./src"} <= tokens


def test_score_memory_record_penalizes_non_active_memories() -> None:
    active = MemoryRecord(
        memory_id="mem_active",
        scope="session",
        subject_id="sess_1",
        kind="task_fact",
        content="Debug repo command failures from the root",
        status="active",
    )
    archived = MemoryRecord(
        memory_id="mem_archived",
        scope="session",
        subject_id="sess_1",
        kind="task_fact",
        content="Debug repo command failures from the root",
        status="archived",
    )

    active_score = score_memory_record(
        active,
        scope_weights={"session": 3.0},
        query_tokens={"debug"},
    )
    archived_score = score_memory_record(
        archived,
        scope_weights={"session": 3.0},
        query_tokens={"debug"},
    )

    assert active_score > archived_score


def test_select_memories_ranks_by_scope_and_overlap() -> None:
    records = [
        MemoryRecord(
            memory_id="mem_user",
            scope="user",
            subject_id="user_1",
            kind="preference",
            content="Prefer concise answers for travel questions",
            salience=0.9,
        ),
        MemoryRecord(
            memory_id="mem_session",
            scope="session",
            subject_id="sess_1",
            kind="task_fact",
            content="Debug repo command failures from the repo root",
            salience=0.4,
        ),
    ]

    selected = select_memories(
        records,
        query=MemoryQuery(
            session_id="sess_1",
            user_id="user_1",
            k=2,
            text_query="debug the repo command failure",
        ),
    )

    assert [record.memory_id for record in selected] == ["mem_session", "mem_user"]


def test_select_memories_collapses_duplicate_canonical_memories() -> None:
    records = [
        MemoryRecord(
            memory_id="mem_old",
            scope="user",
            subject_id="user_1",
            kind="task_fact",
            content="The repo root is /workspace/old.",
            dedupe_key="repo_root",
            salience=0.4,
        ),
        MemoryRecord(
            memory_id="mem_new",
            scope="user",
            subject_id="user_1",
            kind="task_fact",
            content="The repo root is /workspace/app.",
            dedupe_key="repo_root",
            salience=0.9,
        ),
    ]

    selected = select_memories(
        records,
        query=MemoryQuery(user_id="user_1", k=5, text_query="where is the repo root"),
    )

    assert [record.memory_id for record in selected] == ["mem_new"]


def test_score_memory_record_uses_metadata_in_overlap() -> None:
    record = MemoryRecord(
        memory_id="mem_1",
        scope="user",
        subject_id="user_1",
        kind="task_fact",
        content="Use the standard deployment flow.",
        metadata={"title": "Repo root", "description": "The repository root path"},
    )

    score = score_memory_record(
        record,
        scope_weights={"user": 2.0},
        query_tokens={"repo", "root"},
    )

    assert score > 2.0


def test_memory_canonical_key_prefers_metadata_then_dedupe_then_title() -> None:
    record = MemoryRecord(
        memory_id="mem_1",
        scope="user",
        subject_id="user_1",
        kind="preference",
        content="Prefer concise answers",
        dedupe_key="pref:concise",
        metadata={"canonical_key": "pref:override", "title": "Answer style"},
    )

    assert memory_canonical_key(record) == "pref:override"
