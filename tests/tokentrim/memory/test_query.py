from __future__ import annotations

from tokentrim.memory import (
    MemoryQuery,
    MemoryRecord,
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
