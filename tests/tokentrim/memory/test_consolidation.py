from __future__ import annotations

import pytest

from tokentrim.consolidator import (
    ConsolidationPlan,
    MemoryArchive,
    MemoryMerge,
    MemoryUpsert,
    apply_consolidation_plan,
    build_org_promotion,
    build_user_promotion,
)
from tokentrim.memory import (
    InMemoryMemoryStore,
    MemoryRecord,
    MemoryWrite,
)


def test_build_user_and_org_promotions_preserve_source_refs() -> None:
    source = MemoryRecord(
        memory_id="mem_session_1",
        scope="session",
        subject_id="sess_1",
        kind="constraint",
        content="Avoid destructive commands",
        source_refs=("trace_1",),
    )

    user_upsert = build_user_promotion(user_id="user_1", source_memory=source)
    org_upsert = build_org_promotion(org_id="org_1", source_memory=source)

    assert user_upsert.scope == "user"
    assert org_upsert.scope == "org"
    assert user_upsert.write.source_refs[0] == "mem_session_1"
    assert org_upsert.write.source_refs[0] == "mem_session_1"


def test_apply_consolidation_plan_upserts_and_archives_only_durable_scopes() -> None:
    store = InMemoryMemoryStore()
    existing = store.upsert_memory(
        MemoryRecord(
            memory_id="mem_user_existing",
            scope="user",
            subject_id="user_1",
            kind="preference",
            content="Prefer concise answers",
        )
    )

    result = apply_consolidation_plan(
        memory_store=store,
        plan=ConsolidationPlan(
            user_upserts=(
                MemoryUpsert(
                    scope="user",
                    subject_id="user_1",
                    memory_id=None,
                    write=MemoryWrite(
                        content="Avoid destructive commands unless requested",
                        kind="constraint",
                        dedupe_key="avoid_destructive",
                        source_refs=("trace_1",),
                    ),
                ),
            ),
            user_archives=(MemoryArchive(memory_id=existing.memory_id),),
        ),
    )

    assert len(result.upserted) == 1
    assert result.upserted[0].scope == "user"
    assert result.archived_memory_ids == (existing.memory_id,)
    archived = store.list_memories(scope="user", subject_id="user_1", limit=10)
    assert any(record.status == "archived" for record in archived)


def test_apply_consolidation_plan_archives_merge_sources() -> None:
    store = InMemoryMemoryStore()
    source_a = store.upsert_memory(
        MemoryRecord(
            memory_id="mem_org_a",
            scope="org",
            subject_id="org_1",
            kind="task_fact",
            content="Use repo root",
        )
    )
    source_b = store.upsert_memory(
        MemoryRecord(
            memory_id="mem_org_b",
            scope="org",
            subject_id="org_1",
            kind="task_fact",
            content="Debug from repo root",
        )
    )

    result = apply_consolidation_plan(
        memory_store=store,
        plan=ConsolidationPlan(
            merge_operations=(
                MemoryMerge(
                    target_memory_id="mem_org_target",
                    source_memory_ids=(source_a.memory_id, source_b.memory_id),
                ),
            ),
        ),
    )

    assert result.merged_source_ids == (source_a.memory_id, source_b.memory_id)


def test_apply_consolidation_plan_rejects_session_scope_upserts() -> None:
    store = InMemoryMemoryStore()

    with pytest.raises(ValueError):
        apply_consolidation_plan(
            memory_store=store,
            plan=ConsolidationPlan(
                user_upserts=(
                    MemoryUpsert(
                        scope="session",
                        subject_id="sess_1",
                        memory_id=None,
                        write=MemoryWrite(content="x", kind="constraint"),
                    ),
                ),
            ),
        )
