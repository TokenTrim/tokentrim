from __future__ import annotations

"""Core data models for offline durable-memory consolidation."""

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from tokentrim.memory.records import MemoryRecord, MemoryWrite, utc_now_iso
from tokentrim.memory.store import MemoryStore

DurableMemoryWriteScope = Literal["all", "user", "org"]


@dataclass(frozen=True, slots=True)
class ConsolidationInput:
    """Immutable offline bundle for one completed session."""

    session_id: str
    user_id: str
    org_id: str | None
    traces: tuple[object, ...]
    session_memories: tuple[MemoryRecord, ...]
    user_memories: tuple[MemoryRecord, ...]
    org_memories: tuple[MemoryRecord, ...]


@dataclass(frozen=True, slots=True)
class MemoryUpsert:
    """Create or replace one durable memory record."""

    scope: str
    subject_id: str
    memory_id: str | None
    write: MemoryWrite


@dataclass(frozen=True, slots=True)
class MemoryArchive:
    """Archive an obsolete durable memory record."""

    memory_id: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryMerge:
    """Collapse multiple durable memories into one surviving record."""

    target_memory_id: str
    source_memory_ids: tuple[str, ...]
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ConsolidationPlan:
    """Durable-memory edit plan produced by a consolidator agent."""

    user_upserts: tuple[MemoryUpsert, ...] = ()
    org_upserts: tuple[MemoryUpsert, ...] = ()
    user_archives: tuple[MemoryArchive, ...] = ()
    org_archives: tuple[MemoryArchive, ...] = ()
    merge_operations: tuple[MemoryMerge, ...] = ()
    rationale: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConsolidationApplyResult:
    """Observed durable-memory writes after applying a plan."""

    upserted: tuple[MemoryRecord, ...] = ()
    archived_memory_ids: tuple[str, ...] = ()
    merged_source_ids: tuple[str, ...] = ()


def merge_consolidation_plans(*plans: ConsolidationPlan) -> ConsolidationPlan:
    """Merge multiple plans while deduplicating overlapping edits."""
    user_upserts = _dedupe_upserts(upsert for plan in plans for upsert in plan.user_upserts)
    org_upserts = _dedupe_upserts(upsert for plan in plans for upsert in plan.org_upserts)
    user_archives = _dedupe_archives(archive for plan in plans for archive in plan.user_archives)
    org_archives = _dedupe_archives(archive for plan in plans for archive in plan.org_archives)
    merge_operations = _dedupe_merges(merge for plan in plans for merge in plan.merge_operations)
    rationale = _dedupe_strings(item for plan in plans for item in plan.rationale)
    source_refs = _dedupe_strings(item for plan in plans for item in plan.source_refs)
    return ConsolidationPlan(
        user_upserts=user_upserts,
        org_upserts=org_upserts,
        user_archives=user_archives,
        org_archives=org_archives,
        merge_operations=merge_operations,
        rationale=rationale,
        source_refs=source_refs,
    )


def restrict_consolidation_plan(
    *,
    plan: ConsolidationPlan,
    write_scope: DurableMemoryWriteScope,
) -> ConsolidationPlan:
    """Drop plan edits that exceed the allowed durable write scope.

    Merge operations are disabled for partial-scope runs because merges archive
    source memories and therefore implicitly span more than one write target.
    """
    if write_scope == "all":
        return plan
    if write_scope == "user":
        return ConsolidationPlan(
            user_upserts=plan.user_upserts,
            org_upserts=(),
            user_archives=plan.user_archives,
            org_archives=(),
            merge_operations=(),
            rationale=plan.rationale,
            source_refs=plan.source_refs,
        )
    if write_scope == "org":
        return ConsolidationPlan(
            user_upserts=(),
            org_upserts=plan.org_upserts,
            user_archives=(),
            org_archives=plan.org_archives,
            merge_operations=(),
            rationale=plan.rationale,
            source_refs=plan.source_refs,
        )
    raise ValueError("write_scope must be one of: all, user, org.")


def apply_consolidation_plan(
    *,
    plan: ConsolidationPlan,
    memory_store: MemoryStore,
) -> ConsolidationApplyResult:
    """Apply a validated consolidation plan to durable memory."""
    upserted: list[MemoryRecord] = []
    archived_memory_ids: list[str] = []
    merged_source_ids: list[str] = []

    for upsert in (*plan.user_upserts, *plan.org_upserts):
        if upsert.scope not in {"user", "org"}:
            raise ValueError("Consolidation upserts may target only user or org scope.")
        now = utc_now_iso()
        record = MemoryRecord(
            memory_id=upsert.memory_id or _generate_memory_id(upsert.scope),
            scope=upsert.scope,
            subject_id=upsert.subject_id,
            kind=upsert.write.kind,
            content=upsert.write.content,
            salience=upsert.write.salience,
            status="active",
            source_refs=upsert.write.source_refs,
            created_at=now,
            updated_at=now,
            dedupe_key=upsert.write.dedupe_key,
            metadata=upsert.write.metadata,
        )
        upserted.append(memory_store.upsert_memory(record))

    for archive in (*plan.user_archives, *plan.org_archives):
        memory_store.archive_memory(memory_id=archive.memory_id)
        archived_memory_ids.append(archive.memory_id)

    for merge in plan.merge_operations:
        for source_id in merge.source_memory_ids:
            memory_store.archive_memory(memory_id=source_id)
            merged_source_ids.append(source_id)

    return ConsolidationApplyResult(
        upserted=tuple(upserted),
        archived_memory_ids=tuple(archived_memory_ids),
        merged_source_ids=tuple(merged_source_ids),
    )


def build_user_promotion(
    *,
    user_id: str,
    source_memory: MemoryRecord,
    overrides: MemoryWrite | None = None,
) -> MemoryUpsert:
    """Promote a session memory into durable user memory."""
    write = overrides or _build_promotion_write(source_memory)
    return MemoryUpsert(
        scope="user",
        subject_id=user_id,
        memory_id=None,
        write=write,
    )


def build_org_promotion(
    *,
    org_id: str,
    source_memory: MemoryRecord,
    overrides: MemoryWrite | None = None,
) -> MemoryUpsert:
    """Promote a session memory into durable organization memory."""
    write = overrides or _build_promotion_write(source_memory)
    return MemoryUpsert(
        scope="org",
        subject_id=org_id,
        memory_id=None,
        write=write,
    )


def _generate_memory_id(scope: str) -> str:
    timestamp = utc_now_iso().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    return f"{scope}_mem_{timestamp}_{uuid4().hex[:8]}"


def _build_promotion_write(source_memory: MemoryRecord) -> MemoryWrite:
    """Derive a durable write payload from an existing source memory."""
    return MemoryWrite(
        content=source_memory.content,
        kind=source_memory.kind,
        salience=source_memory.salience,
        dedupe_key=source_memory.dedupe_key,
        metadata=source_memory.metadata,
        source_refs=(source_memory.memory_id, *source_memory.source_refs),
    )


def _dedupe_upserts(upserts: object) -> tuple[MemoryUpsert, ...]:
    seen: set[tuple[object, ...]] = set()
    ordered: list[MemoryUpsert] = []
    for upsert in upserts:
        key = (
            upsert.scope,
            upsert.subject_id,
            upsert.write.dedupe_key,
            upsert.write.kind,
            upsert.write.content,
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(upsert)
    return tuple(ordered)


def _dedupe_archives(archives: object) -> tuple[MemoryArchive, ...]:
    seen: set[str] = set()
    ordered: list[MemoryArchive] = []
    for archive in archives:
        if archive.memory_id in seen:
            continue
        seen.add(archive.memory_id)
        ordered.append(archive)
    return tuple(ordered)


def _dedupe_merges(merges: object) -> tuple[MemoryMerge, ...]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    ordered: list[MemoryMerge] = []
    for merge in merges:
        key = (merge.target_memory_id, tuple(merge.source_memory_ids))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(merge)
    return tuple(ordered)


def _dedupe_strings(items: object) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)
