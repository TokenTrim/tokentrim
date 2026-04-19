from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tokentrim.memory.freshness import memory_age_days, memory_freshness_bucket
from tokentrim.memory.records import MemoryRecord
from tokentrim.memory.store import FilesystemMemoryStore, MemoryStore

MAX_MEMORY_FILES = 200


@dataclass(frozen=True, slots=True)
class MemoryHeader:
    memory_id: str
    scope: str
    subject_id: str
    file_path: str
    filename: str
    mtime_ms: float
    title: str
    description: str | None
    kind: str
    status: str
    salience: float
    dedupe_key: str | None
    canonical_key: str
    source_ref_count: int
    content_preview: str
    created_at: str
    updated_at: str
    age_days: int
    freshness_bucket: str


def scan_memory_headers(
    *,
    memory_store: MemoryStore,
    session_id: str | None,
    user_id: str | None,
    org_id: str | None,
) -> tuple[MemoryHeader, ...]:
    if isinstance(memory_store, FilesystemMemoryStore):
        return _scan_filesystem_headers(
            memory_store=memory_store,
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
        )
    return _scan_record_headers(
        memory_store=memory_store,
        session_id=session_id,
        user_id=user_id,
        org_id=org_id,
    )


def format_memory_manifest(headers: tuple[MemoryHeader, ...]) -> str:
    if not headers:
        return "_No candidate memory files._"
    lines: list[str] = []
    for scope in ("session", "user", "org"):
        scoped = [header for header in headers if header.scope == scope]
        if not scoped:
            continue
        lines.append(f"{scope.upper()}:")
        for header in scoped:
            lines.append(
                f"- {header.memory_id} | {header.filename} | {header.kind} | "
                f"status={header.status} | freshness={header.freshness_bucket} | "
                f"updated={header.updated_at} | {header.description or header.title}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _scan_filesystem_headers(
    *,
    memory_store: FilesystemMemoryStore,
    session_id: str | None,
    user_id: str | None,
    org_id: str | None,
) -> tuple[MemoryHeader, ...]:
    headers: list[MemoryHeader] = []
    for scope, subject_id in (
        ("session", session_id),
        ("user", user_id),
        ("org", org_id),
    ):
        if subject_id is None:
            continue
        directory = memory_store.scope_dir(scope=scope, subject_id=subject_id)
        if not directory.exists():
            continue
        for path in directory.glob("*.md"):
            if path.name == "MEMORY.md":
                continue
            record = memory_store._read_record(path)  # type: ignore[attr-defined]
            mtime_ms = path.stat().st_mtime * 1000
            headers.append(_build_header(record=record, file_path=path, mtime_ms=mtime_ms))
    headers.sort(key=lambda item: item.mtime_ms, reverse=True)
    return tuple(headers[:MAX_MEMORY_FILES])


def _scan_record_headers(
    *,
    memory_store: MemoryStore,
    session_id: str | None,
    user_id: str | None,
    org_id: str | None,
) -> tuple[MemoryHeader, ...]:
    records: list[MemoryRecord] = []
    for scope, subject_id in (
        ("session", session_id),
        ("user", user_id),
        ("org", org_id),
    ):
        if subject_id is None:
            continue
        records.extend(memory_store.list_memories(scope=scope, subject_id=subject_id))
    headers = [
        _build_header(
            record=record,
            file_path=Path(f"{record.scope}/{record.subject_id}/{record.memory_id}.md"),
            mtime_ms=0,
        )
        for record in records
    ]
    return tuple(headers[:MAX_MEMORY_FILES])


def _build_header(*, record: MemoryRecord, file_path: Path, mtime_ms: float) -> MemoryHeader:
    metadata = record.metadata if isinstance(record.metadata, Mapping) else {}
    description = metadata.get("description")
    title = metadata.get("title")
    normalized_title = title.strip() if isinstance(title, str) and title.strip() else record.kind.replace("_", " ").title()
    content_preview = " ".join(record.content.split())
    dedupe_key = record.dedupe_key.strip() if isinstance(record.dedupe_key, str) and record.dedupe_key.strip() else None
    return MemoryHeader(
        memory_id=record.memory_id,
        scope=record.scope,
        subject_id=record.subject_id,
        file_path=str(file_path),
        filename=file_path.name,
        mtime_ms=mtime_ms,
        title=normalized_title,
        description=description.strip() if isinstance(description, str) and description.strip() else None,
        kind=record.kind,
        status=record.status,
        salience=record.salience,
        dedupe_key=dedupe_key,
        canonical_key=_canonical_key(record=record, normalized_title=normalized_title),
        source_ref_count=len(record.source_refs),
        content_preview=content_preview[:157] + "..." if len(content_preview) > 160 else content_preview,
        created_at=record.created_at,
        updated_at=record.updated_at,
        age_days=memory_age_days(record.updated_at),
        freshness_bucket=memory_freshness_bucket(record.updated_at),
    )


def _canonical_key(*, record: MemoryRecord, normalized_title: str) -> str:
    metadata = record.metadata if isinstance(record.metadata, Mapping) else {}
    raw_key = metadata.get("canonical_key")
    if isinstance(raw_key, str) and raw_key.strip():
        return raw_key.strip().lower()
    if isinstance(record.dedupe_key, str) and record.dedupe_key.strip():
        return record.dedupe_key.strip().lower()
    return f"{record.scope}:{record.subject_id}:{record.kind}:{normalized_title.strip().lower()}"
