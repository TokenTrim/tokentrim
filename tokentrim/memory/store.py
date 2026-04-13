from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from tokentrim.memory.query import select_memories
from tokentrim.memory.records import MemoryQuery, MemoryRecord, MemoryScope, MemoryWrite, utc_now_iso


class MemoryStore(ABC):
    @abstractmethod
    def list_memories(
        self,
        *,
        scope: MemoryScope,
        subject_id: str,
        kind: str | None = None,
        limit: int | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """Return memories for one scope/subject pair."""

    @abstractmethod
    def query_memories(self, query: MemoryQuery) -> tuple[MemoryRecord, ...]:
        """Return ranked candidates for injection."""

    @abstractmethod
    def write_session_memory(self, *, session_id: str, write: MemoryWrite) -> MemoryRecord:
        """Create or update a session-scoped memory record."""

    @abstractmethod
    def upsert_memory(self, record: MemoryRecord) -> MemoryRecord:
        """Create or replace a memory record."""

    @abstractmethod
    def archive_memory(self, *, memory_id: str) -> None:
        """Mark a memory as archived."""

    @abstractmethod
    def delete_memory(self, *, memory_id: str) -> None:
        """Delete a memory if the backend supports deletion."""


class InMemoryMemoryStore(MemoryStore):
    """Thread-safe process-local memory storage for tests and local runtimes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, MemoryRecord] = {}

    def list_memories(
        self,
        *,
        scope: MemoryScope,
        subject_id: str,
        kind: str | None = None,
        limit: int | None = None,
    ) -> tuple[MemoryRecord, ...]:
        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.scope == scope
                and record.subject_id == subject_id
                and (kind is None or record.kind == kind)
            ]
        return _sorted_records(records, limit=limit)

    def query_memories(self, query: MemoryQuery) -> tuple[MemoryRecord, ...]:
        with self._lock:
            records = list(self._records.values())
        return select_memories(records, query=query)

    def write_session_memory(self, *, session_id: str, write: MemoryWrite) -> MemoryRecord:
        _validate_session_subject(session_id)
        with self._lock:
            existing = _find_deduped_record(
                self._records.values(),
                scope="session",
                subject_id=session_id,
                dedupe_key=write.dedupe_key,
            )
            if existing is not None:
                updated = replace(
                    existing,
                    kind=write.kind,
                    content=write.content,
                    salience=write.salience,
                    dedupe_key=write.dedupe_key,
                    metadata=write.metadata,
                    source_refs=write.source_refs,
                    updated_at=utc_now_iso(),
                )
                self._records[updated.memory_id] = updated
                return updated
            created = _build_session_record(session_id=session_id, write=write)
            self._records[created.memory_id] = created
            return created

    def upsert_memory(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            self._records[record.memory_id] = record
        return record

    def archive_memory(self, *, memory_id: str) -> None:
        with self._lock:
            record = self._records.get(memory_id)
            if record is None:
                return
            self._records[memory_id] = replace(record, status="archived", updated_at=utc_now_iso())

    def delete_memory(self, *, memory_id: str) -> None:
        with self._lock:
            self._records.pop(memory_id, None)


class FilesystemMemoryStore(MemoryStore):
    """Filesystem-backed markdown memory store for local-first iteration."""

    def __init__(self, *, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def scope_dir(self, *, scope: MemoryScope, subject_id: str) -> Path:
        return self._root_dir / scope / subject_id

    def entrypoint_path(self, *, scope: MemoryScope, subject_id: str) -> Path:
        return self.scope_dir(scope=scope, subject_id=subject_id) / "MEMORY.md"

    def list_memories(
        self,
        *,
        scope: MemoryScope,
        subject_id: str,
        kind: str | None = None,
        limit: int | None = None,
    ) -> tuple[MemoryRecord, ...]:
        records = [
            record
            for record in self._read_scope_records(scope=scope, subject_id=subject_id)
            if kind is None or record.kind == kind
        ]
        return _sorted_records(records, limit=limit)

    def query_memories(self, query: MemoryQuery) -> tuple[MemoryRecord, ...]:
        records = [
            self._read_record(path)
            for path in self._root_dir.rglob("*.md")
            if path.name != "MEMORY.md"
        ]
        return select_memories(records, query=query)

    def write_session_memory(self, *, session_id: str, write: MemoryWrite) -> MemoryRecord:
        _validate_session_subject(session_id)
        existing = _find_deduped_record(
            self._read_scope_records(scope="session", subject_id=session_id),
            scope="session",
            subject_id=session_id,
            dedupe_key=write.dedupe_key,
        )
        if existing is not None:
            updated = replace(
                existing,
                kind=write.kind,
                content=write.content,
                salience=write.salience,
                dedupe_key=write.dedupe_key,
                metadata=write.metadata,
                source_refs=write.source_refs,
                updated_at=utc_now_iso(),
            )
            return self.upsert_memory(updated)
        return self.upsert_memory(_build_session_record(session_id=session_id, write=write))

    def upsert_memory(self, record: MemoryRecord) -> MemoryRecord:
        existing = self._find_by_memory_id(record.memory_id)
        path = self._record_path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_serialize_record(record), encoding="utf-8")
        if existing is not None and existing[0] != path:
            existing[0].unlink(missing_ok=True)
        self._refresh_entrypoint(scope=record.scope, subject_id=record.subject_id)
        return record

    def archive_memory(self, *, memory_id: str) -> None:
        existing = self._find_by_memory_id(memory_id)
        if existing is None:
            return
        path, record = existing
        path.write_text(
            _serialize_record(replace(record, status="archived", updated_at=utc_now_iso())),
            encoding="utf-8",
        )
        self._refresh_entrypoint(scope=record.scope, subject_id=record.subject_id)

    def delete_memory(self, *, memory_id: str) -> None:
        existing = self._find_by_memory_id(memory_id)
        if existing is None:
            return
        path, record = existing
        path.unlink(missing_ok=True)
        self._refresh_entrypoint(scope=record.scope, subject_id=record.subject_id)

    def _record_path(self, record: MemoryRecord) -> Path:
        return self.scope_dir(scope=record.scope, subject_id=record.subject_id) / f"{_record_stem(record)}.md"

    def _read_scope_records(self, *, scope: MemoryScope, subject_id: str) -> tuple[MemoryRecord, ...]:
        directory = self._root_dir / scope / subject_id
        if not directory.exists():
            return tuple()
        return tuple(
            self._read_record(path)
            for path in sorted(directory.glob("*.md"))
            if path.name != "MEMORY.md"
        )

    def _find_by_memory_id(self, memory_id: str) -> tuple[Path, MemoryRecord] | None:
        for path in self._root_dir.rglob("*.md"):
            if path.name == "MEMORY.md":
                continue
            record = self._read_record(path)
            if record.memory_id == memory_id:
                return path, record
        return None

    def _read_record(self, path: Path) -> MemoryRecord:
        return _deserialize_record(path.read_text(encoding="utf-8"))

    def _refresh_entrypoint(self, *, scope: MemoryScope, subject_id: str) -> None:
        directory = self.scope_dir(scope=scope, subject_id=subject_id)
        directory.mkdir(parents=True, exist_ok=True)
        records = [
            record
            for record in self._read_scope_records(scope=scope, subject_id=subject_id)
            if record.status == "active"
        ]
        entrypoint = self.entrypoint_path(scope=scope, subject_id=subject_id)
        entrypoint.write_text(_render_entrypoint(records), encoding="utf-8")


def _build_session_record(*, session_id: str, write: MemoryWrite) -> MemoryRecord:
    now = utc_now_iso()
    return MemoryRecord(
        memory_id=f"mem_{uuid4().hex}",
        scope="session",
        subject_id=session_id,
        kind=write.kind,
        content=write.content,
        salience=write.salience,
        status="active",
        source_refs=write.source_refs,
        created_at=now,
        updated_at=now,
        dedupe_key=write.dedupe_key,
        metadata=write.metadata,
    )


def _validate_session_subject(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")


def _find_deduped_record(
    records: object,
    *,
    scope: MemoryScope,
    subject_id: str,
    dedupe_key: str | None,
) -> MemoryRecord | None:
    if dedupe_key is None:
        return None
    for record in records:
        if not isinstance(record, MemoryRecord):
            continue
        if (
            record.scope == scope
            and record.subject_id == subject_id
            and record.status == "active"
            and record.dedupe_key == dedupe_key
        ):
            return record
    return None


def _sorted_records(records: list[MemoryRecord], *, limit: int | None) -> tuple[MemoryRecord, ...]:
    if limit is not None and limit <= 0:
        return tuple()
    ordered = sorted(records, key=lambda record: (record.updated_at, record.memory_id), reverse=True)
    if limit is None:
        return tuple(ordered)
    return tuple(ordered[:limit])


def _serialize_record(record: MemoryRecord) -> str:
    frontmatter = {
        "memory_id": record.memory_id,
        "scope": record.scope,
        "subject_id": record.subject_id,
        "kind": record.kind,
        "salience": record.salience,
        "status": record.status,
        "source_refs": list(record.source_refs),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "dedupe_key": record.dedupe_key,
        "metadata": dict(record.metadata) if record.metadata is not None else None,
    }
    lines = ["---"]
    for key, value in frontmatter.items():
        lines.append(f"{key}: {json.dumps(value, sort_keys=True)}")
    lines.extend(("---", "", record.content))
    return "\n".join(lines)


def _deserialize_record(payload: str) -> MemoryRecord:
    lines = payload.splitlines()
    if len(lines) < 4 or lines[0].strip() != "---":
        raise ValueError("memory file is missing frontmatter.")
    try:
        end_index = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("memory file frontmatter is not terminated.") from exc

    frontmatter: dict[str, object] = {}
    for line in lines[1:end_index]:
        key, separator, raw_value = line.partition(":")
        if separator != ":":
            raise ValueError("invalid frontmatter line.")
        frontmatter[key.strip()] = json.loads(raw_value.strip())

    content = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    metadata = frontmatter.get("metadata")
    return MemoryRecord(
        memory_id=str(frontmatter["memory_id"]),
        scope=str(frontmatter["scope"]),  # type: ignore[arg-type]
        subject_id=str(frontmatter["subject_id"]),
        kind=str(frontmatter["kind"]),
        content=content,
        salience=float(frontmatter["salience"]),
        status=str(frontmatter["status"]),  # type: ignore[arg-type]
        source_refs=tuple(frontmatter.get("source_refs") or ()),
        created_at=str(frontmatter["created_at"]),
        updated_at=str(frontmatter["updated_at"]),
        dedupe_key=frontmatter.get("dedupe_key") if isinstance(frontmatter.get("dedupe_key"), str) else None,
        metadata=metadata if isinstance(metadata, dict) else None,
    )


def _render_entrypoint(records: list[MemoryRecord]) -> str:
    lines = ["# MEMORY", ""]
    if not records:
        lines.append("_No active memories yet._")
        return "\n".join(lines)
    grouped: dict[str, list[MemoryRecord]] = {
        "constraint": [],
        "active_state": [],
        "task_fact": [],
        "decision": [],
        "preference": [],
        "other": [],
    }
    for record in sorted(records, key=lambda item: (item.updated_at, item.memory_id), reverse=True):
        grouped[record.kind if record.kind in grouped else "other"].append(record)
    for group_name, group_records in grouped.items():
        if not group_records:
            continue
        lines.append(f"## {group_name.replace('_', ' ').title()}")
        lines.append("")
        for record in group_records:
            lines.append(
                f"- [{_record_title(record)}]({_record_stem(record)}.md) — {_record_description(record)} "
                f"(updated {record.updated_at})"
            )
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _record_stem(record: MemoryRecord) -> str:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    raw_file_name = metadata.get("file_name") or metadata.get("title")
    if isinstance(raw_file_name, str) and raw_file_name.strip():
        sanitized = _sanitize_memory_file_name(raw_file_name)
        if sanitized:
            return sanitized
    return record.memory_id


def _record_title(record: MemoryRecord) -> str:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    raw_title = metadata.get("title")
    if isinstance(raw_title, str) and raw_title.strip():
        return raw_title.strip()
    return record.kind.replace("_", " ").title()


def _record_description(record: MemoryRecord) -> str:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    raw_description = metadata.get("description")
    if isinstance(raw_description, str) and raw_description.strip():
        return raw_description.strip()
    content = " ".join(record.content.split())
    return content[:117] + "..." if len(content) > 120 else content


def _sanitize_memory_file_name(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "-"
        for character in value.strip()
    )
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or ""
