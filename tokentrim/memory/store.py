from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import threading
from uuid import uuid4

from tokentrim.memory.index import load_index, write_index
from tokentrim.memory.markdown import read_memory_markdown, write_memory_markdown
from tokentrim.memory.writer import normalize_memory_content
from tokentrim.memory.types import (
    MemoryIndexRecord,
    MemoryScope,
    default_memory_root,
    default_project_id,
    sanitize_memory_segment,
)
from tokentrim.salience import extract_query_terms, score_text_salience


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    content: str
    created_at: str
    metadata: dict[str, object]
    keywords: tuple[str, ...]
    scope: MemoryScope = MemoryScope.SESSION


class DurableMemoryStore(ABC):
    @abstractmethod
    def remember(
        self,
        *,
        user_id: str,
        session_id: str,
        content: str,
        metadata: dict[str, object] | None = None,
        scope: MemoryScope = MemoryScope.SESSION,
        project_id: str | None = None,
    ) -> None:
        """Persist a new durable memory record."""

    @abstractmethod
    def retrieve(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        limit: int = 3,
        scopes: tuple[MemoryScope, ...] = (MemoryScope.SESSION,),
        project_id: str | None = None,
    ) -> tuple[MemoryEntry, ...]:
        """Return the most relevant durable memory entries for the query."""


class LocalDirectoryMemoryStore(DurableMemoryStore):
    def __init__(
        self,
        *,
        root_dir: str | Path | None = None,
        max_entries_per_session: int = 200,
    ) -> None:
        self._root_dir = Path(root_dir) if root_dir is not None else default_memory_root()
        self._max_entries_per_session = max_entries_per_session
        self._lock = threading.Lock()

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def remember(
        self,
        *,
        user_id: str,
        session_id: str,
        content: str,
        metadata: dict[str, object] | None = None,
        scope: MemoryScope = MemoryScope.SESSION,
        project_id: str | None = None,
    ) -> None:
        with self._lock:
            entry = self._build_entry(content=content, metadata=metadata, scope=scope)
            if entry is None:
                return

            index_path = self.index_path(
                user_id=user_id,
                session_id=session_id,
                scope=scope,
                project_id=project_id,
            )
            existing_records = list(load_index(index_path))
            if self._is_duplicate(existing_records, entry):
                return
            updated_records = self._append_record(
                existing_records=existing_records,
                entry=entry,
                index_path=index_path,
            )
            write_index(index_path, updated_records)

    def retrieve(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        limit: int = 3,
        scopes: tuple[MemoryScope, ...] = (MemoryScope.SESSION,),
        project_id: str | None = None,
    ) -> tuple[MemoryEntry, ...]:
        entries = self._load_entries(
            user_id=user_id,
            session_id=session_id,
            scopes=scopes,
            project_id=project_id,
        )
        if not entries:
            return ()

        query_terms = tuple(set(_extract_keywords(query)))
        if not query_terms:
            return tuple(entries[-limit:])

        return _select_relevant_entries(entries, query_terms=query_terms, limit=limit)

    def session_index_path(self, *, user_id: str, session_id: str) -> Path:
        safe_user = sanitize_memory_segment(user_id)
        safe_session = sanitize_memory_segment(session_id)
        return self._root_dir / "users" / safe_user / "sessions" / safe_session / "index.jsonl"

    def user_index_path(self, *, user_id: str) -> Path:
        safe_user = sanitize_memory_segment(user_id)
        return self._root_dir / "users" / safe_user / "global" / "index.jsonl"

    def project_index_path(self, *, project_id: str | None = None) -> Path:
        safe_project = sanitize_memory_segment(project_id or default_project_id())
        return self._root_dir / "projects" / safe_project / "index.jsonl"

    def index_path(
        self,
        *,
        user_id: str,
        session_id: str,
        scope: MemoryScope,
        project_id: str | None = None,
    ) -> Path:
        if scope is MemoryScope.SESSION:
            return self.session_index_path(user_id=user_id, session_id=session_id)
        if scope is MemoryScope.USER:
            return self.user_index_path(user_id=user_id)
        return self.project_index_path(project_id=project_id)

    def session_entries_dir(self, *, user_id: str, session_id: str) -> Path:
        return self.session_index_path(user_id=user_id, session_id=session_id).parent / "entries"

    def _build_entry(
        self,
        *,
        content: str,
        metadata: dict[str, object] | None,
        scope: MemoryScope,
    ) -> MemoryEntry | None:
        normalized = normalize_memory_content(content)
        if not normalized:
            return None
        return MemoryEntry(
            content=normalized,
            created_at=datetime.now(UTC).isoformat(),
            metadata=dict(metadata or {}),
            keywords=_extract_keywords(normalized),
            scope=scope,
        )

    def _is_duplicate(
        self,
        existing_records: list[MemoryIndexRecord],
        entry: MemoryEntry,
    ) -> bool:
        if not existing_records:
            return False
        last = existing_records[-1]
        return (
            normalize_memory_content(last.content) == entry.content
            and dict(last.metadata) == entry.metadata
            and last.scope is entry.scope
        )

    def _append_record(
        self,
        *,
        existing_records: list[MemoryIndexRecord],
        entry: MemoryEntry,
        index_path: Path,
    ) -> tuple[MemoryIndexRecord, ...]:
        entry_id = f"mem_{uuid4().hex[:12]}"
        entry_path = self._write_entry_markdown(entry=entry, entry_id=entry_id, index_path=index_path)
        existing_records.append(
            _to_index_record(entry, entry_id=entry_id, path=entry_path.relative_to(index_path.parent).as_posix())
        )
        return tuple(existing_records[-self._max_entries_per_session :])

    def _write_entry_markdown(
        self,
        *,
        entry: MemoryEntry,
        entry_id: str,
        index_path: Path,
    ) -> Path:
        entry_path = index_path.parent / "entries" / _entry_filename(created_at=entry.created_at, entry_id=entry_id)
        write_memory_markdown(
            entry_path,
            entry_id=entry_id,
            scope=entry.scope,
            created_at=entry.created_at,
            keywords=entry.keywords,
            metadata=entry.metadata,
            content=entry.content,
        )
        return entry_path

    def _load_entries(
        self,
        *,
        user_id: str,
        session_id: str,
        scopes: tuple[MemoryScope, ...],
        project_id: str | None,
    ) -> tuple[MemoryEntry, ...]:
        entries: list[MemoryEntry] = []
        for scope in scopes:
            entries.extend(
                self._read_entries(
                    self.index_path(
                        user_id=user_id,
                        session_id=session_id,
                        scope=scope,
                        project_id=project_id,
                    )
                )
            )
        return tuple(entries)

    def _read_entries(self, path: Path) -> tuple[MemoryEntry, ...]:
        entries: list[MemoryEntry] = []
        for record in load_index(path):
            entry = self._load_entry_from_record(path=path, record=record)
            if entry is not None:
                entries.append(entry)
        return tuple(entries)

    def _load_entry_from_record(
        self,
        *,
        path: Path,
        record: MemoryIndexRecord,
    ) -> MemoryEntry | None:
        if record.path is None:
            return MemoryEntry(
                content=record.content,
                created_at=record.created_at,
                metadata=record.metadata,
                keywords=record.keywords or _extract_keywords(record.content),
                scope=record.scope,
            )
        markdown_path = path.parent / record.path
        if not markdown_path.exists():
            return None
        payload = read_memory_markdown(markdown_path)
        content = str(payload["content"])
        created_at = str(payload.get("created_at") or record.created_at)
        metadata = _coerce_metadata(payload.get("metadata_json"), fallback=record.metadata)
        keywords = tuple(payload.get("keywords") or record.keywords or _extract_keywords(content))
        return MemoryEntry(
            content=content,
            created_at=created_at,
            metadata=metadata,
            keywords=keywords,
            scope=record.scope,
        )


def _extract_keywords(text: str) -> tuple[str, ...]:
    return extract_query_terms(text)


def _score_entry(
    entry: MemoryEntry,
    query_terms: tuple[str, ...],
    *,
    recency_rank: int,
) -> int:
    return score_text_salience(
        entry.content,
        query_terms=query_terms,
        recency_rank=recency_rank,
    )


def _select_relevant_entries(
    entries: tuple[MemoryEntry, ...],
    *,
    query_terms: tuple[str, ...],
    limit: int,
) -> tuple[MemoryEntry, ...]:
    scored = sorted(
        (
            (_score_entry(entry, query_terms, recency_rank=len(entries) - index - 1), index, entry)
            for index, entry in enumerate(entries)
        ),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    selected = [entry for score, _, entry in scored if score > 0][:limit]
    if not selected:
        selected = list(entries[-limit:])
    return tuple(selected)


def _to_index_record(entry: MemoryEntry, *, entry_id: str, path: str | None) -> MemoryIndexRecord:
    return MemoryIndexRecord(
        entry_id=entry_id,
        path=path,
        content=entry.content,
        created_at=entry.created_at,
        metadata=entry.metadata,
        keywords=entry.keywords,
        scope=entry.scope,
    )


def _coerce_metadata(value: object, *, fallback: dict[str, object]) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): data for key, data in value.items()}
    return dict(fallback)


def _entry_filename(*, created_at: str, entry_id: str) -> str:
    safe_timestamp = created_at.replace(":", "-").replace("+", "_")
    return f"{safe_timestamp}_{entry_id}.md"
