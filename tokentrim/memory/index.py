from __future__ import annotations

import json
from pathlib import Path

from tokentrim.memory.types import MemoryIndexRecord, MemoryScope


def load_index(path: Path) -> tuple[MemoryIndexRecord, ...]:
    if not path.exists():
        return ()

    records: list[MemoryIndexRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            scope_value = str(payload.get("scope") or MemoryScope.SESSION.value)
            records.append(
                MemoryIndexRecord(
                    entry_id=str(payload.get("entry_id") or ""),
                    path=str(payload["path"]) if payload.get("path") else None,
                    content=str(payload["content"]),
                    created_at=str(payload["created_at"]),
                    metadata=dict(payload.get("metadata") or {}),
                    keywords=tuple(payload.get("keywords") or ()),
                    scope=MemoryScope(scope_value),
                )
            )
    return tuple(records)


def write_index(path: Path, records: tuple[MemoryIndexRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "entry_id": record.entry_id,
                        "path": record.path,
                        "content": record.content,
                        "created_at": record.created_at,
                        "metadata": record.metadata,
                        "keywords": list(record.keywords),
                        "scope": record.scope.value,
                    },
                    sort_keys=True,
                )
            )
            handle.write("\n")
    temp_path.replace(path)
