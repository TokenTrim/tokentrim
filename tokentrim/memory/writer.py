from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from tokentrim.memory.records import MemoryRecord, MemoryWrite
from tokentrim.memory.store import MemoryStore


@dataclass(frozen=True, slots=True)
class SessionMemoryWriter:
    """Narrow facade for explicit live-runtime session memory writes.

    This is a host/runtime surface, not a retrieval surface. Integrations may
    choose to wrap it in a tool, but Tokentrim core does not require that.
    """

    memory_store: MemoryStore
    session_id: str

    def write(
        self,
        *,
        content: str,
        kind: str,
        salience: float = 0.5,
        dedupe_key: str | None = None,
        metadata: Mapping[str, object] | None = None,
        source_refs: tuple[str, ...] = (),
    ) -> MemoryRecord:
        return self.memory_store.write_session_memory(
            session_id=self.session_id,
            write=MemoryWrite(
                content=content,
                kind=kind,
                salience=salience,
                dedupe_key=dedupe_key,
                metadata=metadata,
                source_refs=source_refs,
            ),
        )


def write_session_memory(
    *,
    memory_store: MemoryStore,
    session_id: str,
    content: str,
    kind: str,
    salience: float = 0.5,
    dedupe_key: str | None = None,
    metadata: Mapping[str, object] | None = None,
    source_refs: tuple[str, ...] = (),
) -> MemoryRecord:
    """Convenience helper for explicit session-memory writes.

    Writes are intentionally limited to session scope. Durable memory promotion
    belongs to asynchronous consolidation, not the live runtime.
    """

    return SessionMemoryWriter(memory_store=memory_store, session_id=session_id).write(
        content=content,
        kind=kind,
        salience=salience,
        dedupe_key=dedupe_key,
        metadata=metadata,
        source_refs=source_refs,
    )
