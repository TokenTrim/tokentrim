from __future__ import annotations

import json
from collections.abc import Mapping

from tokentrim.core.llm_client import generate_text
from tokentrim.memory.manifest import format_memory_manifest, scan_memory_headers
from tokentrim.memory.records import MemoryRecord
from tokentrim.memory.store import MemoryStore

_SELECTOR_SYSTEM_PROMPT = (
    "You are selecting memory files that should be injected into an agent context. "
    "Use the memory-file manifest, not the raw body contents, as your retrieval surface. "
    "Be selective. Choose only memories that are clearly useful for the current query. "
    "Prefer memories that are specific, recent, and actionable. "
    "Return JSON with one key: selected_memory_ids."
)


def select_memory_candidates(
    *,
    memory_store: MemoryStore,
    candidates: tuple[MemoryRecord, ...],
    session_id: str | None,
    user_id: str | None,
    org_id: str | None,
    text_query: str | None,
    selector_model: str | None,
) -> tuple[MemoryRecord, ...]:
    if not candidates:
        return ()
    if selector_model is None or text_query is None or not text_query.strip():
        return candidates

    manifest_headers = scan_memory_headers(
        memory_store=memory_store,
        session_id=session_id,
        user_id=user_id,
        org_id=org_id,
    )
    candidate_ids = {candidate.memory_id for candidate in candidates}
    selected_headers = tuple(header for header in manifest_headers if header.memory_id in candidate_ids)
    if not selected_headers:
        return candidates

    try:
        response = generate_text(
            model=selector_model,
            messages=[
                {"role": "system", "content": _SELECTOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": text_query,
                            "available_memories": [
                                {
                                    "memory_id": header.memory_id,
                                    "scope": header.scope,
                                    "kind": header.kind,
                                    "title": header.title,
                                    "description": header.description,
                                    "filename": header.filename,
                                    "updated_at": header.updated_at,
                                    "file_path": header.file_path,
                                }
                                for header in selected_headers
                            ],
                            "manifest": format_memory_manifest(selected_headers),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                },
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "tokentrim_memory_selection",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "selected_memory_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["selected_memory_ids"],
                    },
                },
            },
        )
    except Exception:
        return candidates
    selected_ids = _parse_selected_ids(response)
    if not selected_ids:
        return candidates
    selected_lookup = {candidate.memory_id: candidate for candidate in candidates}
    return tuple(
        selected_lookup[memory_id]
        for memory_id in selected_ids
        if memory_id in selected_lookup
    )


def _parse_selected_ids(payload: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, Mapping):
        return ()
    selected = parsed.get("selected_memory_ids")
    if not isinstance(selected, list):
        return ()
    return tuple(item for item in selected if isinstance(item, str) and item.strip())
