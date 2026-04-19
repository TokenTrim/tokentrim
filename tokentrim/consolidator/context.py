from __future__ import annotations

import json

from tokentrim.consolidator.models import ConsolidationInput
from tokentrim.consolidator.engine import OfflineBundleView
from tokentrim.memory.records import MemoryRecord
from tokentrim.types.message import Message


def build_consolidation_bundle(consolidation_input: ConsolidationInput) -> OfflineBundleView:
    messages: list[Message] = [
        {
            "role": "meta",
            "content": _build_meta_message(consolidation_input),
        }
    ]
    messages.extend(
        {"role": "session_memory", "content": _render_memory(memory)}
        for memory in consolidation_input.session_memories
    )
    messages.extend(
        {"role": "user_memory", "content": _render_memory(memory)}
        for memory in consolidation_input.user_memories
    )
    messages.extend(
        {"role": "org_memory", "content": _render_memory(memory)}
        for memory in consolidation_input.org_memories
    )
    return OfflineBundleView.from_history(
        messages=messages,
        traces=tuple(
            trace for trace in consolidation_input.traces if hasattr(trace, "trace_id") and hasattr(trace, "spans")
        ),
        label="offline_bundle",
    )


def _build_meta_message(consolidation_input: ConsolidationInput) -> str:
    return (
        f"session_id: {consolidation_input.session_id}\n"
        f"user_id: {consolidation_input.user_id}\n"
        f"org_id: {consolidation_input.org_id or '(none)'}\n"
        f"trace_count: {len(consolidation_input.traces)}\n"
        f"session_memory_count: {len(consolidation_input.session_memories)}\n"
        f"user_memory_count: {len(consolidation_input.user_memories)}\n"
        f"org_memory_count: {len(consolidation_input.org_memories)}"
    )


def _render_memory(memory: MemoryRecord) -> str:
    payload = {
        "memory_id": memory.memory_id,
        "scope": memory.scope,
        "subject_id": memory.subject_id,
        "kind": memory.kind,
        "salience": memory.salience,
        "status": memory.status,
        "source_refs": list(memory.source_refs),
        "dedupe_key": memory.dedupe_key,
        "metadata": dict(memory.metadata) if memory.metadata is not None else None,
        "content": memory.content,
    }
    return json.dumps(payload, indent=2, sort_keys=True)
