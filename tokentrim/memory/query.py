from __future__ import annotations

import re
from datetime import datetime

from tokentrim.memory.records import MemoryQuery, MemoryRecord, MemoryScope

_TOKEN_RE = re.compile(r"[a-z0-9_./-]{3,}")
_DEFAULT_SCOPE_WEIGHTS: dict[MemoryScope, float] = {
    "session": 3.0,
    "user": 2.0,
    "org": 1.0,
}


def select_memories(
    records: list[MemoryRecord],
    *,
    query: MemoryQuery,
) -> tuple[MemoryRecord, ...]:
    scope_weights = dict(query.scope_weights or _DEFAULT_SCOPE_WEIGHTS)
    subject_filters = {
        "session": query.session_id,
        "user": query.user_id,
        "org": query.org_id,
    }
    filtered: list[MemoryRecord] = []
    for record in records:
        subject_id = subject_filters.get(record.scope)
        if subject_id is None:
            continue
        if record.subject_id != subject_id:
            continue
        if query.kind_filter and record.kind not in query.kind_filter:
            continue
        filtered.append(record)

    query_tokens = tokenize_text(query.text_query)
    filtered.sort(
        key=lambda record: (
            -score_memory_record(
                record,
                scope_weights=scope_weights,
                query_tokens=query_tokens,
            ),
            -timestamp_score(record.updated_at),
            record.memory_id,
        )
    )
    return tuple(filtered[: query.k])


def score_memory_record(
    record: MemoryRecord,
    *,
    scope_weights: dict[MemoryScope, float],
    query_tokens: set[str],
) -> float:
    score = float(scope_weights.get(record.scope, 0.0)) + record.salience
    if query_tokens:
        memory_tokens = tokenize_text(f"{record.kind} {record.content}")
        overlap = len(query_tokens & memory_tokens)
        score += overlap * 0.5
    if record.status != "active":
        score -= 10.0
    return score


def tokenize_text(text: str | None) -> set[str]:
    if not text:
        return set()
    return {match.group(0) for match in _TOKEN_RE.finditer(text.lower())}


def timestamp_score(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
