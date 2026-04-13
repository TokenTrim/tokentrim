from __future__ import annotations

import re
from datetime import datetime

from tokentrim.memory.freshness import memory_age_days, memory_freshness_bucket
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
    deduped: list[MemoryRecord] = []
    seen_canonical_keys: set[str] = set()
    for record in filtered:
        canonical_key = memory_canonical_key(record)
        if canonical_key in seen_canonical_keys:
            continue
        seen_canonical_keys.add(canonical_key)
        deduped.append(record)
    return tuple(deduped[: query.k])


def score_memory_record(
    record: MemoryRecord,
    *,
    scope_weights: dict[MemoryScope, float],
    query_tokens: set[str],
) -> float:
    score = float(scope_weights.get(record.scope, 0.0)) + record.salience
    if query_tokens:
        memory_tokens = tokenize_text(_record_search_text(record))
        overlap = len(query_tokens & memory_tokens)
        score += overlap * 0.5
    freshness_bucket = memory_freshness_bucket(record.updated_at)
    if freshness_bucket == "aging":
        score -= 0.25
    elif freshness_bucket == "stale":
        score -= 0.75
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


def memory_canonical_key(record: MemoryRecord) -> str:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    raw_key = metadata.get("canonical_key")
    if isinstance(raw_key, str) and raw_key.strip():
        return raw_key.strip().lower()
    if isinstance(record.dedupe_key, str) and record.dedupe_key.strip():
        return record.dedupe_key.strip().lower()
    raw_title = metadata.get("title")
    title = raw_title.strip().lower() if isinstance(raw_title, str) and raw_title.strip() else record.kind.lower()
    return f"{record.scope}:{record.subject_id}:{record.kind}:{title}"


def _record_search_text(record: MemoryRecord) -> str:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    title = metadata.get("title") if isinstance(metadata.get("title"), str) else ""
    description = metadata.get("description") if isinstance(metadata.get("description"), str) else ""
    canonical_key = metadata.get("canonical_key") if isinstance(metadata.get("canonical_key"), str) else ""
    age = str(memory_age_days(record.updated_at))
    return " ".join(
        part
        for part in (
            record.kind,
            record.content,
            title,
            description,
            record.dedupe_key or "",
            canonical_key,
            " ".join(record.source_refs),
            age,
        )
        if part
    )
