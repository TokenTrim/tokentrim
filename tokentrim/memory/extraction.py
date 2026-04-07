from __future__ import annotations

import json
import re
from collections import Counter

from tokentrim.memory.writer import MemoryWriteCandidate
from tokentrim.tracing.records import TokentrimSpanRecord, TokentrimTraceRecord

_PATH_RE = re.compile(r"(?:~|/|\.\.?/)[^\s:;,)\]]+")


def build_trace_memory_candidate(
    traces: tuple[TokentrimTraceRecord, ...],
    *,
    task_hint: str | None = None,
) -> MemoryWriteCandidate | None:
    if not traces:
        return None

    resolution_candidate = _build_resolution_candidate(traces, task_hint=task_hint)
    if resolution_candidate is not None:
        return resolution_candidate

    repeated_error_candidate = _build_repeated_error_candidate(traces, task_hint=task_hint)
    if repeated_error_candidate is not None:
        return repeated_error_candidate
    return None


def _build_repeated_error_candidate(
    traces: tuple[TokentrimTraceRecord, ...],
    *,
    task_hint: str | None,
) -> MemoryWriteCandidate | None:
    signatures: list[tuple[str, TokentrimSpanRecord]] = []
    for trace in traces:
        for span in trace.spans:
            signature = _error_signature(span)
            if signature is not None:
                signatures.append((signature, span))
    if not signatures:
        return None

    counts = Counter(signature for signature, _ in signatures)
    repeated_signature, count = counts.most_common(1)[0]
    if count < 2:
        return None

    sample_span = next(span for signature, span in signatures if signature == repeated_signature)
    lines = [f"Repeated failure observed across {count} traces: {repeated_signature}"]
    if task_hint:
        lines.append(f"Task hint: {task_hint}")
    artifacts = _extract_artifacts(sample_span)
    if artifacts:
        lines.append("Artifacts: " + ", ".join(artifacts))
    return MemoryWriteCandidate(
        content="\n".join(lines),
        metadata={"kind": "trace_repeated_error", "count": count},
    )


def _build_resolution_candidate(
    traces: tuple[TokentrimTraceRecord, ...],
    *,
    task_hint: str | None,
) -> MemoryWriteCandidate | None:
    if len(traces) < 2:
        return None
    latest = traces[0]
    latest_errors = [_error_signature(span) for span in latest.spans]
    latest_errors = [error for error in latest_errors if error is not None]
    if latest_errors:
        return None

    prior_errors: list[str] = []
    for trace in traces[1:]:
        prior_errors.extend(error for error in (_error_signature(span) for span in trace.spans) if error is not None)
    if not prior_errors:
        return None

    repeated_signature, count = Counter(prior_errors).most_common(1)[0]
    if count < 2:
        return None

    lines = [f"Resolved prior repeated failure: {repeated_signature}"]
    if task_hint:
        lines.append(f"Task hint: {task_hint}")
    if latest.ended_at:
        lines.append(f"Resolved by: {latest.ended_at}")
    return MemoryWriteCandidate(
        content="\n".join(lines),
        metadata={"kind": "trace_resolution", "count": count},
    )


def _error_signature(span: TokentrimSpanRecord) -> str | None:
    if span.error is None:
        return None
    if "message" in span.error and isinstance(span.error["message"], str):
        return span.error["message"]
    if "type" in span.error and isinstance(span.error["type"], str):
        return span.error["type"]
    return json.dumps(span.error, sort_keys=True)


def _extract_artifacts(span: TokentrimSpanRecord) -> tuple[str, ...]:
    values: list[str] = []
    for source in (span.data, span.raw_span):
        for value in source.values():
            if isinstance(value, str):
                values.extend(_PATH_RE.findall(value))
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered[:4])
