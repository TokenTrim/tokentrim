from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from tokentrim.consolidator.models import ConsolidationInput, ConsolidationPlan, MemoryUpsert
from tokentrim.memory.records import MemoryRecord, MemoryWrite
from tokentrim.tracing.records import TokentrimSpanRecord, TokentrimTraceRecord

TraceMemoryScope = Literal["user", "org"]

TRACE_FAILURE_RECOVERY_KIND = "failure_recovery"
TRACE_WORKFLOW_PATTERN_KIND = "workflow_pattern"

_FAILURE_MARKERS = (
    "fatal error:",
    "no such file or directory",
    "[exit_code] 1",
    "[exit_code] 2",
    "compilation terminated.",
    "error:",
    "failed",
)
_REPAIR_VERBS = (
    "install",
    "remove",
    "disable",
    "enable",
    "patch",
    "rerun",
    "retry",
    "rebuild",
    "edit",
    "update",
    "switch",
)


@dataclass(frozen=True, slots=True)
class TracePatternCandidate:
    scope: TraceMemoryScope
    subject_id: str
    dedupe_key: str
    content: str
    kind: str
    salience: float
    source_refs: tuple[str, ...]
    metadata: dict[str, object]
    rationale: str


def synthesize_trace_memory_plan(
    consolidation_input: ConsolidationInput,
    *,
    max_user_memories: int = 4,
    max_org_memories: int = 4,
) -> ConsolidationPlan:
    if max_user_memories < 0:
        raise ValueError("max_user_memories must be non-negative.")
    if max_org_memories < 0:
        raise ValueError("max_org_memories must be non-negative.")

    existing_user_keys = _active_dedupe_keys(consolidation_input.user_memories)
    existing_org_keys = _active_dedupe_keys(consolidation_input.org_memories)

    user_upserts: list[MemoryUpsert] = []
    org_upserts: list[MemoryUpsert] = []
    rationale: list[str] = []
    source_refs: list[str] = []

    for candidate in _iter_candidates(consolidation_input):
        if candidate.scope == "user":
            if len(user_upserts) >= max_user_memories or candidate.dedupe_key in existing_user_keys:
                continue
            user_upserts.append(_candidate_to_upsert(candidate))
            existing_user_keys.add(candidate.dedupe_key)
        else:
            if len(org_upserts) >= max_org_memories or candidate.dedupe_key in existing_org_keys:
                continue
            org_upserts.append(_candidate_to_upsert(candidate))
            existing_org_keys.add(candidate.dedupe_key)
        rationale.append(candidate.rationale)
        source_refs.extend(candidate.source_refs)

    return ConsolidationPlan(
        user_upserts=tuple(user_upserts),
        org_upserts=tuple(org_upserts),
        rationale=tuple(rationale),
        source_refs=tuple(dict.fromkeys(source_refs)),
    )


def _iter_candidates(consolidation_input: ConsolidationInput) -> tuple[TracePatternCandidate, ...]:
    candidates = [
        *_collect_failure_recovery_candidates(consolidation_input),
        *_collect_command_failure_insight_candidates(consolidation_input),
        *_collect_workflow_pattern_candidates(consolidation_input),
    ]
    candidates.sort(key=lambda candidate: candidate.salience, reverse=True)
    return tuple(candidates)


def _candidate_to_upsert(candidate: TracePatternCandidate) -> MemoryUpsert:
    return MemoryUpsert(
        scope=candidate.scope,
        subject_id=candidate.subject_id,
        memory_id=None,
        write=MemoryWrite(
            content=candidate.content,
            kind=candidate.kind,
            salience=candidate.salience,
            dedupe_key=candidate.dedupe_key,
            metadata=candidate.metadata,
            source_refs=candidate.source_refs,
        ),
    )


def _active_dedupe_keys(memories: tuple[MemoryRecord, ...]) -> set[str]:
    return {
        memory.dedupe_key
        for memory in memories
        if memory.status == "active" and memory.dedupe_key is not None
    }


def _collect_failure_recovery_candidates(
    consolidation_input: ConsolidationInput,
) -> tuple[TracePatternCandidate, ...]:
    grouped: dict[str, list[tuple[TokentrimTraceRecord, TokentrimSpanRecord, TokentrimSpanRecord]]] = defaultdict(list)

    for trace in consolidation_input.traces:
        if not isinstance(trace, TokentrimTraceRecord):
            continue
        sequence = _find_failure_recovery_sequence(trace)
        if sequence is None:
            continue
        failed_span, recovered_span = sequence
        signature = _failure_recovery_signature(trace, failed_span, recovered_span)
        grouped[signature].append((trace, failed_span, recovered_span))

    candidates: list[TracePatternCandidate] = []
    for signature, matches in grouped.items():
        exemplar_trace, failed_span, recovered_span = matches[-1]
        count = len(matches)
        scope, subject_id = _select_scope(consolidation_input=consolidation_input, evidence_count=count)
        candidates.append(
            TracePatternCandidate(
                scope=scope,
                subject_id=subject_id,
                dedupe_key=f"trace:{scope}:{signature}",
                kind=TRACE_FAILURE_RECOVERY_KIND,
                salience=0.85 if scope == "org" else 0.76,
                content=_render_failure_recovery_content(
                    trace=exemplar_trace,
                    failed_span=failed_span,
                    recovered_span=recovered_span,
                    count=count,
                    scope=scope,
                ),
                source_refs=tuple(
                    dict.fromkeys(
                        ref
                        for trace_record, failure_record, recovery_record in matches
                        for ref in (
                            trace_record.trace_id,
                            failure_record.span_id,
                            recovery_record.span_id,
                        )
                    )
                ),
                metadata={
                    "trace_pattern": "failure_recovery",
                    "workflow_name": exemplar_trace.workflow_name,
                    "failure_span": _span_label(failed_span),
                    "recovery_span": _span_label(recovered_span),
                    "evidence_count": count,
                },
                rationale=(
                    "Promoted a trace-derived failure/recovery pattern because the runtime observed "
                    f"{count} recovery path(s) for the same failure signature."
                ),
            )
        )
    return tuple(candidates)


def _collect_command_failure_insight_candidates(
    consolidation_input: ConsolidationInput,
) -> tuple[TracePatternCandidate, ...]:
    grouped: dict[str, list[tuple[TokentrimTraceRecord, TokentrimSpanRecord, str, str, str]]] = defaultdict(list)

    for trace in consolidation_input.traces:
        if not isinstance(trace, TokentrimTraceRecord):
            continue
        for span in trace.spans:
            if span.kind != "function":
                continue
            event = _extract_execute_command_event(span)
            if event is None:
                continue
            command, analysis, terminal_output = event
            if not _looks_like_failure_output(terminal_output):
                continue
            if not _looks_like_repair_attempt(command, analysis):
                continue
            issue_summary = _summarize_failure_output(terminal_output)
            repair_summary = _summarize_repair(analysis=analysis, command=command)
            signature = _normalize_signature(f"{trace.workflow_name}|{issue_summary}")
            grouped[signature].append((trace, span, issue_summary, repair_summary, command))

    candidates: list[TracePatternCandidate] = []
    for signature, matches in grouped.items():
        exemplar_trace, exemplar_span, issue_summary, repair_summary, command = matches[-1]
        count = len(matches)
        scope, subject_id = _select_scope(consolidation_input=consolidation_input, evidence_count=count)
        source_refs = tuple(
            dict.fromkeys(ref for trace, span, _, _, _ in matches for ref in (trace.trace_id, span.span_id))
        )
        candidates.append(
            TracePatternCandidate(
                scope=scope,
                subject_id=subject_id,
                dedupe_key=f"trace:{scope}:repair:{signature}",
                kind=TRACE_FAILURE_RECOVERY_KIND,
                salience=0.81 if count > 1 else 0.72,
                content=(
                    f"For workflow '{exemplar_trace.workflow_name}', when command output shows '{issue_summary}', "
                    f"a useful repair direction is: {repair_summary}. "
                    f"Representative command: {command}."
                ),
                source_refs=source_refs,
                metadata={
                    "trace_pattern": "command_failure_insight",
                    "workflow_name": exemplar_trace.workflow_name,
                    "issue_summary": issue_summary,
                    "repair_summary": repair_summary,
                    "evidence_count": count,
                },
                rationale=(
                    "Promoted a trace-derived repair heuristic because a failing command included a clear "
                    f"error signature and an explicit remediation attempt. Evidence count: {count}."
                ),
            )
        )
    return tuple(candidates)


def _collect_workflow_pattern_candidates(
    consolidation_input: ConsolidationInput,
) -> tuple[TracePatternCandidate, ...]:
    grouped: dict[str, list[TokentrimTraceRecord]] = defaultdict(list)

    for trace in consolidation_input.traces:
        if not isinstance(trace, TokentrimTraceRecord):
            continue
        if any(span.error for span in trace.spans):
            continue
        span_labels = tuple(_span_label(span) for span in trace.spans if _span_label(span))
        if len(span_labels) < 2:
            continue
        signature = f"{trace.workflow_name}|{'->'.join(span_labels)}"
        grouped[signature].append(trace)

    candidates: list[TracePatternCandidate] = []
    for signature, matches in grouped.items():
        exemplar_trace = matches[-1]
        count = len(matches)
        if count < 2:
            continue
        scope, subject_id = _select_scope(consolidation_input=consolidation_input, evidence_count=count)
        span_labels = tuple(_span_label(span) for span in exemplar_trace.spans if _span_label(span))
        candidates.append(
            TracePatternCandidate(
                scope=scope,
                subject_id=subject_id,
                dedupe_key=f"trace:{scope}:workflow:{signature}",
                kind=TRACE_WORKFLOW_PATTERN_KIND,
                salience=0.71,
                content=(
                    f"In workflow '{exemplar_trace.workflow_name}', a repeated successful sequence is: "
                    f"{' -> '.join(span_labels)}."
                ),
                source_refs=tuple(dict.fromkeys(trace.trace_id for trace in matches)),
                metadata={
                    "trace_pattern": "workflow_pattern",
                    "workflow_name": exemplar_trace.workflow_name,
                    "sequence": span_labels,
                    "evidence_count": count,
                },
                rationale=(
                    "Promoted a repeated successful workflow pattern because the same trace sequence "
                    f"completed successfully {count} times."
                ),
            )
        )
    return tuple(candidates)


def _find_failure_recovery_sequence(
    trace: TokentrimTraceRecord,
) -> tuple[TokentrimSpanRecord, TokentrimSpanRecord] | None:
    for index, span in enumerate(trace.spans):
        if not span.error:
            continue
        for candidate in trace.spans[index + 1 :]:
            if candidate.error:
                continue
            if candidate.kind == span.kind or candidate.parent_id == span.parent_id or candidate.parent_id == span.span_id:
                return span, candidate
        for candidate in trace.spans[index + 1 :]:
            if candidate.error:
                continue
            return span, candidate
    return None


def _failure_recovery_signature(
    trace: TokentrimTraceRecord,
    failed_span: TokentrimSpanRecord,
    recovered_span: TokentrimSpanRecord,
) -> str:
    error_summary = _error_summary(failed_span)
    return (
        f"{trace.workflow_name}|"
        f"{_span_label(failed_span)}|"
        f"{error_summary}|"
        f"{_span_label(recovered_span)}"
    )


def _render_failure_recovery_content(
    *,
    trace: TokentrimTraceRecord,
    failed_span: TokentrimSpanRecord,
    recovered_span: TokentrimSpanRecord,
    count: int,
    scope: TraceMemoryScope,
) -> str:
    prefix = "Across traces" if scope == "org" else "In this user's traces"
    return (
        f"{prefix} for workflow '{trace.workflow_name}', when '{_span_label(failed_span)}' failed "
        f"with '{_error_summary(failed_span)}', the recovery path that succeeded was "
        f"'{_span_label(recovered_span)}'. Observed {count} time(s)."
    )


def _extract_execute_command_event(span: TokentrimSpanRecord) -> tuple[str, str, str] | None:
    data = span.data
    if data.get("name") != "execute_command":
        return None
    input_payload = _parse_mapping_payload(data.get("input"))
    output_payload = _parse_mapping_payload(data.get("output"))
    command = _coerce_string(input_payload.get("command"))
    analysis = _coerce_string(input_payload.get("analysis"))
    terminal_output = _coerce_string(output_payload.get("terminal_output"))
    if not command or not terminal_output:
        return None
    return command, analysis, terminal_output


def _parse_mapping_payload(payload: object) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str) or not payload.strip():
        return {}
    text = payload.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _looks_like_failure_output(terminal_output: str) -> bool:
    lowered = terminal_output.lower()
    return any(marker in lowered for marker in _FAILURE_MARKERS)


def _looks_like_repair_attempt(command: str, analysis: str) -> bool:
    lowered = f"{analysis}\n{command}".lower()
    return any(verb in lowered for verb in _REPAIR_VERBS)


def _summarize_failure_output(terminal_output: str) -> str:
    for line in terminal_output.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if "fatal error:" in lowered:
            return stripped
        if "no such file or directory" in lowered:
            return stripped
        if stripped.startswith("[exit_code]"):
            return stripped
        if "error:" in lowered:
            return stripped
    return terminal_output.splitlines()[-1].strip() if terminal_output.splitlines() else "command failure"


def _summarize_repair(*, analysis: str, command: str) -> str:
    summary = analysis.strip()
    if summary:
        return summary.rstrip(".")
    return f"run `{command}`"


def _normalize_signature(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "trace_pattern"


def _select_scope(
    *,
    consolidation_input: ConsolidationInput,
    evidence_count: int,
) -> tuple[TraceMemoryScope, str]:
    if evidence_count >= 2 and consolidation_input.org_id is not None:
        return "org", consolidation_input.org_id
    return "user", consolidation_input.user_id


def _span_label(span: TokentrimSpanRecord) -> str:
    return span.name or span.data.get("name") or span.kind or span.span_id


def _error_summary(span: TokentrimSpanRecord) -> str:
    if not span.error:
        return "unknown error"
    type_value = span.error.get("type")
    message_value = span.error.get("message")
    if isinstance(type_value, str) and isinstance(message_value, str):
        return f"{type_value}: {message_value}"
    if isinstance(message_value, str):
        return message_value
    if isinstance(type_value, str):
        return type_value
    return "unknown error"


def _coerce_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
