from __future__ import annotations

"""Consolidator agents that convert offline bundles into durable-memory plans."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from tokentrim.core.llm_client import generate_text
from tokentrim.errors.base import TokentrimError
from tokentrim.consolidator.context import build_consolidation_bundle
from tokentrim.consolidator.engine import LocalConsolidatorRuntime
from tokentrim.consolidator.models import ConsolidationInput, ConsolidationPlan, merge_consolidation_plans
from tokentrim.consolidator.models import MemoryArchive, MemoryMerge, MemoryUpsert
from tokentrim.consolidator.synthesis import synthesize_trace_memory_plan
from tokentrim.memory.records import MemoryRecord, MemoryWrite
from tokentrim.tracing.records import TokentrimTraceRecord


class ConsolidatorAgent(Protocol):
    def build_plan(self, consolidation_input: ConsolidationInput) -> ConsolidationPlan:
        """Return a durable-memory edit plan for the completed session bundle."""


@dataclass(frozen=True, slots=True)
class DeterministicConsolidatorAgent:
    """Default controlled consolidator agent backed by deterministic synthesis."""

    max_user_memories: int = 4
    max_org_memories: int = 4

    def build_plan(self, consolidation_input: ConsolidationInput) -> ConsolidationPlan:
        return synthesize_trace_memory_plan(
            consolidation_input,
            max_user_memories=self.max_user_memories,
            max_org_memories=self.max_org_memories,
        )


@dataclass(frozen=True, slots=True)
class ModelConsolidatorAgent:
    """LLM-backed offline consolidator that returns a validated edit plan."""

    model: str
    backend: str = "openai"
    temperature: float = 0.0
    completion_options: Mapping[str, Any] | None = None

    def build_plan(self, consolidation_input: ConsolidationInput) -> ConsolidationPlan:
        baseline_plan = synthesize_trace_memory_plan(consolidation_input)
        payload = serialize_consolidation_input(consolidation_input)
        response = generate_text(
            model=_resolve_model_name(backend=self.backend, model=self.model),
            messages=[
                {"role": "system", "content": build_consolidator_system_prompt()},
                {"role": "user", "content": _build_user_prompt(payload)},
            ],
            temperature=self.temperature,
            response_format=_build_response_format(),
            completion_options=self.completion_options,
        )
        return merge_consolidation_plans(
            baseline_plan,
            parse_consolidation_plan_response(response),
        )


@dataclass(frozen=True, slots=True)
class AgenticConsolidatorAgent:
    """Iterative offline consolidator built on Tokentrim's local agentic runtime."""

    model: str
    backend: str = "openai"
    max_iterations: int = 8
    max_depth: int = 1
    max_subcalls: int = 4
    subcall_model: str | None = None
    tokenizer_model: str | None = None

    def build_plan(self, consolidation_input: ConsolidationInput) -> ConsolidationPlan:
        baseline_plan = synthesize_trace_memory_plan(consolidation_input)
        runtime = LocalConsolidatorRuntime(
            model=self.model,
            backend=self.backend,
            max_iterations=self.max_iterations,
            tokenizer_model=self.tokenizer_model,
            max_depth=self.max_depth,
            max_subcalls=self.max_subcalls,
            subcall_model=self.subcall_model,
            context_variable_name="bundle",
            context_aliases=("context",),
        )
        response = runtime.run(
            build_consolidation_bundle(consolidation_input),
            root_prompt=_build_agentic_root_prompt(),
            system_prompt=build_agentic_consolidator_system_prompt(),
        )
        return merge_consolidation_plans(
            baseline_plan,
            parse_consolidation_plan_response(response),
        )


def build_consolidator_system_prompt() -> str:
    """Return the single-shot system prompt for model-backed consolidation."""
    return (
        "You are Tokentrim's offline memory consolidator. "
        "Read immutable traces plus session, user, and org memories. "
        "Produce only a structured durable-memory edit plan. "
        "Never write session memory. "
        "Never mutate traces. "
        "You may infer durable memories from traces alone even when all memory scopes are empty. "
        "Promote only high-signal facts, repeated workflow patterns, stable preferences, "
        "and validated failure-to-recovery knowledge with provenance."
    )


def build_agentic_consolidator_system_prompt() -> str:
    """Return the system prompt for the iterative consolidator runtime."""
    return (
        "You are Tokentrim's offline memory consolidator agent. "
        "The completed offline bundle is loaded in the REPL variable `bundle` as a structured object. "
        "Use `bundle.latest_messages(n)`, `bundle.message_slice(start, end)`, `bundle.latest_traces(n)`, "
        "`bundle.trace(i)`, `bundle.grep(pattern, target=\"all\")`, `bundle.peek(obj, limit=None)`, "
        "`bundle.to_text(obj, limit=None)`, `bundle_query(query, target=None)`, `SHOW_VARS()`, `FINAL(...)`, and "
        "`FINAL_VAR(variable_name)` to inspect the bundle incrementally. "
        "You may infer durable memories from traces alone even when all memory scopes are empty. "
        "You must think through promotions, merges, archives, and no-op decisions before returning. "
        "When you are ready to finish, either return the final JSON object directly in plain text "
        "or call FINAL(...) / FINAL_VAR(...). "
        "The final output must be a JSON object matching the durable-memory edit plan shape. "
        "Never write session memory. Never mutate traces. Prefer no-op over weak memories."
    )


def serialize_consolidation_input(consolidation_input: ConsolidationInput) -> dict[str, object]:
    """Serialize the offline bundle into a stable LLM payload."""
    return {
        "session_id": consolidation_input.session_id,
        "user_id": consolidation_input.user_id,
        "org_id": consolidation_input.org_id,
        "traces": [_serialize_trace(trace) for trace in consolidation_input.traces if isinstance(trace, TokentrimTraceRecord)],
        "session_memories": [_serialize_memory(memory) for memory in consolidation_input.session_memories],
        "user_memories": [_serialize_memory(memory) for memory in consolidation_input.user_memories],
        "org_memories": [_serialize_memory(memory) for memory in consolidation_input.org_memories],
    }


def parse_consolidation_plan_response(payload: str) -> ConsolidationPlan:
    """Validate and normalize a model response into a consolidation plan."""
    try:
        parsed = json.loads(_strip_json_fence(payload))
    except json.JSONDecodeError as exc:
        raise TokentrimError("Consolidator agent did not return valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise TokentrimError("Consolidator agent response must be a JSON object.")

    return ConsolidationPlan(
        user_upserts=_parse_upserts(parsed.get("user_upserts"), expected_scope="user"),
        org_upserts=_parse_upserts(parsed.get("org_upserts"), expected_scope="org"),
        user_archives=_parse_archives(parsed.get("user_archives")),
        org_archives=_parse_archives(parsed.get("org_archives")),
        merge_operations=_parse_merges(parsed.get("merge_operations")),
        rationale=_parse_string_list(parsed.get("rationale")),
        source_refs=_parse_string_list(parsed.get("source_refs")),
    )


def _build_user_prompt(payload: Mapping[str, object]) -> str:
    return (
        "Produce a durable-memory edit plan for this completed session bundle.\n"
        "Return JSON only.\n"
        "Rules:\n"
        "- write only user/org memory edits\n"
        "- keep session memory untouched\n"
        "- include rationale and source_refs\n"
        "- infer durable memories from traces even if session/user/org memory are empty\n"
        "- prefer no-op over weak memories\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _build_agentic_root_prompt() -> str:
    return (
        "Build the final durable-memory edit plan from `bundle`.\n"
        "Inspect traces and memories before deciding.\n"
        "When done, return the final JSON object directly, or use FINAL(...)/FINAL_VAR(...), "
        "and include these keys: "
        "user_upserts, org_upserts, user_archives, org_archives, merge_operations, rationale, source_refs."
    )


def _build_response_format() -> dict[str, Any]:
    """Return a strict JSON schema for consolidation responses."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tokentrim_consolidation_plan",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "user_upserts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "scope": {"type": "string"},
                                "subject_id": {"type": "string"},
                                "memory_id": {"type": ["string", "null"]},
                                "write": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "content": {"type": "string"},
                                        "kind": {"type": "string"},
                                        "salience": {"type": "number"},
                                        "dedupe_key": {"type": ["string", "null"]},
                                        "metadata": {"type": ["object", "null"]},
                                        "source_refs": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["content", "kind", "source_refs"],
                                },
                            },
                            "required": ["subject_id", "write"],
                        },
                    },
                    "org_upserts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "scope": {"type": "string"},
                                "subject_id": {"type": "string"},
                                "memory_id": {"type": ["string", "null"]},
                                "write": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "content": {"type": "string"},
                                        "kind": {"type": "string"},
                                        "salience": {"type": "number"},
                                        "dedupe_key": {"type": ["string", "null"]},
                                        "metadata": {"type": ["object", "null"]},
                                        "source_refs": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["content", "kind", "source_refs"],
                                },
                            },
                            "required": ["subject_id", "write"],
                        },
                    },
                    "user_archives": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "memory_id": {"type": "string"},
                                "reason": {"type": ["string", "null"]},
                            },
                            "required": ["memory_id"],
                        },
                    },
                    "org_archives": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "memory_id": {"type": "string"},
                                "reason": {"type": ["string", "null"]},
                            },
                            "required": ["memory_id"],
                        },
                    },
                    "merge_operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "target_memory_id": {"type": "string"},
                                "source_memory_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "reason": {"type": ["string", "null"]},
                            },
                            "required": ["target_memory_id", "source_memory_ids"],
                        },
                    },
                    "rationale": {"type": "array", "items": {"type": "string"}},
                    "source_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "user_upserts",
                    "org_upserts",
                    "user_archives",
                    "org_archives",
                    "merge_operations",
                    "rationale",
                    "source_refs",
                ],
            },
        },
    }


def _resolve_model_name(*, backend: str, model: str) -> str:
    normalized_backend = backend.strip().lower()
    normalized_model = model.strip()
    if not normalized_model:
        raise TokentrimError("ModelConsolidatorAgent requires a non-empty model.")
    if "/" in normalized_model:
        return normalized_model
    return f"{normalized_backend}/{normalized_model}"


def _serialize_trace(trace: TokentrimTraceRecord) -> dict[str, object]:
    return {
        "trace_id": trace.trace_id,
        "source": trace.source,
        "workflow_name": trace.workflow_name,
        "started_at": trace.started_at,
        "ended_at": trace.ended_at,
        "metadata": trace.metadata,
        "spans": [
            {
                "span_id": span.span_id,
                "kind": span.kind,
                "name": span.name,
                "parent_id": span.parent_id,
                "started_at": span.started_at,
                "ended_at": span.ended_at,
                "error": span.error,
                "metrics": span.metrics,
                "data": span.data,
            }
            for span in trace.spans
        ],
    }


def _serialize_memory(memory: MemoryRecord) -> dict[str, object]:
    return {
        "memory_id": memory.memory_id,
        "scope": memory.scope,
        "subject_id": memory.subject_id,
        "kind": memory.kind,
        "content": memory.content,
        "salience": memory.salience,
        "status": memory.status,
        "source_refs": list(memory.source_refs),
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "dedupe_key": memory.dedupe_key,
        "metadata": dict(memory.metadata) if memory.metadata is not None else None,
    }


def _parse_upserts(payload: object, *, expected_scope: str) -> tuple[MemoryUpsert, ...]:
    """Parse validated upsert operations for one durable scope."""
    items = _ensure_sequence(payload, field_name=f"{expected_scope}_upserts")
    upserts: list[MemoryUpsert] = []
    for item in items:
        mapping = _ensure_mapping(item, field_name=f"{expected_scope}_upsert")
        scope = mapping.get("scope", expected_scope)
        if scope != expected_scope:
            raise TokentrimError(f"Consolidator returned invalid scope '{scope}' in {expected_scope}_upserts.")
        write_mapping = _ensure_mapping(mapping.get("write"), field_name="write")
        upserts.append(
            MemoryUpsert(
                scope=expected_scope,
                subject_id=_require_string(mapping.get("subject_id"), field_name="subject_id"),
                memory_id=_optional_string(mapping.get("memory_id"), field_name="memory_id"),
                write=MemoryWrite(
                    content=_require_string(write_mapping.get("content"), field_name="write.content"),
                    kind=_require_string(write_mapping.get("kind"), field_name="write.kind"),
                    salience=_optional_float(write_mapping.get("salience"), default=0.5, field_name="write.salience"),
                    dedupe_key=_optional_string(write_mapping.get("dedupe_key"), field_name="write.dedupe_key"),
                    metadata=_optional_mapping(write_mapping.get("metadata"), field_name="write.metadata"),
                    source_refs=_parse_string_list(write_mapping.get("source_refs")),
                ),
            )
        )
    return tuple(upserts)


def _parse_archives(payload: object) -> tuple[MemoryArchive, ...]:
    items = _ensure_sequence(payload, field_name="archives")
    archives: list[MemoryArchive] = []
    for item in items:
        mapping = _ensure_mapping(item, field_name="archive")
        archives.append(
            MemoryArchive(
                memory_id=_require_string(mapping.get("memory_id"), field_name="memory_id"),
                reason=_optional_string(mapping.get("reason"), field_name="reason"),
            )
        )
    return tuple(archives)


def _parse_merges(payload: object) -> tuple[MemoryMerge, ...]:
    """Parse merge operations that archive redundant durable memories."""
    items = _ensure_sequence(payload, field_name="merge_operations")
    merges: list[MemoryMerge] = []
    for item in items:
        mapping = _ensure_mapping(item, field_name="merge")
        merges.append(
            MemoryMerge(
                target_memory_id=_require_string(mapping.get("target_memory_id"), field_name="target_memory_id"),
                source_memory_ids=_parse_string_list(mapping.get("source_memory_ids")),
                reason=_optional_string(mapping.get("reason"), field_name="reason"),
            )
        )
    return tuple(merges)


def _parse_string_list(payload: object) -> tuple[str, ...]:
    if payload is None:
        return ()
    if isinstance(payload, str):
        return (_require_string(payload, field_name="string_list_item"),)
    items = _ensure_sequence(payload, field_name="string_list")
    return tuple(_require_string(item, field_name="string_list_item") for item in items)


def _strip_json_fence(payload: str) -> str:
    stripped = payload.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _ensure_mapping(payload: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TokentrimError(f"Consolidator field '{field_name}' must be an object.")
    return payload


def _optional_mapping(payload: object, *, field_name: str) -> Mapping[str, object] | None:
    if payload is None:
        return None
    return _ensure_mapping(payload, field_name=field_name)


def _ensure_sequence(payload: object, *, field_name: str) -> Sequence[object]:
    if payload is None:
        return ()
    if isinstance(payload, str) or not isinstance(payload, Sequence):
        raise TokentrimError(f"Consolidator field '{field_name}' must be an array.")
    return payload


def _require_string(payload: object, *, field_name: str) -> str:
    if not isinstance(payload, str) or not payload.strip():
        raise TokentrimError(f"Consolidator field '{field_name}' must be a non-empty string.")
    return payload.strip()


def _optional_string(payload: object, *, field_name: str) -> str | None:
    if payload is None:
        return None
    return _require_string(payload, field_name=field_name)


def _optional_float(payload: object, *, default: float, field_name: str) -> float:
    if payload is None:
        return default
    if not isinstance(payload, int | float):
        raise TokentrimError(f"Consolidator field '{field_name}' must be numeric.")
    return float(payload)
