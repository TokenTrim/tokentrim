from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from tokentrim.integrations.openai_agents.trace_constants import (
    TOKENTRIM_TRACE_CAPTURE_MODE,
    TOKENTRIM_TRACE_METADATA_KEY,
)
from tokentrim.tracing import TOKENTRIM_TRANSFORM_SPAN_NAME_PREFIX, TOKENTRIM_TRANSFORM_SPAN_KIND
from tokentrim.tracing.records import TokentrimSpanRecord, TokentrimTraceRecord, build_canonical_id
from tokentrim.tracing.translator import TraceTranslator


class OpenAIAgentsTraceTranslator(TraceTranslator):
    source = "openai_agents"
    capture_mode = TOKENTRIM_TRACE_CAPTURE_MODE

    def translate_trace(
        self,
        payload: Mapping[str, Any],
        *,
        user_id: str,
        session_id: str,
    ) -> TokentrimTraceRecord:
        trace_id, source_trace_id = build_canonical_id(
            source=self.source,
            source_id=payload.get("id") if isinstance(payload.get("id"), str) else None,
        )
        return TokentrimTraceRecord(
            trace_id=trace_id,
            source=self.source,
            capture_mode=self.capture_mode,
            source_trace_id=source_trace_id,
            user_id=user_id,
            session_id=session_id,
            workflow_name=str(payload.get("workflow_name") or ""),
            group_id=payload.get("group_id") if isinstance(payload.get("group_id"), str) else None,
            started_at=None,
            ended_at=None,
            metadata=_normalize_trace_metadata(payload.get("metadata")),
            spans=tuple(),
            raw_trace=deepcopy(dict(payload)),
        )

    def translate_span(self, payload: Mapping[str, Any]) -> TokentrimSpanRecord:
        span_data = dict(payload.get("span_data") or {})
        kind = _normalize_span_kind(span_data)
        span_id, source_span_id = build_canonical_id(
            source=self.source,
            source_id=payload.get("id") if isinstance(payload.get("id"), str) else None,
        )
        trace_id, _ = build_canonical_id(
            source=self.source,
            source_id=payload.get("trace_id") if isinstance(payload.get("trace_id"), str) else None,
        )
        parent_id = _canonical_parent_id(payload.get("parent_id"))

        return TokentrimSpanRecord(
            span_id=span_id,
            trace_id=trace_id,
            source=self.source,
            kind=kind,
            name=_normalize_span_name(kind, span_data),
            source_span_id=source_span_id,
            parent_id=parent_id,
            started_at=payload.get("started_at")
            if isinstance(payload.get("started_at"), str)
            else None,
            ended_at=payload.get("ended_at") if isinstance(payload.get("ended_at"), str) else None,
            error=deepcopy(payload["error"]) if isinstance(payload.get("error"), dict) else None,
            metrics=_extract_metrics(kind, span_data),
            data=_normalize_span_data(kind, span_data),
            raw_span=deepcopy(dict(payload)),
        )


def _normalize_trace_metadata(metadata: object) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None

    normalized = {
        key: deepcopy(value)
        for key, value in metadata.items()
        if key != TOKENTRIM_TRACE_METADATA_KEY
    }
    return normalized or None


def _canonical_parent_id(parent_id: object) -> str | None:
    if not isinstance(parent_id, str) or not parent_id:
        return None
    return f"{OpenAIAgentsTraceTranslator.source}:{parent_id}"


def _normalize_span_kind(span_data: Mapping[str, Any]) -> str:
    raw_kind = span_data.get("type") if isinstance(span_data.get("type"), str) else "custom"
    if raw_kind != "custom":
        return raw_kind
    if _is_tokentrim_transform_span(span_data):
        return TOKENTRIM_TRANSFORM_SPAN_KIND
    return raw_kind


def _normalize_span_name(kind: str, span_data: Mapping[str, Any]) -> str | None:
    if kind == TOKENTRIM_TRANSFORM_SPAN_KIND:
        custom_data = _custom_span_data(span_data)
        if isinstance(custom_data.get("transform_name"), str):
            return custom_data["transform_name"]
        if isinstance(span_data.get("name"), str) and span_data["name"].startswith(
            TOKENTRIM_TRANSFORM_SPAN_NAME_PREFIX
        ):
            return span_data["name"][len(TOKENTRIM_TRANSFORM_SPAN_NAME_PREFIX) :] or None
    if isinstance(span_data.get("name"), str):
        return span_data["name"]
    if kind == "generation" and isinstance(span_data.get("model"), str):
        return span_data["model"]
    if kind == "handoff":
        from_agent = span_data.get("from_agent")
        to_agent = span_data.get("to_agent")
        if isinstance(from_agent, str) and isinstance(to_agent, str):
            return f"{from_agent} -> {to_agent}"
        if isinstance(to_agent, str):
            return to_agent
        if isinstance(from_agent, str):
            return from_agent
    if isinstance(span_data.get("type"), str):
        return span_data["type"]
    return None


def _normalize_span_data(kind: str, span_data: Mapping[str, Any]) -> dict[str, Any]:
    if kind == TOKENTRIM_TRANSFORM_SPAN_KIND:
        return _pick_fields(_custom_span_data(span_data), "transform_name", "changed", "token_budget")
    if kind == "agent":
        return _pick_fields(span_data, "name", "tools", "output_type")
    if kind == "generation":
        return _pick_fields(span_data, "model", "input", "output", "usage", "model_config")
    if kind == "function":
        return _pick_fields(span_data, "name", "input", "output")
    if kind == "handoff":
        return _pick_fields(span_data, "from_agent", "to_agent")

    return {
        key: deepcopy(value)
        for key, value in span_data.items()
        if key != "type"
    }


def _extract_metrics(kind: str, span_data: Mapping[str, Any]) -> dict[str, int] | None:
    if kind == TOKENTRIM_TRANSFORM_SPAN_KIND:
        custom_data = _custom_span_data(span_data)
        metrics = {
            key: value
            for key in ("input_tokens", "output_tokens", "total_tokens", "input_items", "output_items")
            if isinstance((value := custom_data.get(key)), int)
        }
        return metrics or None

    usage = span_data.get("usage")
    if not isinstance(usage, Mapping):
        return None

    metrics = {
        key: value
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if isinstance((value := usage.get(key)), int)
    }
    return metrics or None


def _is_tokentrim_transform_span(span_data: Mapping[str, Any]) -> bool:
    name = span_data.get("name")
    if not isinstance(name, str) or not name.startswith(TOKENTRIM_TRANSFORM_SPAN_NAME_PREFIX):
        return False
    custom_data = _custom_span_data(span_data)
    kind = custom_data.get("kind")
    transform_name = custom_data.get("transform_name")
    return kind == TOKENTRIM_TRANSFORM_SPAN_KIND or isinstance(transform_name, str)


def _custom_span_data(span_data: Mapping[str, Any]) -> Mapping[str, Any]:
    data = span_data.get("data")
    if isinstance(data, Mapping):
        return data
    return {}


def _pick_fields(span_data: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    picked: dict[str, Any] = {}
    for key in keys:
        if key in span_data and span_data[key] is not None:
            picked[key] = deepcopy(span_data[key])
    return picked
