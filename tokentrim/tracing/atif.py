from __future__ import annotations

from typing import Any

from tokentrim.tracing.records import TokentrimSpanRecord, TokentrimTraceRecord

ATIF_SCHEMA_VERSION = "ATIF-v1.4"


def export_trace_to_atif(trace: TokentrimTraceRecord) -> dict[str, Any]:
    steps = [_span_to_atif_step(index=index, span=span) for index, span in enumerate(trace.spans, start=1)]
    final_metrics = _build_final_metrics(trace.spans)
    payload: dict[str, Any] = {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": trace.session_id,
        "agent": {
            "name": "tokentrim",
            "version": "0.1.0",
            "model_name": trace.workflow_name,
            "extra": {
                "trace_id": trace.trace_id,
                "source": trace.source,
                "capture_mode": trace.capture_mode,
            },
        },
        "steps": steps,
        "extra": {
            "tokentrim_trace": _serialize_trace_record(trace),
        },
    }
    if final_metrics is not None:
        payload["final_metrics"] = final_metrics
    return payload


def load_trace_from_atif(payload: dict[str, Any]) -> TokentrimTraceRecord:
    """Load a Tokentrim trace from an ATIF export wrapper.

    The current implementation treats ATIF as an interoperability envelope.
    Round-trip loading relies on `extra.tokentrim_trace` rather than
    reconstructing canonical traces from the ATIF step body alone.
    """
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise ValueError("ATIF payload missing extra metadata")
    serialized = extra.get("tokentrim_trace")
    if not isinstance(serialized, dict):
        raise ValueError("ATIF payload missing tokentrim_trace metadata")
    return _deserialize_trace_record(serialized)


def _span_to_atif_step(*, index: int, span: TokentrimSpanRecord) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_id": index,
        "timestamp": span.started_at or span.ended_at,
        "source": "agent",
        "message": span.name or span.kind,
        "extra": {
            "tokentrim_span": _serialize_span_record(span),
            "source": span.source,
            "kind": span.kind,
        },
    }
    if span.metrics:
        metrics: dict[str, Any] = {}
        if "prompt_tokens" in span.metrics:
            metrics["prompt_tokens"] = span.metrics["prompt_tokens"]
        if "completion_tokens" in span.metrics:
            metrics["completion_tokens"] = span.metrics["completion_tokens"]
        if "cached_tokens" in span.metrics:
            metrics["cached_tokens"] = span.metrics["cached_tokens"]
        if metrics:
            step["metrics"] = metrics
    if span.error is not None:
        step["observation"] = {
            "results": [
                {
                    "source_call_id": span.span_id,
                    "content": f"Error: {span.error}",
                }
            ]
        }
    return step


def _build_final_metrics(spans: tuple[TokentrimSpanRecord, ...]) -> dict[str, Any] | None:
    prompt = 0
    completion = 0
    cached = 0
    saw_metric = False
    for span in spans:
        metrics = span.metrics or {}
        if "prompt_tokens" in metrics:
            prompt += metrics["prompt_tokens"]
            saw_metric = True
        if "completion_tokens" in metrics:
            completion += metrics["completion_tokens"]
            saw_metric = True
        if "cached_tokens" in metrics:
            cached += metrics["cached_tokens"]
            saw_metric = True
    if not saw_metric:
        return {"total_steps": len(spans)}
    return {
        "total_prompt_tokens": prompt,
        "total_completion_tokens": completion,
        "total_cached_tokens": cached,
        "total_steps": len(spans),
    }


def _serialize_trace_record(trace: TokentrimTraceRecord) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "source": trace.source,
        "capture_mode": trace.capture_mode,
        "source_trace_id": trace.source_trace_id,
        "user_id": trace.user_id,
        "session_id": trace.session_id,
        "workflow_name": trace.workflow_name,
        "started_at": trace.started_at,
        "ended_at": trace.ended_at,
        "group_id": trace.group_id,
        "metadata": trace.metadata,
        "raw_trace": trace.raw_trace,
        "spans": [_serialize_span_record(span) for span in trace.spans],
    }


def _serialize_span_record(span: TokentrimSpanRecord) -> dict[str, Any]:
    return {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "source": span.source,
        "kind": span.kind,
        "name": span.name,
        "source_span_id": span.source_span_id,
        "parent_id": span.parent_id,
        "started_at": span.started_at,
        "ended_at": span.ended_at,
        "error": span.error,
        "metrics": span.metrics,
        "data": span.data,
        "raw_span": span.raw_span,
    }


def _deserialize_trace_record(payload: dict[str, Any]) -> TokentrimTraceRecord:
    return TokentrimTraceRecord(
        trace_id=str(payload["trace_id"]),
        source=str(payload["source"]),
        capture_mode=str(payload["capture_mode"]),
        source_trace_id=_optional_string(payload.get("source_trace_id")),
        user_id=str(payload["user_id"]),
        session_id=str(payload["session_id"]),
        workflow_name=str(payload["workflow_name"]),
        started_at=_optional_string(payload.get("started_at")),
        ended_at=_optional_string(payload.get("ended_at")),
        group_id=_optional_string(payload.get("group_id")),
        metadata=_optional_dict(payload.get("metadata")),
        raw_trace=_required_dict(payload.get("raw_trace")),
        spans=tuple(_deserialize_span_record(span) for span in payload.get("spans", ())),
    )


def _deserialize_span_record(payload: dict[str, Any]) -> TokentrimSpanRecord:
    return TokentrimSpanRecord(
        span_id=str(payload["span_id"]),
        trace_id=str(payload["trace_id"]),
        source=str(payload["source"]),
        kind=str(payload["kind"]),
        name=_optional_string(payload.get("name")),
        source_span_id=_optional_string(payload.get("source_span_id")),
        parent_id=_optional_string(payload.get("parent_id")),
        started_at=_optional_string(payload.get("started_at")),
        ended_at=_optional_string(payload.get("ended_at")),
        error=_optional_dict(payload.get("error")),
        metrics=_optional_int_dict(payload.get("metrics")),
        data=_required_dict(payload.get("data")),
        raw_span=_required_dict(payload.get("raw_span")),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_dict(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Expected dict")
    return dict(value)


def _required_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Expected dict")
    return dict(value)


def _optional_int_dict(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Expected int dict")
    return {str(key): int(metric) for key, metric in value.items()}
