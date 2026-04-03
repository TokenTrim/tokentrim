from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tokentrim.errors.base import TokentrimError
from tokentrim.integrations.openai_agents.trace_constants import (
    TOKENTRIM_TRACE_CAPTURE_MODE,
    TOKENTRIM_TRACE_METADATA_KEY,
)
from tokentrim.integrations.openai_agents.translator import OpenAIAgentsTraceTranslator
from tokentrim.tracing import TraceStore

logger = logging.getLogger(__name__)

_TRACE_STORE_REGISTRY: dict[str, TraceStore] = {}
_TRACE_STORE_REGISTRY_LOCK = threading.Lock()
_IDENTITY_PROCESSOR: TokentrimOpenAIIdentityProcessor | None = None
_IDENTITY_PROCESSOR_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _TraceRouting:
    store_id: str
    trace_id: str


def install_identity_processor(trace_store: TraceStore) -> str:
    from agents.tracing import add_trace_processor, get_trace_provider

    store_id = register_trace_store(trace_store)
    with _IDENTITY_PROCESSOR_LOCK:
        global _IDENTITY_PROCESSOR
        if _IDENTITY_PROCESSOR is None:
            _IDENTITY_PROCESSOR = TokentrimOpenAIIdentityProcessor()
        provider = get_trace_provider()
        processors = getattr(getattr(provider, "_multi_processor", None), "_processors", ())
        if _IDENTITY_PROCESSOR not in processors:
            add_trace_processor(_IDENTITY_PROCESSOR)
    return store_id


def register_trace_store(trace_store: TraceStore) -> str:
    store_id = str(id(trace_store))
    with _TRACE_STORE_REGISTRY_LOCK:
        _TRACE_STORE_REGISTRY[store_id] = trace_store
    return store_id


def build_identity_trace_metadata(
    existing_metadata: Mapping[str, Any] | None,
    *,
    store_id: str,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    metadata = dict(existing_metadata or {})
    namespaced = metadata.get(TOKENTRIM_TRACE_METADATA_KEY)
    if namespaced is None:
        merged_namespace: dict[str, Any] = {}
    else:
        merged_namespace = _parse_routing_metadata_value(namespaced)
        if merged_namespace is None:
            raise TokentrimError(
                f"`trace_metadata[{TOKENTRIM_TRACE_METADATA_KEY!r}]` is reserved for Tokentrim."
            )

    merged_namespace.update(
        {
            "capture_mode": TOKENTRIM_TRACE_CAPTURE_MODE,
            "store_id": store_id,
            "user_id": user_id,
            "session_id": session_id,
        }
    )
    metadata[TOKENTRIM_TRACE_METADATA_KEY] = _serialize_routing_metadata_value(merged_namespace)
    return metadata


class TokentrimOpenAIIdentityProcessor:
    """Persists OpenAI Agents traces to a Tokentrim trace store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._translator = OpenAIAgentsTraceTranslator()
        self._source_trace_routing: dict[str, _TraceRouting] = {}

    def on_trace_start(self, trace: Any) -> None:
        try:
            payload = trace.export()
            if not isinstance(payload, dict):
                return

            routing = self._extract_routing(payload)
            if routing is None:
                return

            trace_record = self._translator.translate_trace(
                payload,
                user_id=routing["user_id"],
                session_id=routing["session_id"],
            )
            store = self._get_store(routing["store_id"])
            if store is None:
                return

            store.create_trace(
                user_id=routing["user_id"],
                session_id=routing["session_id"],
                trace=trace_record,
            )
            if trace_record.source_trace_id is None:
                return
            with self._lock:
                self._source_trace_routing[trace_record.source_trace_id] = _TraceRouting(
                    store_id=routing["store_id"],
                    trace_id=trace_record.trace_id,
                )
        except Exception:
            logger.exception("Failed to persist OpenAI trace start.")

    def on_trace_end(self, trace: Any) -> None:
        try:
            payload = trace.export()
            if not isinstance(payload, dict):
                return

            trace_id = payload.get("id")
            if not isinstance(trace_id, str):
                return

            routing = self._pop_routing_for_source_trace(trace_id)
            if routing is None:
                return
            store = self._get_store(routing.store_id)
            if store is None:
                return
            store.complete_trace(trace_id=routing.trace_id)
        except Exception:
            logger.exception("Failed to persist OpenAI trace end.")

    def on_span_start(self, span: Any) -> None:
        del span

    def on_span_end(self, span: Any) -> None:
        try:
            payload = span.export()
            if not isinstance(payload, dict):
                return

            trace_id = payload.get("trace_id")
            if not isinstance(trace_id, str):
                return

            routing = self._get_routing_for_source_trace(trace_id)
            if routing is None:
                return
            store = self._get_store(routing.store_id)
            if store is None:
                return
            span_record = self._translator.translate_span(payload)
            store.append_span(
                trace_id=routing.trace_id,
                span=span_record,
            )
        except Exception:
            logger.exception("Failed to persist OpenAI span.")

    def shutdown(self) -> None:
        return None

    def force_flush(self) -> None:
        return None

    def _get_routing_for_source_trace(self, trace_id: str) -> _TraceRouting | None:
        with self._lock:
            return self._source_trace_routing.get(trace_id)

    def _pop_routing_for_source_trace(self, trace_id: str) -> _TraceRouting | None:
        with self._lock:
            return self._source_trace_routing.pop(trace_id, None)

    def _get_store(self, store_id: str) -> TraceStore | None:
        with _TRACE_STORE_REGISTRY_LOCK:
            return _TRACE_STORE_REGISTRY.get(store_id)

    def _extract_routing(self, payload: Mapping[str, Any]) -> dict[str, str] | None:
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            return None

        namespaced = metadata.get(TOKENTRIM_TRACE_METADATA_KEY)
        routing = _parse_routing_metadata_value(namespaced)
        if routing is None:
            return None
        if routing.get("capture_mode") != TOKENTRIM_TRACE_CAPTURE_MODE:
            return None

        store_id = routing.get("store_id")
        user_id = routing.get("user_id")
        session_id = routing.get("session_id")
        if not isinstance(store_id, str):
            return None
        if not isinstance(user_id, str):
            return None
        if not isinstance(session_id, str):
            return None

        return {
            "store_id": store_id,
            "user_id": user_id,
            "session_id": session_id,
        }


def _serialize_routing_metadata_value(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), separators=(",", ":"), sort_keys=True)


def _parse_routing_metadata_value(value: object) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
