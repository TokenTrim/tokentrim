from __future__ import annotations

import importlib.util
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import tokentrim as tokentrim_module
import tokentrim.tracing as tracing_module

from tokentrim import Tokentrim
from tokentrim.errors.base import TokentrimError
from tokentrim.integrations.base import IntegrationAdapter
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.tracing import (
    InMemoryTraceStore,
    PipelineSpan,
    PipelineTracer,
    TokentrimTraceRecord,
)
from tokentrim.types.result import Result
from tokentrim.types.state import PipelineState
from tokentrim.types.trace import Trace
from tokentrim.transforms import CompactConversation, CompressToolDescriptions, CreateTools
from tokentrim.transforms import FilterMessages, RetrieveMemory
from tokentrim.transforms.base import Transform


def _seed_trace_store() -> InMemoryTraceStore:
    store = InMemoryTraceStore()
    store.create_trace(
        user_id="u1",
        session_id="s1",
        trace=TokentrimTraceRecord(
            trace_id="openai_agents:trace_1",
            source="openai_agents",
            capture_mode="identity",
            source_trace_id="trace_1",
            user_id="u1",
            session_id="s1",
            workflow_name="workflow-1",
            started_at=None,
            ended_at=None,
            group_id=None,
            metadata={"topic": "support"},
            raw_trace={"ignored": "raw"},
        ),
    )
    store.complete_trace(trace_id="openai_agents:trace_1")
    return store


def _install_fake_rlm(monkeypatch: pytest.MonkeyPatch, *, response: str) -> None:
    class FakeRLM:
        def __init__(self, **kwargs):
            del kwargs

        def completion(self, prompt, root_prompt=None):
            del prompt
            del root_prompt
            return SimpleNamespace(response=response)

        def close(self):
            return None

    monkeypatch.setattr(
        "tokentrim.transforms.rlm.transform.import_module",
        lambda name: SimpleNamespace(RLM=FakeRLM),
    )


class EchoAdapter(IntegrationAdapter[str]):
    def wrap(self, tokentrim: Tokentrim, config: str | None = None) -> str:
        assert isinstance(tokentrim, Tokentrim)
        return config or "default"


class RecordingPipelineSpan(PipelineSpan):
    def __init__(self, name: str, data: dict[str, object]) -> None:
        self.name = name
        self.data = dict(data)
        self.error: str | None = None

    def __enter__(self) -> PipelineSpan:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        del exc_type
        del exc_val
        del exc_tb
        return None

    def set_data(self, data) -> None:
        self.data = dict(data)

    def set_error(self, error: BaseException) -> None:
        self.error = error.__class__.__name__


class RecordingPipelineTracer(PipelineTracer):
    def __init__(self) -> None:
        self.spans: list[RecordingPipelineSpan] = []

    def start_span(self, *, name: str, data=None) -> PipelineSpan:
        span = RecordingPipelineSpan(name=name, data=dict(data or {}))
        self.spans.append(span)
        return span


class ExplodingTransform(Transform):
    @property
    def name(self) -> str:
        return "explode"

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        del state
        del request
        raise RuntimeError("boom")


def test_constructor_wires_tokenizer_model() -> None:
    client = Tokentrim(tokenizer="shared-model")

    assert client._pipeline._tokenizer_model == "shared-model"


def test_default_token_budget_propagates_to_compose_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Tokentrim(tokenizer="shared-model", token_budget=123)
    captured = {}

    def fake_run(request):
        captured["token_budget"] = request.token_budget
        return Result(
            context=tuple(),
            trace=Trace(
                id="trace",
                token_budget=request.token_budget,
                input_tokens=0,
                output_tokens=0,
                steps=tuple(),
            ),
        )

    monkeypatch.setattr(client._pipeline, "run", fake_run)

    result = client.compose(FilterMessages()).apply(context=[])

    assert captured["token_budget"] == 123
    assert result.trace.id == "trace"


def test_compose_apply_propagates_trace_store(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Tokentrim()
    trace_store = InMemoryTraceStore()
    captured = {}

    def fake_run(request):
        captured["trace_store"] = request.trace_store
        return Result(
            context=tuple(),
            trace=Trace(
                id="trace",
                token_budget=request.token_budget,
                input_tokens=0,
                output_tokens=0,
                steps=tuple(),
            ),
        )

    monkeypatch.setattr(client._pipeline, "run", fake_run)

    client.compose(FilterMessages()).apply(context=[], trace_store=trace_store)

    assert captured["trace_store"] is trace_store


def test_compose_apply_propagates_pipeline_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Tokentrim()
    pipeline_tracer = RecordingPipelineTracer()
    captured = {}

    def fake_run(request):
        captured["pipeline_tracer"] = request.pipeline_tracer
        return Result(
            context=tuple(),
            trace=Trace(
                id="trace",
                token_budget=request.token_budget,
                input_tokens=0,
                output_tokens=0,
                steps=tuple(),
            ),
        )

    monkeypatch.setattr(client._pipeline, "run", fake_run)

    client.compose(FilterMessages()).apply(context=[], pipeline_tracer=pipeline_tracer)

    assert captured["pipeline_tracer"] is pipeline_tracer


def test_rlm_store_belongs_to_rlm_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_store = _seed_trace_store()
    client = Tokentrim()
    messages = [{"role": "user", "content": "hello"}]
    _install_fake_rlm(monkeypatch, response="stored context")

    result = client.compose(RetrieveMemory(model="memory-model")).apply(
        messages,
        user_id="u1",
        session_id="s1",
        token_budget=10,
        trace_store=trace_store,
    )

    assert result.context[0] == {"role": "system", "content": "stored context"}


def test_constructor_does_not_define_per_transform_models() -> None:
    client = Tokentrim(tokenizer="shared-model")

    assert not hasattr(client._pipeline, "_compaction_model")
    assert not hasattr(client._pipeline, "_tool_creation_model")


def test_per_call_token_budget_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Tokentrim(tokenizer="shared-model", token_budget=123)
    captured = {}

    def fake_run(request):
        captured["token_budget"] = request.token_budget
        return Result(
            tools=tuple(),
            trace=Trace(
                id="trace",
                token_budget=request.token_budget,
                input_tokens=0,
                output_tokens=0,
                steps=tuple(),
            ),
        )

    monkeypatch.setattr(client._pipeline, "run", fake_run)

    result = client.compose(CompressToolDescriptions()).apply(tools=[], token_budget=9)

    assert captured["token_budget"] == 9
    assert result.trace.id == "trace"


def test_wrap_integration_delegates_to_adapter() -> None:
    client = Tokentrim()

    result = client.wrap_integration(EchoAdapter(), config="wrapped")

    assert result == "wrapped"


def test_public_tracing_exports_use_canonical_names() -> None:
    assert hasattr(tokentrim_module, "TokentrimTraceRecord")
    assert hasattr(tokentrim_module, "TokentrimSpanRecord")
    assert hasattr(tokentrim_module, "PipelineTracer")
    assert hasattr(tokentrim_module, "PipelineSpan")
    assert not hasattr(tokentrim_module, "IdentityTraceRecord")
    assert not hasattr(tokentrim_module, "IdentitySpanRecord")
    assert hasattr(tracing_module, "TokentrimTraceRecord")
    assert hasattr(tracing_module, "TokentrimSpanRecord")
    assert hasattr(tracing_module, "PipelineTracer")
    assert hasattr(tracing_module, "PipelineSpan")
    assert not hasattr(tracing_module, "IdentityTraceRecord")
    assert not hasattr(tracing_module, "IdentitySpanRecord")


def test_compose_apply_returns_frozen_result_objects() -> None:
    client = Tokentrim()

    context_result = client.compose(FilterMessages()).apply(context=[])
    tools_result = client.compose(CompressToolDescriptions()).apply(tools=[])

    with pytest.raises((FrozenInstanceError, TypeError)):
        context_result.trace.id = "other"

    with pytest.raises((FrozenInstanceError, TypeError)):
        tools_result.trace.id = "other"

    assert isinstance(context_result, Result)
    assert isinstance(tools_result, Result)


def test_unified_pipeline_emits_transform_span_via_pipeline_tracer() -> None:
    client = Tokentrim()
    pipeline_tracer = RecordingPipelineTracer()

    result = client.compose(FilterMessages()).apply(
        context=[
            {"role": "user", "content": "keep"},
            {"role": "assistant", "content": "   "},
        ],
        pipeline_tracer=pipeline_tracer,
    )

    assert [message["content"] for message in result.context] == ["keep"]
    assert len(pipeline_tracer.spans) == 1
    span = pipeline_tracer.spans[0]
    assert span.name == "tokentrim.transform.filter"
    assert span.error is None
    assert span.data == {
        "kind": "transform",
        "transform_name": "filter",
        "input_items": 2,
        "input_tokens": result.trace.steps[0].input_tokens,
        "output_items": 1,
        "output_tokens": result.trace.steps[0].output_tokens,
        "changed": True,
    }


def test_unified_pipeline_records_transform_span_errors() -> None:
    client = Tokentrim()
    pipeline_tracer = RecordingPipelineTracer()

    with pytest.raises(RuntimeError, match="boom"):
        client.compose(ExplodingTransform()).apply(
            context=[{"role": "user", "content": "hello"}],
            pipeline_tracer=pipeline_tracer,
        )

    assert len(pipeline_tracer.spans) == 1
    assert pipeline_tracer.spans[0].name == "tokentrim.transform.explode"
    assert pipeline_tracer.spans[0].error == "RuntimeError"


def test_compose_apply_context_wires_filter_compaction_and_rlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Tokentrim()
    messages = [
        {"role": "user", "content": "old-a " * 20},
        {"role": "assistant", "content": "old-b " * 20},
        {"role": "assistant", "content": "   "},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "5"},
        {"role": "assistant", "content": "6"},
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )
    _install_fake_rlm(monkeypatch, response="stored context")
    result = client.compose(
        FilterMessages(),
        CompactConversation(model="compact-model"),
        RetrieveMemory(model="memory-model"),
    ).apply(
        messages,
        user_id="u1",
        session_id="s1",
        token_budget=30,
        trace_store=_seed_trace_store(),
    )

    assert isinstance(result.context, tuple)
    assert isinstance(result.trace.steps, tuple)
    assert result.trace.id
    assert result.context[0] == {"role": "system", "content": "stored context"}
    assert result.context[1] == {"role": "system", "content": "summary"}
    assert [trace.step_name for trace in result.trace.steps] == [
        "filter",
        "compaction",
        "rlm",
    ]
    assert all(message["content"].strip() for message in result.context)
    assert result.trace.output_tokens > 0
    assert result.trace.input_tokens >= result.trace.output_tokens


def test_compose_apply_tools_wires_bpe_and_creator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Tokentrim()
    tools = [
        {
            "name": "search",
            "description": "search   the   docs",
            "input_schema": {"type": "object"},
        }
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.create_tools.transform.generate_text",
        lambda **kwargs: (
            '{"tools": ['
            '{"name": "search", "description": "duplicate", "input_schema": {}}, '
            '{"name": "lookup", "description": "new tool", "input_schema": {"type": "object"}}'
            "]}"
        ),
    )
    result = client.compose(
        CompressToolDescriptions(),
        CreateTools(model="creator-model"),
    ).apply(
        tools,
        task_hint="investigate",
    )

    assert isinstance(result.tools, tuple)
    assert isinstance(result.trace.steps, tuple)
    assert result.trace.id
    assert result.tools[0]["description"] == "search the docs"
    assert [tool["name"] for tool in result.tools] == ["search", "lookup"]
    assert [trace.step_name for trace in result.trace.steps] == ["bpe", "creator"]
    assert [trace.output_items - trace.input_items for trace in result.trace.steps] == [0, 1]
    assert result.trace.output_tokens > 0
    assert result.trace.output_tokens >= result.trace.input_tokens


def test_compose_apply_supports_mixed_context_and_tool_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Tokentrim()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "   "},
    ]
    tools = [
        {
            "name": "search",
            "description": "search   docs",
            "input_schema": {"type": "object"},
        }
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.create_tools.transform.generate_text",
        lambda **kwargs: '{"tools": [{"name": "lookup", "description": "new tool", "input_schema": {}}]}',
    )
    result = client.compose(
        FilterMessages(),
        CompressToolDescriptions(),
        CreateTools(model="creator-model"),
    ).apply(
        context=messages,
        tools=tools,
        task_hint="investigate",
    )

    assert result.context == ({"role": "user", "content": "hello"},)
    assert [tool["name"] for tool in result.tools] == ["search", "lookup"]
    assert [trace.step_name for trace in result.trace.steps] == ["filter", "bpe", "creator"]


def test_compose_apply_rejects_empty_payload_when_no_steps() -> None:
    client = Tokentrim()

    with pytest.raises(TokentrimError) as exc_info:
        client.compose().apply([])

    assert "cannot infer payload kind" in str(exc_info.value)


def test_compose_to_openai_agents_builds_run_config_for_any_pipeline() -> None:
    client = Tokentrim()

    if importlib.util.find_spec("agents") is None:
        with pytest.raises(TokentrimError) as exc_info:
            client.compose(CompressToolDescriptions()).to_openai_agents()
        assert "openai-agents is required" in str(exc_info.value)
        return

    wrapped = client.compose(CompressToolDescriptions()).to_openai_agents()

    assert wrapped.call_model_input_filter is not None
