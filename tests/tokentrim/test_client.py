from __future__ import annotations

import importlib.util
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import tokentrim as tokentrim_module
import tokentrim.tracing as tracing_module

from tokentrim import Tokentrim
from tokentrim.errors.base import TokentrimError
from tokentrim.integrations.base import IntegrationAdapter
from tokentrim.memory import FilesystemMemoryStore, InMemoryMemoryStore, MemoryWrite
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.tracing import (
    FilesystemTraceStore,
    InMemoryTraceStore,
    PipelineSpan,
    PipelineTracer,
    TokentrimTraceRecord,
)
from tokentrim.types.result import Result
from tokentrim.types.state import PipelineState
from tokentrim.types.tool import Tool
from tokentrim.types.trace import Trace
from tokentrim.transforms import CompactConversation
from tokentrim.transforms.base import Transform


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


class TestFilterTransform(Transform):
    @property
    def name(self) -> str:
        return "test-filter"

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        del request
        filtered = [message for message in state.context if message["content"].strip()]
        return PipelineState(context=filtered, tools=state.tools)


class TestToolTransform(Transform):
    @property
    def name(self) -> str:
        return "test-tools"

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        del request
        tools: list[Tool] = list(state.tools)
        if tools:
            tools[0] = {
                "name": tools[0]["name"],
                "description": "normalized",
                "input_schema": tools[0]["input_schema"],
            }
        tools.append(
            {
                "name": "lookup",
                "description": "new tool",
                "input_schema": {"type": "object"},
            }
        )
        return PipelineState(context=state.context, tools=tools)


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

    result = client.compose(TestFilterTransform()).apply(context=[])

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

    client.compose(TestFilterTransform()).apply(context=[], trace_store=trace_store)

    assert captured["trace_store"] is trace_store


def test_tokentrim_defaults_to_local_first_store_roots(tmp_path: Path) -> None:
    client = Tokentrim(storage_root=tmp_path / ".tokentrim")

    assert isinstance(client.default_memory_store, FilesystemMemoryStore)
    assert isinstance(client.default_trace_store, FilesystemTraceStore)
    assert client.memory_root == tmp_path / ".tokentrim" / "memory"
    assert client.trace_root == tmp_path / ".tokentrim" / "traces"


def test_compose_apply_uses_default_filesystem_stores_when_not_overridden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = Tokentrim(storage_root=tmp_path / ".tokentrim")
    captured = {}

    def fake_run(request):
        captured["memory_store"] = request.memory_store
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

    client.compose(TestFilterTransform()).apply(context=[], user_id="u1", session_id="s1", org_id="o1")

    assert captured["memory_store"] is client.default_memory_store
    assert captured["trace_store"] is client.default_trace_store


def test_compose_apply_does_not_force_default_stores_without_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = Tokentrim(storage_root=tmp_path / ".tokentrim")
    captured = {}

    def fake_run(request):
        captured["memory_store"] = request.memory_store
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

    client.compose(TestFilterTransform()).apply(context=[])

    assert captured["memory_store"] is None
    assert captured["trace_store"] is None


def test_compose_apply_propagates_memory_store_and_org_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Tokentrim()
    memory_store = InMemoryMemoryStore()
    captured = {}

    def fake_run(request):
        captured["memory_store"] = request.memory_store
        captured["org_id"] = request.org_id
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

    client.compose(TestFilterTransform()).apply(
        context=[],
        org_id="org_1",
        memory_store=memory_store,
    )

    assert captured["memory_store"] is memory_store
    assert captured["org_id"] == "org_1"


def test_compose_apply_auto_injects_memory_when_store_is_present() -> None:
    client = Tokentrim()
    memory_store = InMemoryMemoryStore()
    memory_store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(
            content="Use the repository root when debugging command failures",
            kind="active_state",
        ),
    )

    result = client.compose().apply(
        context=[{"role": "user", "content": "debug the command failure"}],
        user_id="user_1",
        session_id="sess_1",
        org_id="org_1",
        memory_store=memory_store,
        token_budget=500,
    )

    assert result.context[0]["role"] == "system"
    assert "Injected memory:" in str(result.context[0]["content"])
    assert "repository root" in str(result.context[0]["content"])


def test_compose_apply_can_enable_agent_aware_memory_mode() -> None:
    client = Tokentrim()
    memory_store = InMemoryMemoryStore()

    result = client.compose().apply(
        context=[{"role": "user", "content": "debug the command failure"}],
        user_id="user_1",
        session_id="sess_1",
        org_id="org_1",
        memory_store=memory_store,
        agent_aware_memory=True,
        token_budget=500,
    )

    assert result.context[0]["role"] == "system"
    assert "Session memory is available" in str(result.context[0]["content"])
    assert any(tool["name"] == "remember" for tool in result.tools)


def test_tokentrim_exposes_explicit_session_memory_write_helpers() -> None:
    client = Tokentrim()
    memory_store = InMemoryMemoryStore()

    record = client.write_session_memory(
        memory_store=memory_store,
        session_id="sess_1",
        content="Avoid destructive commands unless requested",
        kind="constraint",
        dedupe_key="avoid_destructive",
    )
    writer = client.session_memory_writer(memory_store=memory_store, session_id="sess_1")
    updated = writer.write(
        content="Avoid destructive commands without explicit approval",
        kind="constraint",
        dedupe_key="avoid_destructive",
    )

    assert record.memory_id == updated.memory_id
    assert updated.content == "Avoid destructive commands without explicit approval"


def test_tokentrim_session_memory_helpers_use_default_store_when_not_provided(tmp_path: Path) -> None:
    client = Tokentrim(storage_root=tmp_path / ".tokentrim")

    record = client.write_session_memory(
        session_id="sess_1",
        content="Avoid destructive commands unless requested",
        kind="constraint",
        dedupe_key="avoid_destructive",
    )
    updated = client.session_memory_writer(session_id="sess_1").write(
        content="Avoid destructive commands without explicit approval",
        kind="constraint",
        dedupe_key="avoid_destructive",
    )

    assert record.memory_id == updated.memory_id
    assert updated.content == "Avoid destructive commands without explicit approval"


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

    client.compose(TestFilterTransform()).apply(context=[], pipeline_tracer=pipeline_tracer)

    assert captured["pipeline_tracer"] is pipeline_tracer


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

    result = client.compose(TestToolTransform()).apply(tools=[], token_budget=9)

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


def test_public_top_level_exports_include_primary_transforms() -> None:
    assert hasattr(tokentrim_module, "CompactConversation")


def test_compose_apply_returns_frozen_result_objects() -> None:
    client = Tokentrim()

    context_result = client.compose(TestFilterTransform()).apply(context=[])
    tools_result = client.compose(TestToolTransform()).apply(tools=[])

    with pytest.raises((FrozenInstanceError, TypeError)):
        context_result.trace.id = "other"

    with pytest.raises((FrozenInstanceError, TypeError)):
        tools_result.trace.id = "other"

    assert isinstance(context_result, Result)
    assert isinstance(tools_result, Result)


def test_unified_pipeline_emits_transform_span_via_pipeline_tracer() -> None:
    client = Tokentrim()
    pipeline_tracer = RecordingPipelineTracer()

    result = client.compose(TestFilterTransform()).apply(
        context=[
            {"role": "user", "content": "keep"},
            {"role": "assistant", "content": "   "},
        ],
        pipeline_tracer=pipeline_tracer,
    )

    assert [message["content"] for message in result.context] == ["keep"]
    assert len(pipeline_tracer.spans) == 1
    span = pipeline_tracer.spans[0]
    assert span.name == "tokentrim.transform.test-filter"
    assert span.error is None
    assert span.data == {
        "kind": "transform",
        "transform_name": "test-filter",
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


def test_compose_apply_context_wires_compaction(
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
        {"role": "user", "content": "issue-five"},
        {"role": "assistant", "content": "6"},
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )
    original_message_count = len(messages)
    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.count_message_tokens",
        lambda counted_messages, tokenizer_model=None: (
            1_000 if len(counted_messages) == original_message_count else 10
        ),
    )
    result = client.compose(
        CompactConversation(model="compact-model"),
    ).apply(
        messages,
        user_id="u1",
        session_id="s1",
        token_budget=40,
    )

    assert isinstance(result.context, tuple)
    assert isinstance(result.trace.steps, tuple)
    assert result.trace.id
    assert result.context[0] == {"role": "system", "content": "History only.\n\nsummary"}
    assert result.context[-2:] == (
        {"role": "user", "content": "issue-five"},
        {"role": "assistant", "content": "6"},
    )
    assert [trace.step_name for trace in result.trace.steps] == ["inject_memory", "compaction"]
    assert all(message["content"].strip() for message in result.context)
    assert result.trace.output_tokens > 0


def test_compose_apply_tools_wires_local_tool_transform() -> None:
    client = Tokentrim()
    tools = [
        {
            "name": "search",
            "description": "search   the   docs",
            "input_schema": {"type": "object"},
        }
    ]

    result = client.compose(
        TestToolTransform(),
    ).apply(
        tools,
        task_hint="investigate",
    )

    assert isinstance(result.tools, tuple)
    assert isinstance(result.trace.steps, tuple)
    assert result.trace.id
    assert result.tools[0]["description"] == "normalized"
    assert [tool["name"] for tool in result.tools] == ["search", "lookup"]
    assert [trace.step_name for trace in result.trace.steps] == ["test-tools"]
    assert result.trace.output_tokens > 0
    assert result.trace.output_tokens >= result.trace.input_tokens


def test_compose_apply_supports_mixed_context_and_tool_steps() -> None:
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

    result = client.compose(
        TestFilterTransform(),
        TestToolTransform(),
    ).apply(
        context=messages,
        tools=tools,
        task_hint="investigate",
    )

    assert result.context == ({"role": "user", "content": "hello"},)
    assert [tool["name"] for tool in result.tools] == ["search", "lookup"]
    assert [trace.step_name for trace in result.trace.steps] == ["test-filter", "test-tools"]


def test_compose_apply_rejects_empty_payload_when_no_steps() -> None:
    client = Tokentrim()

    with pytest.raises(TokentrimError) as exc_info:
        client.compose().apply([])

    assert "cannot infer payload kind" in str(exc_info.value)


def test_compose_apply_rejects_mixed_positional_payload_entries() -> None:
    client = Tokentrim()

    with pytest.raises(TokentrimError) as exc_info:
        client.compose().apply(
            [
                {"role": "user", "content": "hello"},
                {"name": "search", "description": "docs", "input_schema": {"type": "object"}},
            ]
        )

    assert "must not mix messages and tools" in str(exc_info.value)


def test_compose_apply_rejects_invalid_positional_payload_entry_shape() -> None:
    client = Tokentrim()

    with pytest.raises(TokentrimError) as exc_info:
        client.compose().apply(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant"},
            ]
        )

    assert "must all be message-shaped or tool-shaped dicts" in str(exc_info.value)


def test_compose_to_openai_agents_builds_run_config_for_any_pipeline() -> None:
    client = Tokentrim()

    if importlib.util.find_spec("agents") is None:
        with pytest.raises(TokentrimError) as exc_info:
            client.compose(TestToolTransform()).to_openai_agents()
        assert "openai-agents is required" in str(exc_info.value)
        return

    wrapped = client.compose(TestToolTransform()).to_openai_agents()

    assert wrapped.call_model_input_filter is not None
