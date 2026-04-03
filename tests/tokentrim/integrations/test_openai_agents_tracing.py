from __future__ import annotations

from collections.abc import AsyncIterator
import json

import pytest

agents = pytest.importorskip("agents")

from agents import Agent, Runner
from agents.items import ModelResponse, TResponseStreamEvent
from agents.models.interface import Model, ModelTracing
from agents.tracing import (
    agent_span,
    custom_span,
    function_span,
    generation_span,
    get_trace_provider,
    handoff_span,
    set_trace_processors,
    trace,
)
from agents.usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from openai.types.responses.response_prompt_param import ResponsePromptParam

from tokentrim import InMemoryTraceStore, Tokentrim
from tokentrim.integrations.openai_agents.translator import OpenAIAgentsTraceTranslator
from tokentrim.integrations.openai_agents.tracing import (
    TOKENTRIM_TRACE_CAPTURE_MODE,
    TOKENTRIM_TRACE_METADATA_KEY,
    TokentrimOpenAIIdentityProcessor,
    build_identity_trace_metadata,
    install_identity_processor,
)
from tokentrim.transforms import FilterMessages


class RecordingSession:
    def __init__(self, items: list[dict[str, str]] | None = None) -> None:
        self.session_id = "sdk-session"
        self._items = list(items or [])

    async def get_items(self, limit: int | None = None) -> list[dict[str, str]]:
        if limit is None:
            return list(self._items)
        return list(self._items[-limit:])

    async def add_items(self, items: list[dict[str, str]]) -> None:
        self._items.extend(items)

    async def pop_item(self):
        if not self._items:
            return None
        return self._items.pop()

    async def clear_session(self) -> None:
        self._items.clear()


class DummyModel(Model):
    async def get_response(
        self,
        system_instructions: str | None,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        del model_settings
        del tools
        del output_schema
        del handoffs
        del previous_response_id
        del conversation_id
        del prompt

        usage = Usage(requests=1, input_tokens=5, output_tokens=3, total_tokens=8)
        include_data = tracing.include_data()
        generation_input = input if include_data and isinstance(input, list) else None
        with generation_span(
            input=generation_input,
            model="dummy-model",
            model_config={"system_instructions": system_instructions},
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            },
            disabled=tracing.is_disabled(),
        ) as span:
            response = ModelResponse(
                output=[
                    ResponseOutputMessage(
                        id="msg_1",
                        content=[
                            ResponseOutputText(
                                annotations=[],
                                text="done",
                                type="output_text",
                            )
                        ],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ],
                usage=usage,
                response_id="resp_1",
            )
            if include_data:
                span.span_data.output = response.to_input_items()
            return response

    async def stream_response(
        self,
        system_instructions: str | None,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        del system_instructions
        del input
        del model_settings
        del tools
        del output_schema
        del handoffs
        del tracing
        del previous_response_id
        del conversation_id
        del prompt
        if False:
            yield None  # pragma: no cover
        raise NotImplementedError("stream_response is not used in these tests")


@pytest.fixture()
def isolated_trace_processors():
    provider = get_trace_provider()
    original_processors = list(provider._multi_processor._processors)
    set_trace_processors([])
    try:
        yield
    finally:
        set_trace_processors(original_processors)


def test_openai_trace_translator_normalizes_trace_metadata_and_ids() -> None:
    translator = OpenAIAgentsTraceTranslator()

    record = translator.translate_trace(
        {
            "id": "trace_123",
            "workflow_name": "Agent workflow",
            "group_id": "group_1",
            "metadata": {
                "team": "support",
                TOKENTRIM_TRACE_METADATA_KEY: json.dumps(
                    {
                        "capture_mode": TOKENTRIM_TRACE_CAPTURE_MODE,
                        "store_id": "store_1",
                        "user_id": "u1",
                        "session_id": "s1",
                    }
                ),
            },
        },
        user_id="u1",
        session_id="s1",
    )

    assert record.trace_id == "openai_agents:trace_123"
    assert record.source == "openai_agents"
    assert record.capture_mode == "identity"
    assert record.source_trace_id == "trace_123"
    assert record.user_id == "u1"
    assert record.session_id == "s1"
    assert record.metadata == {"team": "support"}
    assert (
        json.loads(record.raw_trace["metadata"][TOKENTRIM_TRACE_METADATA_KEY])["store_id"]
        == "store_1"
    )


def test_openai_trace_translator_normalizes_span_payloads() -> None:
    translator = OpenAIAgentsTraceTranslator()

    span = translator.translate_span(
        {
            "id": "span_123",
            "trace_id": "trace_123",
            "parent_id": "span_root",
            "started_at": "2026-04-03T10:00:00+00:00",
            "ended_at": "2026-04-03T10:00:01+00:00",
            "error": None,
            "span_data": {
                "type": "generation",
                "model": "dummy-model",
                "input": [{"role": "user", "content": "hello"}],
                "output": [{"role": "assistant", "content": "world"}],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                "model_config": {"temperature": 0},
            },
        }
    )

    assert span.span_id == "openai_agents:span_123"
    assert span.trace_id == "openai_agents:trace_123"
    assert span.parent_id == "openai_agents:span_root"
    assert span.kind == "generation"
    assert span.name == "dummy-model"
    assert span.source_span_id == "span_123"
    assert span.metrics == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    assert span.data == {
        "model": "dummy-model",
        "input": [{"role": "user", "content": "hello"}],
        "output": [{"role": "assistant", "content": "world"}],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        "model_config": {"temperature": 0},
    }
    assert span.raw_span["id"] == "span_123"


def test_openai_trace_translator_normalizes_tokentrim_transform_span_payloads() -> None:
    translator = OpenAIAgentsTraceTranslator()

    span = translator.translate_span(
        {
            "id": "span_transform",
            "trace_id": "trace_123",
            "parent_id": "span_root",
            "started_at": "2026-04-03T10:00:00+00:00",
            "ended_at": "2026-04-03T10:00:01+00:00",
            "error": None,
            "span_data": {
                "type": "custom",
                "name": "tokentrim.transform.compaction",
                "data": {
                    "kind": "transform",
                    "transform_name": "compaction",
                    "changed": True,
                    "token_budget": 30,
                    "input_items": 8,
                    "output_items": 3,
                    "input_tokens": 120,
                    "output_tokens": 48,
                },
            },
        }
    )

    assert span.kind == "transform"
    assert span.name == "compaction"
    assert span.parent_id == "openai_agents:span_root"
    assert span.metrics == {
        "input_tokens": 120,
        "output_tokens": 48,
        "input_items": 8,
        "output_items": 3,
    }
    assert span.data == {
        "transform_name": "compaction",
        "changed": True,
        "token_budget": 30,
    }
    assert span.raw_span["span_data"]["name"] == "tokentrim.transform.compaction"


def test_openai_trace_translator_generates_namespaced_ids_without_source_ids() -> None:
    translator = OpenAIAgentsTraceTranslator()

    trace_record = translator.translate_trace(
        {"workflow_name": "Agent workflow", "metadata": {}},
        user_id="u1",
        session_id="s1",
    )
    span_record = translator.translate_span({"span_data": {"type": "custom", "detail": "x"}})

    assert trace_record.trace_id.startswith("openai_agents:")
    assert trace_record.source_trace_id is None
    assert span_record.span_id.startswith("openai_agents:")
    assert span_record.trace_id.startswith("openai_agents:")
    assert span_record.source_span_id is None
    assert span_record.parent_id is None


def test_identity_processor_persists_openai_export_payloads(isolated_trace_processors) -> None:
    store = InMemoryTraceStore()
    metadata = build_identity_trace_metadata(
        {},
        store_id=install_identity_processor(store),
        user_id="u1",
        session_id="s1",
    )

    with trace("Agent workflow", metadata=metadata):
        with agent_span(name="assistant"):
            with custom_span(
                "tokentrim.transform.filter",
                {
                    "kind": "transform",
                    "transform_name": "filter",
                    "changed": False,
                    "input_items": 1,
                    "output_items": 1,
                    "input_tokens": 3,
                    "output_tokens": 3,
                },
            ):
                pass
            with generation_span(
                input=[{"role": "user", "content": "hello"}],
                output=[{"role": "assistant", "content": "world"}],
                model="dummy-model",
                usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            ):
                pass
            with function_span(name="lookup", input="{}", output="done"):
                pass
            with handoff_span(from_agent="assistant", to_agent="worker"):
                pass

    traces = store.list_traces(user_id="u1", session_id="s1")

    assert len(traces) == 1
    assert traces[0].trace_id.startswith("openai_agents:trace_")
    assert traces[0].source == "openai_agents"
    assert traces[0].capture_mode == "identity"
    assert traces[0].user_id == "u1"
    assert traces[0].session_id == "s1"
    assert traces[0].workflow_name == "Agent workflow"
    assert TOKENTRIM_TRACE_METADATA_KEY not in (traces[0].metadata or {})
    assert json.loads(traces[0].raw_trace["metadata"][TOKENTRIM_TRACE_METADATA_KEY])[
        "capture_mode"
    ] == (
        TOKENTRIM_TRACE_CAPTURE_MODE
    )
    assert [span.kind for span in traces[0].spans] == [
        "agent",
        "transform",
        "generation",
        "function",
        "handoff",
    ]
    assert traces[0].spans[0].data == {"name": "assistant"}
    assert traces[0].spans[1].name == "filter"
    assert traces[0].spans[1].metrics == {
        "input_tokens": 3,
        "output_tokens": 3,
        "input_items": 1,
        "output_items": 1,
    }
    assert traces[0].spans[1].data == {"transform_name": "filter", "changed": False}
    assert traces[0].spans[2].metrics == {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    assert traces[0].spans[3].data == {"name": "lookup", "input": "{}", "output": "done"}
    assert traces[0].spans[4].name == "assistant -> worker"


def test_identity_processor_ignores_traces_without_tokentrim_metadata(
    isolated_trace_processors,
) -> None:
    processor = TokentrimOpenAIIdentityProcessor()
    set_trace_processors([processor])
    store = InMemoryTraceStore()
    install_identity_processor(store)

    with trace("Ignored workflow"):
        with agent_span(name="assistant"):
            pass

    assert store.list_traces(user_id="u1", session_id="s1") == ()


def test_tokentrim_openai_run_persists_one_trace(isolated_trace_processors) -> None:
    client = Tokentrim()
    trace_store = InMemoryTraceStore()
    run_config = client.openai_agents_config(
        user_id="u1",
        session_id="s1",
        trace_store=trace_store,
    )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Be brief.", model=DummyModel()),
        "hello",
        run_config=run_config,
    )

    assert result.final_output == "done"
    traces = trace_store.list_traces(user_id="u1", session_id="s1")
    assert len(traces) == 1
    assert traces[0].source == "openai_agents"
    assert traces[0].workflow_name == run_config.workflow_name
    assert [span.kind for span in traces[0].spans] == ["agent", "generation"]
    assert traces[0].spans[1].metrics == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}


def test_tokentrim_openai_run_persists_transform_span_for_pipeline_step(
    isolated_trace_processors,
) -> None:
    client = Tokentrim()
    trace_store = InMemoryTraceStore()
    run_config = client.openai_agents_config(
        steps=(FilterMessages(),),
        user_id="u1",
        session_id="s1",
        trace_store=trace_store,
    )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Be brief.", model=DummyModel()),
        "hello",
        run_config=run_config,
    )

    assert result.final_output == "done"
    traces = trace_store.list_traces(user_id="u1", session_id="s1")
    assert len(traces) == 1
    transform_spans = [span for span in traces[0].spans if span.kind == "transform"]
    assert len(transform_spans) == 1
    transform_span = transform_spans[0]
    assert transform_span.name == "filter"
    assert transform_span.data == {"transform_name": "filter", "changed": False}
    assert transform_span.metrics is not None
    assert transform_span.metrics["input_items"] == 1
    assert transform_span.metrics["output_items"] == 1


def test_session_history_callback_does_not_create_extra_identity_traces(
    isolated_trace_processors,
) -> None:
    client = Tokentrim()
    trace_store = InMemoryTraceStore()
    run_config = client.openai_agents_config(
        user_id="u1",
        session_id="s1",
        trace_store=trace_store,
        apply_to_session_history=True,
    )
    session = RecordingSession(items=[{"role": "user", "content": "old"}])

    Runner.run_sync(
        Agent(name="assistant", instructions="Be brief.", model=DummyModel()),
        "new",
        run_config=run_config,
        session=session,
    )

    traces = trace_store.list_traces(user_id="u1", session_id="s1")
    assert len(traces) == 1
