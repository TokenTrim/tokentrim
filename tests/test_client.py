from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tokentrim import Tokentrim
from tokentrim.context.compaction import CompactionStep
from tokentrim.context.store import MemoryStore
from tokentrim.integrations.base import IntegrationAdapter
from tokentrim.tools.creator import ToolCreatorStep
from tokentrim.types.context_result import ContextResult
from tokentrim.types.step_trace import StepTrace
from tokentrim.types.tools_result import ToolsResult


class InMemoryStore:
    def retrieve(self, *, user_id: str, session_id: str) -> str | None:
        if user_id == "u1" and session_id == "s1":
            return "stored context"
        return None


class EchoAdapter(IntegrationAdapter[str]):
    def wrap(self, tokentrim: Tokentrim, config: str | None = None) -> str:
        assert isinstance(tokentrim, Tokentrim)
        return config or "default"


def _pipeline_step(steps, name):
    return next(step for step in steps if step.name == name)


def test_constructor_wires_per_feature_models() -> None:
    client = Tokentrim(
        model="shared-model",
        compaction_model="compact-model",
        tool_creation_model="creator-model",
    )

    compaction = _pipeline_step(client._context_pipeline._steps, "compaction")
    creator = _pipeline_step(client._tools_pipeline._steps, "creator")

    assert isinstance(compaction, CompactionStep)
    assert isinstance(creator, ToolCreatorStep)
    assert compaction._model == "compact-model"
    assert creator._model == "creator-model"
    assert client._context_pipeline._tokenizer_model == "shared-model"


def test_default_token_budget_propagates_to_context_request(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Tokentrim(model="shared-model", token_budget=123)
    captured = {}

    def fake_run(request):
        captured["token_budget"] = request.token_budget
        return ContextResult(
            messages=tuple(),
            step_traces=tuple(),
            token_count=0,
            trace_id="trace",
        )

    monkeypatch.setattr(client._context_pipeline, "run", fake_run)

    result = client.get_better_context([])

    assert captured["token_budget"] == 123
    assert result.trace_id == "trace"


def test_memory_store_is_propagated_to_context_pipeline() -> None:
    store: MemoryStore = InMemoryStore()
    client = Tokentrim(memory_store=store)

    rlm = _pipeline_step(client._context_pipeline._steps, "rlm")

    assert rlm._memory_store is store


def test_constructor_uses_shared_model_as_feature_fallback() -> None:
    client = Tokentrim(model="shared-model")

    compaction = _pipeline_step(client._context_pipeline._steps, "compaction")
    creator = _pipeline_step(client._tools_pipeline._steps, "creator")

    assert compaction._model == "shared-model"
    assert creator._model == "shared-model"


def test_per_call_token_budget_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Tokentrim(model="shared-model", token_budget=123)
    captured = {}

    def fake_run(request):
        captured["token_budget"] = request.token_budget
        return ToolsResult(
            tools=tuple(),
            step_traces=tuple(),
            token_count=0,
            trace_id="trace",
        )

    monkeypatch.setattr(client._tools_pipeline, "run", fake_run)

    result = client.get_better_tools([], token_budget=9)

    assert captured["token_budget"] == 9
    assert result.trace_id == "trace"


def test_wrap_integration_delegates_to_adapter() -> None:
    client = Tokentrim()

    result = client.wrap_integration(EchoAdapter(), config="wrapped")

    assert result == "wrapped"


def test_public_methods_return_frozen_result_objects() -> None:
    client = Tokentrim()

    context_result = client.get_better_context([])
    tools_result = client.get_better_tools([])

    with pytest.raises(FrozenInstanceError):
        context_result.trace_id = "other"

    with pytest.raises(FrozenInstanceError):
        tools_result.trace_id = "other"

    assert isinstance(context_result, ContextResult)
    assert isinstance(tools_result, ToolsResult)


def test_get_better_context_wires_filter_compaction_and_rlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Tokentrim(
        compaction_model="compact-model",
        memory_store=InMemoryStore(),
    )
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
        "tokentrim.context.compaction.generate_text",
        lambda **kwargs: "summary",
    )
    result = client.get_better_context(
        messages,
        user_id="u1",
        session_id="s1",
        token_budget=30,
        steps=("filter", "compaction", "rlm"),
    )

    assert isinstance(result.messages, tuple)
    assert isinstance(result.step_traces, tuple)
    assert result.trace_id
    assert result.messages[0] == {"role": "system", "content": "stored context"}
    assert result.messages[1] == {"role": "system", "content": "summary"}
    assert [trace.step_name for trace in result.step_traces] == [
        "filter",
        "compaction",
        "rlm",
    ]
    assert all(message["content"].strip() for message in result.messages)
    assert result.token_count > 0


def test_get_better_tools_wires_bpe_and_creator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Tokentrim(tool_creation_model="creator-model")
    tools = [
        {
            "name": "search",
            "description": "search   the   docs",
            "input_schema": {"type": "object"},
        }
    ]

    monkeypatch.setattr(
        "tokentrim.tools.creator.generate_text",
        lambda **kwargs: (
            '{"tools": ['
            '{"name": "search", "description": "duplicate", "input_schema": {}}, '
            '{"name": "lookup", "description": "new tool", "input_schema": {"type": "object"}}'
            "]}"
        ),
    )
    result = client.get_better_tools(
        tools,
        task_hint="investigate",
        steps=("bpe", "creator"),
    )

    assert isinstance(result.tools, tuple)
    assert isinstance(result.step_traces, tuple)
    assert result.trace_id
    assert result.tools[0]["description"] == "search the docs"
    assert [tool["name"] for tool in result.tools] == ["search", "lookup"]
    assert [trace.step_name for trace in result.step_traces] == ["bpe", "creator"]
    assert [trace.output_count - trace.input_count for trace in result.step_traces] == [0, 1]
    assert result.token_count > 0
