from __future__ import annotations

import importlib.util
from dataclasses import FrozenInstanceError

import pytest

from tokentrim import Tokentrim
from tokentrim.errors.base import TokentrimError
from tokentrim.integrations.base import IntegrationAdapter
from tokentrim.types.result import Result
from tokentrim.types.trace import Trace
from tokentrim.transforms import CompactConversation, CompressToolDescriptions, CreateTools
from tokentrim.transforms import FilterMessages, RetrieveMemory


class InMemoryStore:
    def retrieve(self, *, user_id: str, session_id: str) -> str | None:
        if user_id == "u1" and session_id == "s1":
            return "stored context"
        return None


class EchoAdapter(IntegrationAdapter[str]):
    def wrap(self, tokentrim: Tokentrim, config: str | None = None) -> str:
        assert isinstance(tokentrim, Tokentrim)
        return config or "default"


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


def test_rlm_store_belongs_to_rlm_transform() -> None:
    store = InMemoryStore()
    client = Tokentrim()
    messages = [{"role": "user", "content": "hello"}]

    result = client.compose(RetrieveMemory(memory_store=store)).apply(
        messages,
        user_id="u1",
        session_id="s1",
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
    result = client.compose(
        FilterMessages(),
        CompactConversation(model="compact-model"),
        RetrieveMemory(memory_store=InMemoryStore()),
    ).apply(
        messages,
        user_id="u1",
        session_id="s1",
        token_budget=30,
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
