from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

agents = pytest.importorskip("agents")

from agents import Agent, RunConfig
from agents.handoffs import HandoffInputData
from agents.run import CallModelData, ModelInputData

from tokentrim import Tokentrim
from tokentrim.transforms import CompactConversation, FilterMessages, RetrieveMemory
from tokentrim.integrations.base import IntegrationAdapter
from tokentrim.integrations.openai_agents import (
    OpenAIAgentsAdapter,
    OpenAIAgentsOptions,
)
from tokentrim.integrations.openai_agents.tracing import TOKENTRIM_TRACE_METADATA_KEY
from tokentrim.tracing import InMemoryTraceStore, TokentrimTraceRecord


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


def _message_items(count: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"message {index} " + ("x" * 40)} for index in range(count)]


def test_openai_agents_adapter_compacts_plain_text_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Tokentrim()
    wrapped = OpenAIAgentsAdapter(
        options=OpenAIAgentsOptions(
            token_budget=25,
            steps=(CompactConversation(model="compact-model"),),
        )
    ).wrap(client)
    input_items = [
        {"role": "user", "content": "old 0 " + ("x" * 80)},
        {"role": "assistant", "content": "old 1 " + ("x" * 80)},
        {"role": "user", "content": "old 2 " + ("x" * 80)},
        {"role": "assistant", "content": "old 3 " + ("x" * 80)},
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "a5"},
        {"role": "user", "content": "u6"},
        {"role": "assistant", "content": "a7"},
        {"role": "user", "content": "u8"},
        {"role": "assistant", "content": "a9"},
    ]
    payload = CallModelData(
        model_data=ModelInputData(
            input=input_items,
            instructions="answer briefly",
        ),
        agent=Agent(name="assistant"),
        context=None,
    )

    monkeypatch.setattr(
        "tokentrim.transforms.compaction.transform.generate_text",
        lambda **kwargs: "summary",
    )
    result = asyncio.run(wrapped.call_model_input_filter(payload))

    assert result.instructions == "answer briefly"
    assert result.input[0] == {"role": "system", "content": "summary"}
    assert result.input[1:] == input_items[-6:]


def test_openai_agents_adapter_implements_integration_adapter() -> None:
    client = Tokentrim()
    adapter = OpenAIAgentsAdapter(options=OpenAIAgentsOptions(steps=(FilterMessages(),)))

    wrapped = adapter.wrap(client)

    assert isinstance(adapter, IntegrationAdapter)
    assert isinstance(wrapped, RunConfig)


def test_openai_agents_adapter_merges_identity_trace_metadata() -> None:
    client = Tokentrim()
    store = InMemoryTraceStore()

    wrapped = OpenAIAgentsAdapter(
        options=OpenAIAgentsOptions(
            user_id="u1",
            session_id="s1",
            trace_store=store,
        )
    ).wrap(
        client,
        config=RunConfig(trace_metadata={"team": "support"}),
    )

    assert wrapped.trace_metadata["team"] == "support"
    namespaced = json.loads(wrapped.trace_metadata[TOKENTRIM_TRACE_METADATA_KEY])
    assert namespaced["capture_mode"] == "identity"
    assert namespaced["user_id"] == "u1"
    assert namespaced["session_id"] == "s1"
    assert isinstance(namespaced["store_id"], str)


def test_openai_agents_adapter_requires_user_and_session_for_trace_store() -> None:
    client = Tokentrim()

    with pytest.raises(Exception) as exc_info:
        OpenAIAgentsAdapter(
            options=OpenAIAgentsOptions(trace_store=InMemoryTraceStore())
        ).wrap(client)

    assert "trace_store" in str(exc_info.value)


def test_openai_agents_adapter_chains_existing_model_filter() -> None:
    client = Tokentrim()
    wrapped = OpenAIAgentsAdapter(
        options=OpenAIAgentsOptions(steps=(FilterMessages(),))
    ).wrap(
        client,
        config=RunConfig(
            call_model_input_filter=lambda data: ModelInputData(
                input=[
                    {"role": "user", "content": "hello"},
                    {"role": "user", "content": "hello"},
                ],
                instructions="custom instructions",
            )
        ),
    )
    payload = CallModelData(
        model_data=ModelInputData(
            input=[{"role": "user", "content": "ignored"}],
            instructions="ignored",
        ),
        agent=Agent(name="assistant"),
        context=None,
    )

    result = asyncio.run(wrapped.call_model_input_filter(payload))

    assert result.instructions == "custom instructions"
    assert result.input == [{"role": "user", "content": "hello [repeated 2x]"}]


def test_openai_agents_adapter_chains_existing_session_callback() -> None:
    client = Tokentrim()

    def session_callback(history_items, new_items):
        return [*history_items, {"role": "assistant", "content": "   "}, *new_items]

    wrapped = OpenAIAgentsAdapter(
        options=OpenAIAgentsOptions(steps=(FilterMessages(),))
    ).wrap(
        client,
        config=RunConfig(session_input_callback=session_callback),
    )

    result = asyncio.run(
        wrapped.session_input_callback(
            [{"role": "user", "content": "ping"}],
            [{"role": "assistant", "content": "pong"}],
        )
    )

    assert result == [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]


def test_openai_agents_adapter_applies_rlm_to_handoff_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Tokentrim()
    _install_fake_rlm(monkeypatch, response="stored context")
    wrapped = OpenAIAgentsAdapter(
        options=OpenAIAgentsOptions(
            token_budget=10,
            user_id="u1",
            session_id="s1",
            trace_store=_seed_trace_store(),
            steps=(RetrieveMemory(model="memory-model"),),
        )
    ).wrap(client)
    payload = HandoffInputData(
        input_history="hello",
        pre_handoff_items=(),
        new_items=(),
    )

    result = asyncio.run(wrapped.handoff_input_filter(payload))

    assert result.input_history == (
        {"role": "system", "content": "stored context"},
        {"role": "user", "content": "hello"},
    )
    assert result.pre_handoff_items == ()
    assert result.new_items == ()


def test_openai_agents_adapter_preserves_rich_response_items() -> None:
    client = Tokentrim()
    wrapped = OpenAIAgentsAdapter(
        options=OpenAIAgentsOptions(
            token_budget=1,
            steps=(FilterMessages(), CompactConversation()),
        )
    ).wrap(client)
    input_items = [
        {"role": "user", "content": "hello"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
        },
    ]
    payload = CallModelData(
        model_data=ModelInputData(
            input=input_items,
            instructions=None,
        ),
        agent=Agent(name="assistant"),
        context=None,
    )

    result = asyncio.run(wrapped.call_model_input_filter(payload))

    assert result.input == input_items


def test_client_wrap_integration_accepts_openai_agents_adapter() -> None:
    client = Tokentrim()

    wrapped = client.wrap_integration(
        OpenAIAgentsAdapter(options=OpenAIAgentsOptions(steps=(FilterMessages(),)))
    )

    assert isinstance(wrapped, RunConfig)
    assert wrapped.call_model_input_filter is not None


def test_client_openai_agents_config_defaults_to_call_model_hook_only() -> None:
    client = Tokentrim()

    wrapped = client.openai_agents_config(steps=(FilterMessages(),))

    assert isinstance(wrapped, RunConfig)
    assert wrapped.call_model_input_filter is not None
    assert wrapped.session_input_callback is None
    assert wrapped.handoff_input_filter is None


def test_client_openai_agents_config_supports_advanced_opt_in_hooks() -> None:
    client = Tokentrim()

    wrapped = client.openai_agents_config(
        steps=(FilterMessages(),),
        apply_to_session_history=True,
        apply_to_handoffs=True,
    )

    assert isinstance(wrapped, RunConfig)
    assert wrapped.call_model_input_filter is not None
    assert wrapped.session_input_callback is not None
    assert wrapped.handoff_input_filter is not None


def test_compose_to_openai_agents_builds_run_config() -> None:
    client = Tokentrim()

    wrapped = client.compose(FilterMessages()).to_openai_agents(token_budget=100)

    assert isinstance(wrapped, RunConfig)
    assert wrapped.call_model_input_filter is not None
    assert wrapped.session_input_callback is None
    assert wrapped.handoff_input_filter is None
