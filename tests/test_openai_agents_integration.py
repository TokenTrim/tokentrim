from __future__ import annotations

import asyncio

import pytest

agents = pytest.importorskip("agents")

from agents import Agent, RunConfig
from agents.handoffs import HandoffInputData
from agents.run import CallModelData, ModelInputData

from tokentrim import Tokentrim
from tokentrim.integrations.openai_agents import OpenAIAgentsOptions, wrap_run_config


class InMemoryStore:
    def retrieve(self, *, user_id: str, session_id: str) -> str | None:
        if user_id == "u1" and session_id == "s1":
            return "stored context"
        return None


def _message_items(count: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"message {index} " + ("x" * 40)} for index in range(count)]


def test_wrap_run_config_compacts_plain_text_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Tokentrim(compaction_model="compact-model")
    wrapped = wrap_run_config(
        client,
        options=OpenAIAgentsOptions(
            token_budget=25,
            enable_compaction=True,
        ),
    )
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
        "tokentrim.context.compaction.generate_text",
        lambda **kwargs: "summary",
    )
    result = asyncio.run(wrapped.call_model_input_filter(payload))

    assert result.instructions == "answer briefly"
    assert result.input[0] == {"role": "system", "content": "summary"}
    assert result.input[1:] == input_items[-6:]


def test_wrap_run_config_chains_existing_model_filter() -> None:
    client = Tokentrim()
    wrapped = wrap_run_config(
        client,
        run_config=RunConfig(
            call_model_input_filter=lambda data: ModelInputData(
                input=[
                    {"role": "user", "content": "hello"},
                    {"role": "user", "content": "hello"},
                ],
                instructions="custom instructions",
            )
        ),
        options=OpenAIAgentsOptions(enable_filter=True),
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


def test_wrap_run_config_chains_existing_session_callback() -> None:
    client = Tokentrim()

    def session_callback(history_items, new_items):
        return [*history_items, {"role": "assistant", "content": "   "}, *new_items]

    wrapped = wrap_run_config(
        client,
        run_config=RunConfig(session_input_callback=session_callback),
        options=OpenAIAgentsOptions(enable_filter=True),
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


def test_wrap_run_config_applies_rlm_to_handoff_history() -> None:
    client = Tokentrim(memory_store=InMemoryStore())
    wrapped = wrap_run_config(
        client,
        options=OpenAIAgentsOptions(
            user_id="u1",
            session_id="s1",
            enable_rlm=True,
        ),
    )
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


def test_wrap_run_config_preserves_rich_response_items() -> None:
    client = Tokentrim()
    wrapped = wrap_run_config(
        client,
        options=OpenAIAgentsOptions(
            token_budget=1,
            enable_filter=True,
            enable_compaction=True,
        ),
    )
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


def test_client_wrap_openai_agents_run_config_delegates() -> None:
    client = Tokentrim()

    wrapped = client.wrap_openai_agents_run_config(
        options=OpenAIAgentsOptions(enable_filter=True)
    )

    assert isinstance(wrapped, RunConfig)
    assert wrapped.call_model_input_filter is not None
