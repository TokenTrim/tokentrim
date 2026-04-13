from __future__ import annotations

from tokentrim.memory import InMemoryMemoryStore, MemoryRecord, MemoryWrite
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.memory import AgentAwareMemory, InjectMemory
from tokentrim.types.state import PipelineState


def _request(*, memory_store: InMemoryMemoryStore, token_budget: int | None = 500) -> PipelineRequest:
    return PipelineRequest(
        messages=tuple(),
        tools=tuple(),
        user_id="user_1",
        session_id="sess_1",
        org_id="org_1",
        task_hint="debug repo command failure",
        token_budget=token_budget,
        memory_store=memory_store,
        trace_store=None,
        pipeline_tracer=None,
        steps=tuple(),
    )


def test_inject_memory_is_noop_without_store() -> None:
    step = InjectMemory()
    state = PipelineState(context=[{"role": "user", "content": "debug this"}], tools=[])
    request = PipelineRequest(
        messages=tuple(),
        tools=tuple(),
        user_id="user_1",
        session_id="sess_1",
        org_id="org_1",
        task_hint="debug this",
        token_budget=500,
        memory_store=None,
        trace_store=None,
        pipeline_tracer=None,
        steps=tuple(),
    )

    result = step.run(state, request)

    assert result.context == state.context


def test_inject_memory_inserts_memory_after_existing_system_message() -> None:
    store = InMemoryMemoryStore()
    store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(
            content="Use repo root when running commands",
            kind="active_state",
            dedupe_key="repo_root",
        ),
    )

    step = InjectMemory(max_memories=3, max_memory_tokens=200)
    state = PipelineState(
        context=[
            {"role": "system", "content": "You are scoped to the task."},
            {"role": "user", "content": "debug the repo command failure"},
        ],
        tools=[],
    )

    result = step.run(state, _request(memory_store=store))

    assert result.context[1]["role"] == "system"
    assert "Injected memory:" in str(result.context[1]["content"])
    assert "Use repo root when running commands" in str(result.context[1]["content"])


def test_inject_memory_prefers_relevant_session_memory_over_less_relevant_user_memory() -> None:
    store = InMemoryMemoryStore()
    store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(content="Debug repo command failures from the repo root", kind="task_fact"),
    )
    store.upsert_memory(
        MemoryRecord(
            memory_id="mem_user_1",
            scope="user",
            subject_id="user_1",
            kind="preference",
            content="Prefer concise answers for travel questions",
            salience=0.9,
        )
    )

    step = InjectMemory(max_memories=1, max_memory_tokens=200)
    state = PipelineState(context=[{"role": "user", "content": "debug the repo command failure"}], tools=[])

    result = step.run(state, _request(memory_store=store))

    assert "Debug repo command failures" in str(result.context[0]["content"])
    assert "travel questions" not in str(result.context[0]["content"])


def test_inject_memory_respects_memory_token_cap() -> None:
    store = InMemoryMemoryStore()
    store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(content="x" * 500, kind="task_fact"),
    )
    step = InjectMemory(max_memories=3, max_memory_tokens=10)
    state = PipelineState(context=[{"role": "user", "content": "debug this"}], tools=[])

    result = step.run(state, _request(memory_store=store))

    assert result.context == state.context


def test_agent_aware_memory_adds_policy_prompt_and_tool() -> None:
    store = InMemoryMemoryStore()
    step = AgentAwareMemory()
    state = PipelineState(context=[{"role": "user", "content": "debug this"}], tools=[])

    result = step.run(state, _request(memory_store=store))

    assert result.context[0]["role"] == "system"
    assert "file-backed session memory directory" in str(result.context[0]["content"])
    assert {tool["name"] for tool in result.tools} == {"read_session_memory", "write_session_memory"}
