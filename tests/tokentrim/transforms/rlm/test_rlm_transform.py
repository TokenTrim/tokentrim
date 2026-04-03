from __future__ import annotations

from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms.rlm import RetrieveMemory
from tokentrim.transforms.rlm.store import NoOpMemoryStore
from tokentrim.types.state import PipelineState


class FakeStore:
    def __init__(self, value: str | None) -> None:
        self._value = value
        self.calls: list[tuple[str, str]] = []

    def retrieve(self, *, user_id: str, session_id: str) -> str | None:
        self.calls.append((user_id, session_id))
        return self._value


def _request(*, user_id: str | None, session_id: str | None) -> ContextRequest:
    return ContextRequest(
        messages=tuple(),
        user_id=user_id,
        session_id=session_id,
        token_budget=None,
        steps=(RetrieveMemory(),),
    )


def test_rlm_is_noop_without_identifiers() -> None:
    step = RetrieveMemory(memory_store=NoOpMemoryStore())
    messages = [{"role": "user", "content": "hello"}]

    result = step.run(PipelineState(context=messages, tools=[]), _request(user_id=None, session_id="session"))

    assert result.context == messages


def test_rlm_is_noop_without_retrieved_state() -> None:
    step = RetrieveMemory(memory_store=FakeStore(None))
    messages = [{"role": "user", "content": "hello"}]

    result = step.run(PipelineState(context=messages, tools=[]), _request(user_id="user", session_id="session"))

    assert result.context == messages


def test_rlm_prepends_retrieved_state() -> None:
    step = RetrieveMemory(memory_store=FakeStore("remember this"))
    messages = [{"role": "user", "content": "hello"}]

    result = step.run(PipelineState(context=messages, tools=[]), _request(user_id="user", session_id="session"))

    assert result.context == [
        {"role": "system", "content": "remember this"},
        {"role": "user", "content": "hello"},
    ]


def test_rlm_calls_store_with_exact_identifiers() -> None:
    store = FakeStore("remember this")
    step = RetrieveMemory(memory_store=store)

    step.run(
        PipelineState(context=[{"role": "user", "content": "hello"}], tools=[]),
        _request(user_id="user-1", session_id="session-1"),
    )

    assert store.calls == [("user-1", "session-1")]


def test_rlm_merges_into_existing_system_message() -> None:
    step = RetrieveMemory(memory_store=FakeStore("remember this"))
    messages = [
        {"role": "system", "content": "older summary"},
        {"role": "user", "content": "hello"},
    ]

    result = step.run(PipelineState(context=messages, tools=[]), _request(user_id="user", session_id="session"))

    assert result.context == [
        {"role": "system", "content": "remember this\n\nolder summary"},
        {"role": "user", "content": "hello"},
    ]
