from __future__ import annotations

from tokentrim.context.request import ContextRequest
from tokentrim.context.rlm import RLMStep
from tokentrim.context.store import NoOpMemoryStore


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
        steps=(RLMStep(),),
    )


def test_rlm_is_noop_without_identifiers() -> None:
    step = RLMStep(memory_store=NoOpMemoryStore())
    messages = [{"role": "user", "content": "hello"}]

    result = step.run(messages, _request(user_id=None, session_id="session"))

    assert result == messages


def test_rlm_is_noop_without_retrieved_state() -> None:
    step = RLMStep(memory_store=FakeStore(None))
    messages = [{"role": "user", "content": "hello"}]

    result = step.run(messages, _request(user_id="user", session_id="session"))

    assert result == messages


def test_rlm_prepends_retrieved_state() -> None:
    step = RLMStep(memory_store=FakeStore("remember this"))
    messages = [{"role": "user", "content": "hello"}]

    result = step.run(messages, _request(user_id="user", session_id="session"))

    assert result == [
        {"role": "system", "content": "remember this"},
        {"role": "user", "content": "hello"},
    ]


def test_rlm_calls_store_with_exact_identifiers() -> None:
    store = FakeStore("remember this")
    step = RLMStep(memory_store=store)

    step.run(
        [{"role": "user", "content": "hello"}],
        _request(user_id="user-1", session_id="session-1"),
    )

    assert store.calls == [("user-1", "session-1")]
