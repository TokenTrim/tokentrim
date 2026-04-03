from __future__ import annotations

from types import SimpleNamespace

import pytest

from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.tracing import InMemoryTraceStore, TokentrimSpanRecord, TokentrimTraceRecord
from tokentrim.transforms.rlm import RetrieveMemory
from tokentrim.transforms.rlm.error import RLMConfigurationError, RLMExecutionError
from tokentrim.types.state import PipelineState


def _request(
    *,
    trace_store: InMemoryTraceStore | None,
    user_id: str | None,
    session_id: str | None,
    task_hint: str | None = None,
) -> PipelineRequest:
    return PipelineRequest(
        messages=tuple(),
        tools=tuple(),
        user_id=user_id,
        session_id=session_id,
        task_hint=task_hint,
        token_budget=None,
        trace_store=trace_store,
        pipeline_tracer=None,
        steps=tuple(),
    )


def _trace(trace_id: str, *, workflow_name: str, topic: str) -> TokentrimTraceRecord:
    return TokentrimTraceRecord(
        trace_id=trace_id,
        source="openai_agents",
        capture_mode="identity",
        source_trace_id=trace_id.removeprefix("openai_agents:"),
        user_id="u1",
        session_id="s1",
        workflow_name=workflow_name,
        started_at=f"2026-04-03T10:00:0{trace_id[-1]}+00:00",
        ended_at=f"2026-04-03T10:00:1{trace_id[-1]}+00:00",
        group_id=None,
        metadata={"topic": topic},
        raw_trace={"sentinel": f"RAW_TRACE_{workflow_name}"},
    )


def _span(span_id: str, trace_id: str, *, summary: str) -> TokentrimSpanRecord:
    return TokentrimSpanRecord(
        span_id=span_id,
        trace_id=trace_id,
        source="openai_agents",
        kind="generation",
        name=span_id.removeprefix("openai_agents:"),
        source_span_id=span_id.removeprefix("openai_agents:"),
        parent_id=None,
        started_at="2026-04-03T10:00:00+00:00",
        ended_at="2026-04-03T10:00:01+00:00",
        error=None,
        metrics={"input_tokens": 10, "output_tokens": 5},
        data={"summary": summary},
        raw_span={"sentinel": f"RAW_SPAN_{summary}"},
    )


def _seed_trace_store(count: int) -> InMemoryTraceStore:
    store = InMemoryTraceStore()
    for index in range(1, count + 1):
        trace_id = f"openai_agents:trace_{index}"
        store.create_trace(
            user_id="u1",
            session_id="s1",
            trace=_trace(
                trace_id,
                workflow_name=f"workflow-{index}",
                topic=f"topic-{index}",
            ),
        )
        store.append_span(
            trace_id=trace_id,
            span=_span(
                f"openai_agents:span_{index}",
                trace_id,
                summary=f"summary-{index}",
            ),
        )
        store.complete_trace(trace_id=trace_id)
    return store


def _install_fake_rlm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: str = "relevant memory",
    error: Exception | None = None,
) -> dict[str, object]:
    captured: dict[str, object] = {
        "init_kwargs": None,
        "prompt": None,
        "root_prompt": None,
        "closed": False,
    }

    class FakeRLM:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def completion(self, prompt, root_prompt=None):
            captured["prompt"] = prompt
            captured["root_prompt"] = root_prompt
            if error is not None:
                raise error
            return SimpleNamespace(response=response)

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(
        "tokentrim.transforms.rlm.transform.import_module",
        lambda name: SimpleNamespace(RLM=FakeRLM),
    )
    return captured


def test_rlm_is_noop_without_trace_store() -> None:
    step = RetrieveMemory(model="memory-model")
    messages = [{"role": "user", "content": "hello"}]

    result = step.run(
        PipelineState(context=messages, tools=[]),
        _request(trace_store=None, user_id="u1", session_id="s1"),
    )

    assert result.context == messages


@pytest.mark.parametrize(
    ("user_id", "session_id"),
    [
        (None, "s1"),
        ("u1", None),
    ],
)
def test_rlm_is_noop_without_scope_identifiers(
    user_id: str | None,
    session_id: str | None,
) -> None:
    step = RetrieveMemory(model="memory-model")
    messages = [{"role": "user", "content": "hello"}]

    result = step.run(
        PipelineState(context=messages, tools=[]),
        _request(trace_store=_seed_trace_store(1), user_id=user_id, session_id=session_id),
    )

    assert result.context == messages


def test_rlm_is_noop_when_store_returns_no_traces() -> None:
    step = RetrieveMemory(model="memory-model")
    messages = [{"role": "user", "content": "hello"}]

    result = step.run(
        PipelineState(context=messages, tools=[]),
        _request(trace_store=InMemoryTraceStore(), user_id="u1", session_id="s1"),
    )

    assert result.context == messages


def test_rlm_uses_recent_bounded_history(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_rlm(monkeypatch)
    step = RetrieveMemory(model="memory-model", trace_limit=2)

    step.run(
        PipelineState(context=[{"role": "user", "content": "hello"}], tools=[]),
        _request(trace_store=_seed_trace_store(3), user_id="u1", session_id="s1"),
    )

    prompt = captured["prompt"]
    assert isinstance(prompt, str)
    assert "workflow-1" not in prompt
    assert "workflow-2" in prompt
    assert "workflow-3" in prompt
    assert captured["init_kwargs"] == {
        "backend": "openai",
        "backend_kwargs": {"model_name": "memory-model"},
        "environment": "local",
        "max_depth": 2,
        "max_iterations": 4,
        "verbose": False,
    }


def test_rlm_serializes_selected_traces_in_chronological_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_rlm(monkeypatch)
    step = RetrieveMemory(model="memory-model", trace_limit=2)

    step.run(
        PipelineState(context=[{"role": "user", "content": "hello"}], tools=[]),
        _request(trace_store=_seed_trace_store(3), user_id="u1", session_id="s1"),
    )

    prompt = captured["prompt"]
    assert isinstance(prompt, str)
    assert prompt.index("workflow-2") < prompt.index("workflow-3")
    assert "RAW_TRACE_workflow-2" not in prompt
    assert "RAW_TRACE_workflow-3" not in prompt
    assert "RAW_SPAN_summary-2" not in prompt
    assert "RAW_SPAN_summary-3" not in prompt
    assert '"summary":"summary-2"' in prompt
    assert '"summary":"summary-3"' in prompt


def test_rlm_prepends_synthesized_memory_and_preserves_live_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rlm(monkeypatch, response="remember the escalation path")
    step = RetrieveMemory(model="memory-model")
    messages = [
        {"role": "system", "content": "existing system"},
        {"role": "user", "content": "hello"},
    ]

    result = step.run(
        PipelineState(context=messages, tools=[]),
        _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1"),
    )

    assert result.context == [
        {"role": "system", "content": "remember the escalation path"},
        {"role": "system", "content": "existing system"},
        {"role": "user", "content": "hello"},
    ]


def test_rlm_uses_task_hint_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_rlm(monkeypatch)
    step = RetrieveMemory(model="memory-model")

    step.run(
        PipelineState(context=[{"role": "user", "content": "ignored"}], tools=[]),
        _request(
            trace_store=_seed_trace_store(1),
            user_id="u1",
            session_id="s1",
            task_hint="debug a failed database connection",
        ),
    )

    assert captured["root_prompt"] == "Current task: debug a failed database connection"
    prompt = captured["prompt"]
    assert isinstance(prompt, str)
    assert "Current task: debug a failed database connection" in prompt


def test_rlm_falls_back_to_latest_user_message_and_live_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_rlm(monkeypatch)
    step = RetrieveMemory(model="memory-model")
    messages = [
        {"role": "system", "content": "system note"},
        {"role": "assistant", "content": "what do you need?"},
        {"role": "user", "content": "please summarize our last outage"},
    ]

    step.run(
        PipelineState(context=messages, tools=[]),
        _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1"),
    )

    assert captured["root_prompt"] == "Current user request: please summarize our last outage"
    prompt = captured["prompt"]
    assert isinstance(prompt, str)
    assert "assistant: what do you need?" in prompt
    assert "user: please summarize our last outage" in prompt


def test_rlm_wraps_missing_optional_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error(name: str):
        raise ImportError(name)

    monkeypatch.setattr("tokentrim.transforms.rlm.transform.import_module", _raise_import_error)

    with pytest.raises(RLMConfigurationError) as exc_info:
        RetrieveMemory(model="memory-model").run(
            PipelineState(context=[{"role": "user", "content": "hello"}], tools=[]),
            _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1"),
        )

    assert 'tokentrim[rlm]' in str(exc_info.value)


def test_rlm_wraps_runtime_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rlm(monkeypatch, error=RuntimeError("boom"))

    with pytest.raises(RLMExecutionError) as exc_info:
        RetrieveMemory(model="memory-model").run(
            PipelineState(context=[{"role": "user", "content": "hello"}], tools=[]),
            _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1"),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_rlm_is_noop_on_blank_synthesized_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rlm(monkeypatch, response="   ")
    step = RetrieveMemory(model="memory-model")
    messages = [{"role": "user", "content": "hello"}]

    result = step.run(
        PipelineState(context=messages, tools=[]),
        _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1"),
    )

    assert result.context == messages


@pytest.mark.parametrize("response", ["FINAL_VAR(memory_block)", "FINAL(relevant memory)"])
def test_rlm_rejects_unresolved_control_output(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
) -> None:
    captured = _install_fake_rlm(monkeypatch, response=response)

    with pytest.raises(RLMExecutionError) as exc_info:
        RetrieveMemory(model="memory-model").run(
            PipelineState(context=[{"role": "user", "content": "hello"}], tools=[]),
            _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1"),
        )

    assert "unresolved RLM control text" in str(exc_info.value)
    assert captured["closed"] is True
