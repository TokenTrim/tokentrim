from __future__ import annotations

import json

import pytest

from tokentrim.core.token_counting import count_message_tokens
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.tracing import InMemoryTraceStore, TokentrimSpanRecord, TokentrimTraceRecord
from tokentrim.transforms.rlm import RetrieveMemory, capture_rlm_invocation_logs
from tokentrim.transforms.rlm.error import RLMConfigurationError, RLMExecutionError
from tokentrim.transforms.rlm.runtime import RLMContextView
from tokentrim.types.state import PipelineState


def _request(
    *,
    trace_store: InMemoryTraceStore | None,
    user_id: str | None,
    session_id: str | None,
    task_hint: str | None = None,
    token_budget: int | None = 1,
) -> PipelineRequest:
    return PipelineRequest(
        messages=tuple(),
        tools=tuple(),
        user_id=user_id,
        session_id=session_id,
        task_hint=task_hint,
        token_budget=token_budget,
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


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: str = "relevant memory",
    error: Exception | None = None,
) -> dict[str, object]:
    captured: dict[str, object] = {
        "init_kwargs": None,
        "context": None,
        "root_prompt": None,
        "system_prompt": None,
        "trajectory": None,
    }

    class FakeRuntime:
        def __init__(
            self,
            *,
            model,
            backend,
            max_iterations,
            tokenizer_model,
            max_depth,
            max_subcalls,
            subcall_model,
        ):
            captured["init_kwargs"] = {
                "model": model,
                "backend": backend,
                "max_iterations": max_iterations,
                "tokenizer_model": tokenizer_model,
                "max_depth": max_depth,
                "max_subcalls": max_subcalls,
                "subcall_model": subcall_model,
            }
            self.trajectory = {"depth": 0, "iterations": [{"response": response}]}

        def run(self, context, root_prompt=None, system_prompt=None):
            captured["context"] = context
            captured["root_prompt"] = root_prompt
            captured["system_prompt"] = system_prompt
            captured["trajectory"] = self.trajectory
            if error is not None:
                raise error
            return response

    monkeypatch.setattr(
        "tokentrim.transforms.rlm.transform.LocalRLMRuntime",
        FakeRuntime,
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


def test_rlm_runs_even_without_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_runtime(monkeypatch)

    result = RetrieveMemory(model="memory-model").run(
        PipelineState(context=[{"role": "user", "content": "hello"}], tools=[]),
        _request(
            trace_store=_seed_trace_store(1),
            user_id="u1",
            session_id="s1",
            token_budget=None,
        ),
    )

    assert isinstance(captured["context"], RLMContextView)
    assert result.context[0] == {
        "role": "system",
        "content": "Retrieved memory:\nrelevant memory",
    }


def test_rlm_uses_structured_recent_bounded_history(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_runtime(monkeypatch)
    step = RetrieveMemory(
        model="memory-model",
        trace_limit=2,
        max_subcalls=6,
        subcall_model="gpt-4o-mini",
        max_memory_tokens=256,
    )

    step.run(
        PipelineState(context=[{"role": "user", "content": "hello " * 4}], tools=[]),
        _request(trace_store=_seed_trace_store(3), user_id="u1", session_id="s1", token_budget=200),
    )

    context = captured["context"]
    assert isinstance(context, RLMContextView)
    assert [trace.workflow_name for trace in context.traces] == ["workflow-2", "workflow-3"]
    assert context.messages[0].content.startswith("hello")
    assert captured["init_kwargs"] == {
        "model": "memory-model",
        "backend": "openai",
        "max_iterations": 4,
        "tokenizer_model": None,
        "max_depth": 1,
        "max_subcalls": 6,
        "subcall_model": "gpt-4o-mini",
    }


def test_rlm_root_prompt_uses_task_brief_not_serialized_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_runtime(monkeypatch)
    step = RetrieveMemory(model="memory-model")
    messages = [
        {"role": "user", "content": "train the cifar classifier"},
        {"role": "assistant", "content": "installing dependencies"},
        {"role": "user", "content": "$ make -j2\n[exit_code] 2"},
    ]

    step.run(
        PipelineState(context=messages, tools=[]),
        _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1", token_budget=20),
    )

    root_prompt = captured["root_prompt"]
    assert isinstance(root_prompt, str)
    assert "Top-level task:\ntrain the cifar classifier" in root_prompt
    assert "Latest assistant action:\ninstalling dependencies" in root_prompt
    assert 'return FINAL("") immediately' in root_prompt
    assert "Stored trace history" not in root_prompt
    assert "Current live context" not in root_prompt


def test_rlm_adds_system_memory_and_preserves_live_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch, response="remember the escalation path")
    step = RetrieveMemory(model="memory-model")
    messages = [
        {"role": "user", "content": "hello " * 40},
        {"role": "assistant", "content": "rerunning make"},
        {"role": "user", "content": "$ make -j2\n[exit_code] 2"},
    ]

    result = step.run(
        PipelineState(context=messages, tools=[]),
        _request(
            trace_store=_seed_trace_store(1),
            user_id="u1",
            session_id="s1",
            token_budget=500,
        ),
    )

    assert result.context[0] == {
        "role": "system",
        "content": "Retrieved memory:\nremember the escalation path",
    }
    assert result.context[1:] == messages


def test_rlm_inserts_memory_after_leading_system_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runtime(monkeypatch, response="use the internal mirror URL")
    messages = [
        {"role": "system", "content": "summary"},
        {"role": "assistant", "content": "checking mirrors"},
        {"role": "user", "content": "$ curl ...\n[exit_code] 22"},
    ]

    result = RetrieveMemory(model="memory-model").run(
        PipelineState(context=messages, tools=[]),
        _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1", token_budget=500),
    )

    assert result.context[:2] == [
        {"role": "system", "content": "summary"},
        {"role": "system", "content": "Retrieved memory:\nuse the internal mirror URL"},
    ]
    assert result.context[2:] == messages[1:]


def test_rlm_blank_output_is_noop_and_logged_as_no_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch, response="   ")
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "checking the logs"},
    ]

    with capture_rlm_invocation_logs() as logs:
        result = RetrieveMemory(model="memory-model").run(
            PipelineState(context=messages, tools=[]),
            _request(
                trace_store=_seed_trace_store(1),
                user_id="u1",
                session_id="s1",
                token_budget=50,
            ),
        )

    assert result.context == messages
    assert logs[-1]["status"] == "no_memory"
    assert logs[-1]["synthesized_memory"] == ""


def test_rlm_drops_memory_when_no_budget_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runtime(monkeypatch, response="remember the failing dependency")
    messages = [{"role": "user", "content": "hello " * 80}]

    result = RetrieveMemory(model="memory-model").run(
        PipelineState(context=messages, tools=[]),
        _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1", token_budget=20),
    )

    assert result.context == messages


def test_rlm_records_invocation_log_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runtime(monkeypatch, response='FINAL("remember the escalation path")')

    with capture_rlm_invocation_logs() as logs:
        RetrieveMemory(model="memory-model").run(
            PipelineState(context=[{"role": "user", "content": "hello " * 10}], tools=[]),
            _request(
                trace_store=_seed_trace_store(1),
                user_id="u1",
                session_id="s1",
                token_budget=200,
            ),
        )

    assert len(logs) == 1
    assert logs[0]["artifact_type"] == "tokentrim_rlm_invocation"
    assert logs[0]["status"] == "ok"
    assert logs[0]["synthesized_memory"] == "remember the escalation path"
    assert logs[0]["response"] == 'FINAL("remember the escalation path")'
    assert logs[0]["trajectory"] == {"depth": 0, "iterations": [{"response": 'FINAL("remember the escalation path")'}]}


def test_rlm_rejects_recursive_max_depth() -> None:
    with pytest.raises(RLMConfigurationError) as exc_info:
        RetrieveMemory(model="memory-model", max_depth=2).run(
            PipelineState(context=[{"role": "user", "content": "hello"}], tools=[]),
            _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1"),
        )

    assert "max_depth=1" in str(exc_info.value)


def test_rlm_wraps_runtime_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runtime(monkeypatch, error=RuntimeError("boom"))

    with pytest.raises(RLMExecutionError) as exc_info:
        RetrieveMemory(model="memory-model").run(
            PipelineState(context=[{"role": "user", "content": "hello " * 10}], tools=[]),
            _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1", token_budget=200),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_rlm_propagates_iteration_limit_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runtime(
        monkeypatch,
        error=RLMExecutionError("RLM runtime hit max iterations without producing a final answer."),
    )

    with pytest.raises(RLMExecutionError) as exc_info:
        RetrieveMemory(model="memory-model").run(
            PipelineState(context=[{"role": "user", "content": "hello " * 10}], tools=[]),
            _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1", token_budget=200),
        )

    assert str(exc_info.value) == "RLM runtime hit max iterations without producing a final answer."


@pytest.mark.parametrize("response", ["FINAL_VAR(memory_block)", " FINAL_VAR(memory_block)"])
def test_rlm_rejects_unresolved_control_output(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
) -> None:
    _install_fake_runtime(monkeypatch, response=response)

    with pytest.raises(RLMExecutionError) as exc_info:
        RetrieveMemory(model="memory-model").run(
            PipelineState(context=[{"role": "user", "content": "hello " * 10}], tools=[]),
            _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1", token_budget=200),
        )

    assert "unresolved RLM control text" in str(exc_info.value)


def test_rlm_writes_debug_artifact_for_unresolved_final_var(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TOKENTRIM_RLM_DEBUG_DIR", str(tmp_path))
    _install_fake_runtime(monkeypatch, response="FINAL_VAR(final_answer)")

    with pytest.raises(RLMExecutionError) as exc_info:
        RetrieveMemory(model="memory-model").run(
            PipelineState(context=[{"role": "user", "content": "hello " * 10}], tools=[]),
            _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1", token_budget=200),
        )

    message = str(exc_info.value)
    assert "Debug artifact written to" in message

    artifacts = list(tmp_path.glob("unresolved-final-var-*.json"))
    assert len(artifacts) == 1

    payload = json.loads(artifacts[0].read_text())
    assert payload["artifact_type"] == "tokentrim_rlm_unresolved_final_var"
    assert payload["response"] == "FINAL_VAR(final_answer)"
    assert payload["final_var_name"] == "final_answer"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("FINAL(\"relevant memory\")", "relevant memory"),
        ("FINAL(\n\"relevant memory\nsecond line\"\n)", "relevant memory\nsecond line"),
    ],
)
def test_rlm_unwraps_final_string_output(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    expected: str,
) -> None:
    _install_fake_runtime(monkeypatch, response=response)

    result = RetrieveMemory(model="memory-model").run(
        PipelineState(context=[{"role": "user", "content": "hello " * 10}], tools=[]),
        _request(
            trace_store=_seed_trace_store(1),
            user_id="u1",
            session_id="s1",
            token_budget=200,
        ),
    )

    assert result.context[0] == {
        "role": "system",
        "content": f"Retrieved memory:\n{expected}",
    }


@pytest.mark.parametrize(
    ("backend", "model", "expected_model"),
    [
        ("openai", "gpt-4.1-mini", "openai/gpt-4.1-mini"),
        ("anthropic", "claude-3-5-sonnet", "anthropic/claude-3-5-sonnet"),
        ("openai", "openai/gpt-4.1-mini", "openai/gpt-4.1-mini"),
    ],
)
def test_rlm_uses_runtime_model_normalization(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    model: str,
    expected_model: str,
) -> None:
    observed_models: list[str] = []

    def fake_generate_text(**kwargs):
        observed_models.append(kwargs["model"])
        return 'FINAL("relevant memory")'

    monkeypatch.setattr("tokentrim.transforms.rlm.runtime.generate_text", fake_generate_text)

    result = RetrieveMemory(model=model, backend=backend).run(
        PipelineState(context=[{"role": "user", "content": "hello " * 10}], tools=[]),
        _request(
            trace_store=_seed_trace_store(1),
            user_id="u1",
            session_id="s1",
            token_budget=200,
        ),
    )

    assert result.context[0] == {
        "role": "system",
        "content": "Retrieved memory:\nrelevant memory",
    }
    assert observed_models == [expected_model]


def test_rlm_trims_memory_to_max_memory_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runtime(monkeypatch, response="chunk " * 200)

    result = RetrieveMemory(model="memory-model", max_memory_tokens=20).run(
        PipelineState(context=[{"role": "user", "content": "hello"}], tools=[]),
        _request(trace_store=_seed_trace_store(1), user_id="u1", session_id="s1", token_budget=None),
    )

    assert result.context[0]["role"] == "system"
    assert count_message_tokens([result.context[0]], None) <= 20
