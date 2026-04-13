from __future__ import annotations

import pytest
from tokentrim.consolidator.engine import (
    LocalConsolidatorRuntime,
    OfflineBundleView,
    resolve_model_name,
)
from tokentrim.consolidator.errors import ConsolidatorRuntimeError
from tokentrim.tracing.records import TokentrimSpanRecord, TokentrimTraceRecord


def _trace(trace_id: str, *, workflow_name: str, topic: str) -> TokentrimTraceRecord:
    return TokentrimTraceRecord(
        trace_id=trace_id,
        source="openai_agents",
        capture_mode="identity",
        source_trace_id=trace_id.removeprefix("openai_agents:"),
        user_id="u1",
        session_id="s1",
        workflow_name=workflow_name,
        started_at="2026-04-03T10:00:00+00:00",
        ended_at="2026-04-03T10:00:01+00:00",
        group_id=None,
        metadata={"topic": topic},
        raw_trace={"ignored": "raw"},
        spans=(
            TokentrimSpanRecord(
                span_id=f"{trace_id}:span-1",
                trace_id=trace_id,
                source="openai_agents",
                kind="generation",
                name="agent_turn",
                source_span_id="span-1",
                parent_id=None,
                started_at="2026-04-03T10:00:00+00:00",
                ended_at="2026-04-03T10:00:01+00:00",
                error=None,
                metrics={"input_tokens": 10, "output_tokens": 5},
                data={"summary": "compiler dependency missing during build"},
                raw_span={"ignored": "raw"},
            ),
        ),
    )


def _bundle() -> OfflineBundleView:
    return OfflineBundleView.from_history(
        messages=[
            {"role": "user", "content": "build caffe from source"},
            {"role": "assistant", "content": "checking dependency versions"},
            {"role": "user", "content": "$ make -j2\n[exit_code] 2\ncompiler missing header"},
        ],
        traces=[_trace("openai_agents:trace_1", workflow_name="workflow-1", topic="compiler")],
        label="test_bundle",
    )


def test_runtime_root_prompt_does_not_inline_serialized_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[dict[str, str]]] = []

    def fake_generate_text(**kwargs):
        calls.append(kwargs["messages"])
        return 'FINAL("relevant memory")'

    monkeypatch.setattr("tokentrim.consolidator.engine.generate_text", fake_generate_text)

    result = LocalConsolidatorRuntime(
        model="gpt-4.1-mini",
        backend="openai",
        max_iterations=1,
    ).run(_bundle(), root_prompt="Retrieve the next-step bundle.")

    assert result == "relevant memory"
    serialized_prompt = "\n".join(message["content"] for message in calls[0])
    assert "workflow-1" not in serialized_prompt
    assert "compiler dependency missing during build" not in serialized_prompt
    assert "build caffe from source" not in serialized_prompt
    assert "`bundle.latest_messages(n)`" in serialized_prompt


def test_bundle_browser_methods_are_bounded_and_deterministic() -> None:
    bundle = _bundle()

    recent = bundle.latest_messages(2)
    trace = bundle.trace(0)
    matches = bundle.grep("compiler")

    assert [message.role for message in recent] == ["assistant", "user"]
    assert bundle.message_slice(0, 1)[0].content == "build caffe from source"
    assert bundle.latest_traces(1)[0].workflow_name == "workflow-1"
    assert matches[0].source == "message[2]"
    assert any(match.source == "trace[0]" for match in matches)
    assert "compiler" in bundle.peek(matches, limit=120).lower()
    assert "workflow: workflow-1" in bundle.to_text(trace, limit=400)


def test_runtime_records_browser_calls_and_persists_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            (
                "```repl\n"
                "recent = bundle.latest_messages(2)\n"
                "one = bundle.message_slice(0, 1)\n"
                "trace0 = bundle.trace(0)\n"
                "matches = bundle.grep('compiler')\n"
                "summary = bundle.to_text(trace0, limit=400)\n"
                "print(bundle.peek(recent, limit=200))\n"
                "```"
            ),
            "FINAL_VAR(summary)",
        ]
    )

    monkeypatch.setattr(
        "tokentrim.consolidator.engine.generate_text",
        lambda **kwargs: next(responses),
    )

    runtime = LocalConsolidatorRuntime(
        model="gpt-4.1-mini",
        backend="openai",
        max_iterations=2,
    )
    result = runtime.run(_bundle(), root_prompt="Find the relevant bundle.")

    assert "workflow: workflow-1" in result
    operations = runtime.trajectory["iterations"][0]["code_blocks"][0]["operations"]
    operation_names = [operation["name"] for operation in operations if operation["type"] == "browser_call"]
    assert operation_names == ["latest_messages", "message_slice", "trace", "grep", "to_text", "peek"]


def test_runtime_bundle_query_launches_isolated_subcall_and_records_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    responses = iter(
        [
            (
                "```repl\n"
                "matches = bundle.grep('compiler')\n"
                "answer = bundle_query('What failed?', matches)\n"
                "print(answer)\n"
                "```"
            ),
            'FINAL("compiler dependency missing")',
            "FINAL_VAR(answer)",
        ]
    )

    def fake_generate_text(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr("tokentrim.consolidator.engine.generate_text", fake_generate_text)

    runtime = LocalConsolidatorRuntime(
        model="gpt-4.1-mini",
        backend="openai",
        max_iterations=2,
        subcall_model="gpt-4o-mini",
    )
    result = runtime.run(_bundle(), root_prompt="Recover the failing dependency.")

    assert result == "compiler dependency missing"
    assert calls[0]["model"] == "openai/gpt-4.1-mini"
    assert calls[1]["model"] == "openai/gpt-4o-mini"
    subcall_ops = [
        operation
        for operation in runtime.trajectory["iterations"][0]["code_blocks"][0]["operations"]
        if operation["type"] == "subcall"
    ]
    assert subcall_ops[0]["status"] == "ok"
    assert subcall_ops[0]["trajectory"]["depth"] == 1
    assert runtime.trajectory["subcalls_used"] == 1


def test_runtime_depth_one_subcalls_cannot_recurse_further(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            (
                "```repl\n"
                "answer = bundle_query('nested', bundle.trace(0))\n"
                "print(answer)\n"
                "```"
            ),
            (
                "```repl\n"
                "nested = bundle_query('too deep')\n"
                "print(nested)\n"
                "FINAL_VAR(nested)\n"
                "```"
            ),
            "FINAL_VAR(answer)",
        ]
    )

    monkeypatch.setattr(
        "tokentrim.consolidator.engine.generate_text",
        lambda **kwargs: next(responses),
    )

    result = LocalConsolidatorRuntime(
        model="gpt-4.1-mini",
        backend="openai",
        max_iterations=2,
        max_depth=1,
    ).run(_bundle(), root_prompt="Test recursive limits.")

    assert result == "Error: bundle_query depth limit reached."


def test_runtime_continues_after_unresolved_final_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            "FINAL_VAR(missing_memory)",
            'FINAL("remember the runbook")',
        ]
    )
    calls: list[list[dict[str, str]]] = []

    def fake_generate_text(**kwargs):
        calls.append(kwargs["messages"])
        return next(responses)

    monkeypatch.setattr("tokentrim.consolidator.engine.generate_text", fake_generate_text)

    runtime = LocalConsolidatorRuntime(
        model="gpt-4.1-mini",
        backend="openai",
        max_iterations=2,
    )
    result = runtime.run(_bundle(), root_prompt="Keep going after a missing variable.")

    assert result == "remember the runbook"
    assert len(calls) == 2
    assert len(runtime.trajectory["iterations"]) == 2


def test_runtime_subcall_limit_breaches_fail_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            (
                "```repl\n"
                "first = bundle_query('one')\n"
                "second = bundle_query('two')\n"
                "print(first)\n"
                "print(second)\n"
                "```"
            ),
            'FINAL("first answer")',
            "FINAL_VAR(second)",
        ]
    )

    monkeypatch.setattr(
        "tokentrim.consolidator.engine.generate_text",
        lambda **kwargs: next(responses),
    )

    runtime = LocalConsolidatorRuntime(
        model="gpt-4.1-mini",
        backend="openai",
        max_iterations=2,
        max_subcalls=1,
    )
    result = runtime.run(_bundle(), root_prompt="Test subcall limits.")

    assert result == "Error: bundle_query subcall limit reached."
    subcall_ops = [
        operation
        for operation in runtime.trajectory["iterations"][0]["code_blocks"][0]["operations"]
        if operation["type"] == "subcall"
    ]
    assert [operation["status"] for operation in subcall_ops] == ["ok", "subcall_limit"]


def test_runtime_llm_query_uses_plain_generate_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    responses = iter(
        [
            (
                "```repl\n"
                "memory_block = llm_query('Summarize the bundle in five words.')\n"
                "print(memory_block)\n"
                "```"
            ),
            "five word memory summary",
            "FINAL_VAR(memory_block)",
        ]
    )

    def fake_generate_text(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr("tokentrim.consolidator.engine.generate_text", fake_generate_text)

    result = LocalConsolidatorRuntime(
        model="claude-3-5-sonnet",
        backend="anthropic",
        max_iterations=2,
    ).run(_bundle(), root_prompt="Summarize the bundle.")

    assert result == "five word memory summary"
    assert calls[0]["model"] == "anthropic/claude-3-5-sonnet"
    assert calls[1]["messages"] == [
        {"role": "user", "content": "Summarize the bundle in five words."}
    ]


def test_runtime_surfaces_python_errors_in_follow_up_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            "```repl\nprint(missing_name)\n```",
            'FINAL("resolved after seeing the error")',
        ]
    )
    observed_errors: list[str] = []

    def fake_generate_text(**kwargs):
        messages = kwargs["messages"]
        if len(messages) > 2:
            observed_errors.append(messages[-2]["content"])
        return next(responses)

    monkeypatch.setattr("tokentrim.consolidator.engine.generate_text", fake_generate_text)

    result = LocalConsolidatorRuntime(
        model="gpt-4.1-mini",
        backend="openai",
        max_iterations=2,
    ).run(_bundle(), root_prompt="Handle the exception.")

    assert result == "resolved after seeing the error"
    assert any("NameError" in message for message in observed_errors)


def test_runtime_errors_after_iteration_limit_without_plain_text_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(["```repl\nx = 1\n```"])

    monkeypatch.setattr(
        "tokentrim.consolidator.engine.generate_text",
        lambda **kwargs: next(responses),
    )

    runtime = LocalConsolidatorRuntime(
        model="gpt-4.1-mini",
        backend="openai",
        max_iterations=1,
    )
    with pytest.raises(ConsolidatorRuntimeError) as exc_info:
        runtime.run(_bundle(), root_prompt="Stop after the iteration limit.")

    assert "max iterations" in str(exc_info.value)
    assert len(runtime.trajectory["iterations"]) == 1


def test_runtime_accepts_direct_json_final_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tokentrim.consolidator.engine.generate_text",
        lambda **kwargs: (
            '{'
            '"user_upserts":[],"org_upserts":[],"user_archives":[],"org_archives":[],'
            '"merge_operations":[],"rationale":["enough evidence"],"source_refs":["openai_agents:trace_1"]'
            '}'
        ),
    )

    runtime = LocalConsolidatorRuntime(
        model="gpt-4.1-mini",
        backend="openai",
        max_iterations=1,
    )
    result = runtime.run(_bundle(), root_prompt="Return the durable plan directly.")

    assert '"rationale":["enough evidence"]' in result


@pytest.mark.parametrize(
    ("backend", "model", "expected"),
    [
        ("openai", "gpt-4.1-mini", "openai/gpt-4.1-mini"),
        ("anthropic", "claude-3-5-sonnet", "anthropic/claude-3-5-sonnet"),
        ("openai", "openai/gpt-4.1-mini", "openai/gpt-4.1-mini"),
    ],
)
def test_resolve_model_name(
    backend: str,
    model: str,
    expected: str,
) -> None:
    assert resolve_model_name(backend=backend, model=model) == expected
