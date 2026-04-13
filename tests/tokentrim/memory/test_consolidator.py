from __future__ import annotations

import json
from dataclasses import dataclass

from tokentrim.consolidator import (
    AgenticConsolidatorAgent,
    ConsolidationInput,
    ConsolidationJobConfig,
    ConsolidationPlan,
    ConsolidatorAgent,
    DeterministicConsolidatorAgent,
    MemoryUpsert,
    ModelConsolidatorAgent,
    OfflineMemoryConsolidator,
    SessionConsolidationJob,
    build_agentic_consolidator_system_prompt,
    build_agentic_session_consolidation_job,
    build_consolidation_bundle,
    build_consolidator_system_prompt,
    build_model_session_consolidation_job,
    parse_consolidation_plan_response,
    run_session_consolidation,
    serialize_consolidation_input,
)
from tokentrim.memory import (
    InMemoryMemoryStore,
    MemoryRecord,
    MemoryWrite,
)
from tokentrim.errors.base import TokentrimError
from tokentrim.tracing import InMemoryTraceStore, TokentrimSpanRecord, TokentrimTraceRecord


def _trace(trace_id: str, *, user_id: str = "user_1", session_id: str = "session_1") -> TokentrimTraceRecord:
    return TokentrimTraceRecord(
        trace_id=trace_id,
        source="openai_agents",
        capture_mode="identity",
        source_trace_id=trace_id.removeprefix("openai_agents:"),
        user_id=user_id,
        session_id=session_id,
        workflow_name="code_agent",
        started_at="2026-04-13T10:00:00Z",
        ended_at="2026-04-13T10:01:00Z",
        group_id=None,
        metadata={"source": "test"},
        raw_trace={"id": trace_id.removeprefix("openai_agents:")},
    )


def _span(
    span_id: str,
    *,
    trace_id: str,
    name: str,
    error: dict[str, object] | None = None,
) -> TokentrimSpanRecord:
    return TokentrimSpanRecord(
        span_id=span_id,
        trace_id=trace_id,
        source="openai_agents",
        kind="tool_call",
        name=name,
        source_span_id=span_id.removeprefix("openai_agents:"),
        parent_id=None,
        started_at="2026-04-13T10:00:00Z",
        ended_at="2026-04-13T10:00:01Z",
        error=error,
        metrics=None,
        data={"name": name},
        raw_span={"id": span_id.removeprefix("openai_agents:")},
    )


def _complete_trace(
    trace_store: InMemoryTraceStore,
    *,
    trace_id: str,
    user_id: str = "user_1",
    session_id: str = "session_1",
    failure_message: str = "pytest missing",
) -> None:
    trace = _trace(trace_id, user_id=user_id, session_id=session_id)
    trace_store.create_trace(user_id=user_id, session_id=session_id, trace=trace)
    trace_store.append_span(
        trace_id=trace_id,
        span=_span(
            f"{trace_id}:fail",
            trace_id=trace_id,
            name="run_tests",
            error={"type": "CommandError", "message": failure_message},
        ),
    )
    trace_store.append_span(
        trace_id=trace_id,
        span=_span(
            f"{trace_id}:fix",
            trace_id=trace_id,
            name="install_pytest_and_rerun",
        ),
    )
    trace_store.complete_trace(trace_id=trace_id)


def _complete_execute_command_trace(
    trace_store: InMemoryTraceStore,
    *,
    trace_id: str,
    user_id: str = "user_1",
    session_id: str = "session_1",
    analysis: str = "Disable X11 dependency and rebuild the package.",
    command: str = "cd /app/pmars && sed -i '/libx11-dev/d' debian/control && dpkg-buildpackage -b -us -uc",
    terminal_output: str = (
        "$ dpkg-buildpackage -b -us -uc\n"
        "xwindisp.c:39:10: fatal error: X11/Xlib.h: No such file or directory\n"
        "compilation terminated.\n"
        "[exit_code] 2"
    ),
) -> None:
    trace = _trace(trace_id, user_id=user_id, session_id=session_id)
    trace_store.create_trace(user_id=user_id, session_id=session_id, trace=trace)
    trace_store.append_span(
        trace_id=trace_id,
        span=TokentrimSpanRecord(
            span_id=f"{trace_id}:execute_command",
            trace_id=trace_id,
            source="openai_agents",
            kind="function",
            name="execute_command",
            source_span_id=f"{trace_id.removeprefix('openai_agents:')}:execute_command",
            parent_id=None,
            started_at="2026-04-13T10:00:00Z",
            ended_at="2026-04-13T10:00:05Z",
            error=None,
            metrics=None,
            data={
                "name": "execute_command",
                "input": json.dumps(
                    {
                        "analysis": analysis,
                        "command": command,
                        "duration_seconds": 60,
                    }
                ),
                "output": repr(
                    {
                        "action": "execute_command",
                        "assistant_text": analysis,
                        "command": command,
                        "duration_seconds": 60.0,
                        "terminal_output": terminal_output,
                    }
                ),
            },
            raw_span={"id": f"{trace_id.removeprefix('openai_agents:')}:execute_command"},
        ),
    )
    trace_store.complete_trace(trace_id=trace_id)


def test_offline_consolidator_builds_input_from_trace_and_memory_scopes() -> None:
    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")
    _complete_trace(trace_store, trace_id="openai_agents:trace_2")

    session_record = memory_store.write_session_memory(
        session_id="session_1",
        write=MemoryWrite(content="Avoid reset --hard", kind="constraint"),
    )
    user_record = memory_store.upsert_memory(
        MemoryRecord(
            memory_id="mem_user_1",
            scope="user",
            subject_id="user_1",
            kind="preference",
            content="Prefer concise responses",
        )
    )
    org_record = memory_store.upsert_memory(
        MemoryRecord(
            memory_id="mem_org_1",
            scope="org",
            subject_id="org_1",
            kind="workflow_pattern",
            content="Run tests before commit",
        )
    )

    consolidator = OfflineMemoryConsolidator(memory_store=memory_store, trace_store=trace_store)
    consolidation_input = consolidator.build_input(
        session_id="session_1",
        user_id="user_1",
        org_id="org_1",
    )

    assert len(consolidation_input.traces) == 2
    assert consolidation_input.session_memories == (session_record,)
    assert consolidation_input.user_memories == (user_record,)
    assert consolidation_input.org_memories == (org_record,)


def test_offline_consolidator_dry_run_does_not_write_durable_memory() -> None:
    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")
    _complete_trace(trace_store, trace_id="openai_agents:trace_2")

    result = run_session_consolidation(
        memory_store=memory_store,
        trace_store=trace_store,
        session_id="session_1",
        user_id="user_1",
        org_id="org_1",
        apply=False,
    )

    assert result.apply_result is None
    assert len(result.plan.org_upserts) == 1
    assert memory_store.list_memories(scope="org", subject_id="org_1") == ()


def test_offline_consolidator_apply_persists_durable_memory_only() -> None:
    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")
    _complete_trace(trace_store, trace_id="openai_agents:trace_2")
    session_record = memory_store.write_session_memory(
        session_id="session_1",
        write=MemoryWrite(content="Temporary repo state", kind="active_state"),
    )

    result = run_session_consolidation(
        memory_store=memory_store,
        trace_store=trace_store,
        session_id="session_1",
        user_id="user_1",
        org_id="org_1",
        apply=True,
    )

    assert result.apply_result is not None
    assert len(result.apply_result.upserted) == 1
    stored_org_memories = memory_store.list_memories(scope="org", subject_id="org_1")
    assert len(stored_org_memories) == 1
    assert stored_org_memories[0].kind == "failure_recovery"
    stored_session_memories = memory_store.list_memories(scope="session", subject_id="session_1")
    assert stored_session_memories == (session_record,)


def test_apply_consolidation_plan_generates_distinct_memory_ids_for_multiple_upserts() -> None:
    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_execute_command_trace(trace_store, trace_id="openai_agents:trace_pmars_1")
    _complete_execute_command_trace(
        trace_store,
        trace_id="openai_agents:trace_pmars_2",
        analysis="Install X11 headers and rebuild.",
        command="apt-get install -y libx11-dev && cd /app/pmars && dpkg-buildpackage -b -us -uc",
        terminal_output=(
            "$ apt-get install -y libx11-dev && dpkg-buildpackage -b -us -uc\n"
            "xwindisp.c:39:10: fatal error: X11/Xlib.h: No such file or directory\n"
            "[exit_code] 2"
        ),
    )

    result = run_session_consolidation(
        memory_store=memory_store,
        trace_store=trace_store,
        session_id="session_1",
        user_id="user_1",
        org_id=None,
        apply=True,
        agent=DeterministicConsolidatorAgent(),
    )

    assert result.apply_result is not None
    memory_ids = tuple(record.memory_id for record in result.apply_result.upserted)
    assert len(memory_ids) == len(set(memory_ids))


def test_deterministic_consolidator_agent_builds_plan() -> None:
    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")
    _complete_trace(trace_store, trace_id="openai_agents:trace_2")

    consolidator = OfflineMemoryConsolidator(
        memory_store=memory_store,
        trace_store=trace_store,
        agent=DeterministicConsolidatorAgent(),
    )

    result = consolidator.run(
        session_id="session_1",
        user_id="user_1",
        org_id="org_1",
        apply=False,
    )

    assert len(result.plan.org_upserts) == 1


def test_deterministic_consolidator_extracts_repair_memory_from_failed_command_trace() -> None:
    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_execute_command_trace(trace_store, trace_id="openai_agents:trace_pmars")

    result = run_session_consolidation(
        memory_store=memory_store,
        trace_store=trace_store,
        session_id="session_1",
        user_id="user_1",
        org_id=None,
        apply=False,
        agent=DeterministicConsolidatorAgent(),
    )

    assert len(result.plan.user_upserts) == 1
    upsert = result.plan.user_upserts[0]
    assert upsert.write.kind == "failure_recovery"
    assert "X11/Xlib.h" in upsert.write.content
    assert "Disable X11 dependency" in upsert.write.content
    assert upsert.write.source_refs == (
        "openai_agents:trace_pmars",
        "openai_agents:trace_pmars:execute_command",
    )


@dataclass(frozen=True, slots=True)
class _RecordingAgent:
    seen_input: list[ConsolidationInput]

    def build_plan(self, consolidation_input: ConsolidationInput) -> ConsolidationPlan:
        self.seen_input.append(consolidation_input)
        return ConsolidationPlan(
            user_upserts=(
                MemoryUpsert(
                    scope="user",
                    subject_id=consolidation_input.user_id,
                    memory_id=None,
                    write=MemoryWrite(
                        content="User prefers deterministic consolidator runs",
                        kind="preference",
                        dedupe_key="pref:deterministic_consolidator",
                    ),
                ),
            ),
        )


def test_offline_consolidator_uses_injected_agent_contract() -> None:
    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")
    recorded_inputs: list[ConsolidationInput] = []
    agent: ConsolidatorAgent = _RecordingAgent(recorded_inputs)

    result = run_session_consolidation(
        memory_store=memory_store,
        trace_store=trace_store,
        session_id="session_1",
        user_id="user_1",
        org_id="org_1",
        apply=True,
        agent=agent,
    )

    assert len(recorded_inputs) == 1
    assert recorded_inputs[0].session_id == "session_1"
    assert result.apply_result is not None
    stored_user_memories = memory_store.list_memories(scope="user", subject_id="user_1")
    assert len(stored_user_memories) == 1
    assert stored_user_memories[0].dedupe_key == "pref:deterministic_consolidator"


def test_build_consolidator_system_prompt_describes_boundaries() -> None:
    prompt = build_consolidator_system_prompt()

    assert "offline memory consolidator" in prompt
    assert "Never write session memory" in prompt
    assert "Never mutate traces" in prompt


def test_serialize_consolidation_input_includes_traces_and_memory_scopes() -> None:
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")
    trace = trace_store.list_traces(user_id="user_1", session_id="session_1")[0]
    payload = serialize_consolidation_input(
        ConsolidationInput(
            session_id="session_1",
            user_id="user_1",
            org_id="org_1",
            traces=(trace,),
            session_memories=(
                MemoryRecord(
                    memory_id="mem_session_1",
                    scope="session",
                    subject_id="session_1",
                    kind="constraint",
                    content="Avoid reset --hard",
                ),
            ),
            user_memories=(),
            org_memories=(),
        )
    )

    assert payload["session_id"] == "session_1"
    assert payload["org_id"] == "org_1"
    assert payload["traces"][0]["trace_id"] == "openai_agents:trace_1"
    assert payload["traces"][0]["spans"][0]["name"] == "run_tests"
    assert payload["session_memories"][0]["memory_id"] == "mem_session_1"


def test_parse_consolidation_plan_response_parses_valid_json() -> None:
    plan = parse_consolidation_plan_response(
        """
        {
          "user_upserts": [
            {
              "subject_id": "user_1",
              "write": {
                "content": "User prefers concise summaries",
                "kind": "preference",
                "salience": 0.9,
                "dedupe_key": "pref:concise",
                "metadata": {"origin": "consolidator"},
                "source_refs": ["trace_1"]
              }
            }
          ],
          "org_upserts": [],
          "user_archives": [],
          "org_archives": [],
          "merge_operations": [],
          "rationale": ["Promote stable preference"],
          "source_refs": ["trace_1"]
        }
        """
    )

    assert len(plan.user_upserts) == 1
    assert plan.user_upserts[0].write.kind == "preference"
    assert plan.user_upserts[0].write.metadata == {"origin": "consolidator"}
    assert plan.rationale == ("Promote stable preference",)


def test_parse_consolidation_plan_response_rejects_invalid_scope() -> None:
    try:
        parse_consolidation_plan_response(
            """
            {
              "user_upserts": [{"scope": "session", "subject_id": "user_1", "write": {"content": "x", "kind": "preference"}}],
              "org_upserts": [],
              "user_archives": [],
              "org_archives": [],
              "merge_operations": [],
              "rationale": [],
              "source_refs": []
            }
            """
        )
    except TokentrimError as exc:
        assert "invalid scope" in str(exc)
    else:
        raise AssertionError("Expected TokentrimError for invalid scope.")


def test_model_consolidator_agent_calls_model_and_validates_plan(
    monkeypatch,
) -> None:
    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")

    calls: list[dict[str, object]] = []

    def fake_generate_text(**kwargs):
        calls.append(kwargs)
        return """
        {
          "user_upserts": [],
          "org_upserts": [
            {
              "subject_id": "org_1",
              "write": {
                "content": "When run_tests fails because pytest is missing, install pytest and rerun.",
                "kind": "failure_recovery",
                "salience": 0.88,
                "dedupe_key": "trace:org:pytest_missing",
                "metadata": {"trace_pattern": "failure_recovery"},
                "source_refs": ["openai_agents:trace_1"]
              }
            }
          ],
          "user_archives": [],
          "org_archives": [],
          "merge_operations": [],
          "rationale": ["Observed a recoverable failure pattern."],
          "source_refs": ["openai_agents:trace_1"]
        }
        """

    monkeypatch.setattr("tokentrim.consolidator.agent.generate_text", fake_generate_text)

    result = run_session_consolidation(
        memory_store=memory_store,
        trace_store=trace_store,
        session_id="session_1",
        user_id="user_1",
        org_id="org_1",
        apply=True,
        agent=ModelConsolidatorAgent(model="gpt-4.1-mini"),
    )

    assert len(calls) == 1
    assert calls[0]["model"] == "openai/gpt-4.1-mini"
    assert "response_format" in calls[0]
    assert result.apply_result is not None
    stored_org = memory_store.list_memories(scope="org", subject_id="org_1")
    assert len(stored_org) == 1
    assert stored_org[0].dedupe_key == "trace:org:pytest_missing"


def test_build_consolidation_bundle_exposes_memories_and_traces() -> None:
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")
    trace = trace_store.list_traces(user_id="user_1", session_id="session_1")[0]
    bundle = build_consolidation_bundle(
        ConsolidationInput(
            session_id="session_1",
            user_id="user_1",
            org_id="org_1",
            traces=(trace,),
            session_memories=(
                MemoryRecord(
                    memory_id="mem_session_1",
                    scope="session",
                    subject_id="session_1",
                    kind="constraint",
                    content="Avoid hard reset",
                ),
            ),
            user_memories=(),
            org_memories=(),
        )
    )

    assert len(bundle.messages) == 2
    assert bundle.messages[0].role == "meta"
    assert bundle.messages[1].role == "session_memory"
    assert bundle.label == "offline_bundle"
    assert len(bundle.traces) == 1


def test_agentic_consolidator_agent_uses_runtime_and_validates_plan(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(self, context_prompt, root_prompt=None, *, system_prompt=None):
        captured["context"] = context_prompt
        captured["root_prompt"] = root_prompt
        captured["system_prompt"] = system_prompt
        return """
        {
          "user_upserts": [],
          "org_upserts": [
            {
              "subject_id": "org_1",
              "write": {
                "content": "Repeated fix: install pytest and rerun tests.",
                "kind": "failure_recovery",
                "dedupe_key": "trace:org:pytest_fix",
                "source_refs": ["openai_agents:trace_1"]
              }
            }
          ],
          "user_archives": [],
          "org_archives": [],
          "merge_operations": [],
          "rationale": ["Observed repeated repair workflow."],
          "source_refs": ["openai_agents:trace_1"]
        }
        """

    monkeypatch.setattr("tokentrim.consolidator.agent.LocalConsolidatorRuntime.run", fake_run)

    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")

    result = run_session_consolidation(
        memory_store=memory_store,
        trace_store=trace_store,
        session_id="session_1",
        user_id="user_1",
        org_id="org_1",
        apply=True,
        agent=AgenticConsolidatorAgent(model="gpt-4.1-mini"),
    )

    assert "offline memory consolidator agent" in str(captured["system_prompt"])
    assert "REPL variable `bundle`" in str(captured["system_prompt"])
    assert "Build the final durable-memory edit plan from `bundle`" in str(captured["root_prompt"])
    assert result.apply_result is not None
    assert memory_store.list_memories(scope="org", subject_id="org_1")[0].dedupe_key == "trace:org:pytest_fix"


def test_agentic_consolidator_agent_keeps_trace_synthesis_when_runtime_returns_noop(monkeypatch) -> None:
    def fake_run(self, context_prompt, root_prompt=None, *, system_prompt=None):
        return """
        {
          "user_upserts": [],
          "org_upserts": [],
          "user_archives": [],
          "org_archives": [],
          "merge_operations": [],
          "rationale": ["No durable edits from the model response."],
          "source_refs": []
        }
        """

    monkeypatch.setattr("tokentrim.consolidator.agent.LocalConsolidatorRuntime.run", fake_run)

    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_execute_command_trace(trace_store, trace_id="openai_agents:trace_pmars")

    result = run_session_consolidation(
        memory_store=memory_store,
        trace_store=trace_store,
        session_id="session_1",
        user_id="user_1",
        org_id=None,
        apply=True,
        agent=AgenticConsolidatorAgent(model="gpt-4.1-mini"),
    )

    assert result.apply_result is not None
    stored = memory_store.list_memories(scope="user", subject_id="user_1")
    assert len(stored) == 1
    assert "X11/Xlib.h" in stored[0].content
    assert any("trace-derived repair heuristic" in item for item in result.plan.rationale)


def test_agentic_consolidator_agent_accepts_scalar_rationale_and_source_refs(monkeypatch) -> None:
    def fake_run(self, context_prompt, root_prompt=None, *, system_prompt=None):
        return """
        {
          "user_upserts": [
            {
              "subject_id": "user_1",
              "write": {
                "content": "Remember to disable X11 and install debhelper before rebuilding pmars.",
                "kind": "failure_recovery",
                "dedupe_key": "trace:user:pmars:x11",
                "source_refs": "openai_agents:trace_1"
              }
            }
          ],
          "org_upserts": [],
          "user_archives": [],
          "org_archives": [],
          "merge_operations": [],
          "rationale": "Observed the same repair pattern in the trace.",
          "source_refs": "openai_agents:trace_1"
        }
        """

    monkeypatch.setattr("tokentrim.consolidator.agent.LocalConsolidatorRuntime.run", fake_run)

    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")

    result = run_session_consolidation(
        memory_store=memory_store,
        trace_store=trace_store,
        session_id="session_1",
        user_id="user_1",
        org_id="org_1",
        apply=True,
        agent=AgenticConsolidatorAgent(model="gpt-4.1-mini"),
    )

    assert result.apply_result is not None
    stored = memory_store.list_memories(scope="user", subject_id="user_1")[0]
    assert "openai_agents:trace_1" in stored.source_refs
    assert "Observed the same repair pattern in the trace." in result.plan.rationale
    assert any("trace-derived failure/recovery pattern" in item for item in result.plan.rationale)
    assert "openai_agents:trace_1" in result.plan.source_refs
    assert "openai_agents:trace_1:fail" in result.plan.source_refs


def test_build_agentic_consolidator_system_prompt_mentions_repl_workflow() -> None:
    prompt = build_agentic_consolidator_system_prompt()

    assert "REPL variable `bundle`" in prompt
    assert "FINAL(...)" in prompt
    assert "return the final JSON object directly" in prompt


def test_session_consolidation_job_runs_with_model_agent(monkeypatch) -> None:
    def fake_generate_text(**kwargs):
        return """
        {
          "user_upserts": [{"subject_id": "user_1", "write": {"content": "Concise", "kind": "preference", "dedupe_key": "pref:concise"}}],
          "org_upserts": [],
          "user_archives": [],
          "org_archives": [],
          "merge_operations": [],
          "rationale": ["Stable preference."],
          "source_refs": ["openai_agents:trace_1"]
        }
        """

    monkeypatch.setattr("tokentrim.consolidator.agent.generate_text", fake_generate_text)

    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")

    job = build_model_session_consolidation_job(
        memory_store=memory_store,
        trace_store=trace_store,
        model="gpt-4.1-mini",
        config=ConsolidationJobConfig(apply=True),
    )
    result = job.run(session_id="session_1", user_id="user_1", org_id="org_1")

    assert result.apply_result is not None
    stored = memory_store.list_memories(scope="user", subject_id="user_1")
    assert any(memory.dedupe_key == "pref:concise" for memory in stored)


def test_session_consolidation_job_runs_with_agentic_agent(monkeypatch) -> None:
    def fake_run(self, context_prompt, root_prompt=None, *, system_prompt=None):
        return """
        {
          "user_upserts": [],
          "org_upserts": [{"subject_id": "org_1", "write": {"content": "Run tests before commit.", "kind": "workflow_pattern", "dedupe_key": "workflow:tests_before_commit"}}],
          "user_archives": [],
          "org_archives": [],
          "merge_operations": [],
          "rationale": ["Observed stable org workflow."],
          "source_refs": ["openai_agents:trace_1"]
        }
        """

    monkeypatch.setattr("tokentrim.consolidator.agent.LocalConsolidatorRuntime.run", fake_run)

    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")

    job = build_agentic_session_consolidation_job(
        memory_store=memory_store,
        trace_store=trace_store,
        model="gpt-4.1-mini",
        config=ConsolidationJobConfig(apply=True),
    )
    result = job.run(session_id="session_1", user_id="user_1", org_id="org_1")

    assert result.apply_result is not None
    assert memory_store.list_memories(scope="org", subject_id="org_1")[0].dedupe_key == "workflow:tests_before_commit"


def test_session_consolidation_job_direct_class_works() -> None:
    memory_store = InMemoryMemoryStore()
    trace_store = InMemoryTraceStore()
    _complete_trace(trace_store, trace_id="openai_agents:trace_1")
    job = SessionConsolidationJob(
        memory_store=memory_store,
        trace_store=trace_store,
        agent=DeterministicConsolidatorAgent(),
        config=ConsolidationJobConfig(apply=False),
    )

    result = job.run(session_id="session_1", user_id="user_1", org_id="org_1")

    assert result.apply_result is None
