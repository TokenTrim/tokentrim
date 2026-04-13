from __future__ import annotations

from tokentrim.memory import InMemoryMemoryStore, MemoryRecord, select_memory_candidates


def test_select_memory_candidates_returns_all_without_selector_model() -> None:
    store = InMemoryMemoryStore()
    candidates = (
        MemoryRecord(
            memory_id="mem_1",
            scope="user",
            subject_id="user_1",
            kind="preference",
            content="Prefer concise answers",
        ),
    )

    selected = select_memory_candidates(
        memory_store=store,
        candidates=candidates,
        session_id=None,
        user_id="user_1",
        org_id=None,
        text_query="debug the repo failure",
        selector_model=None,
    )

    assert selected == candidates


def test_select_memory_candidates_falls_back_to_candidates_on_selector_failure(monkeypatch) -> None:
    store = InMemoryMemoryStore()
    candidates = (
        MemoryRecord(
            memory_id="mem_1",
            scope="user",
            subject_id="user_1",
            kind="preference",
            content="Prefer concise answers",
        ),
    )

    def fake_generate_text(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("tokentrim.memory.selector.generate_text", fake_generate_text)

    selected = select_memory_candidates(
        memory_store=store,
        candidates=candidates,
        session_id=None,
        user_id="user_1",
        org_id=None,
        text_query="debug the repo failure",
        selector_model="openai/gpt-4.1-mini",
    )

    assert selected == candidates


def test_select_memory_candidates_uses_manifest_and_selected_ids(monkeypatch) -> None:
    store = InMemoryMemoryStore()
    candidates = (
        MemoryRecord(
            memory_id="mem_1",
            scope="user",
            subject_id="user_1",
            kind="preference",
            content="Prefer concise answers",
            metadata={"title": "Answer style", "description": "User prefers concise answers"},
        ),
        MemoryRecord(
            memory_id="mem_2",
            scope="user",
            subject_id="user_1",
            kind="task_fact",
            content="Repo root is /workspace/app",
            metadata={"title": "Repo root", "description": "Project root path"},
        ),
    )
    for record in candidates:
        store.upsert_memory(record)

    captured: dict[str, object] = {}

    def fake_generate_text(**kwargs):
        captured["messages"] = kwargs["messages"]
        return '{"selected_memory_ids":["mem_2"]}'

    monkeypatch.setattr("tokentrim.memory.selector.generate_text", fake_generate_text)

    selected = select_memory_candidates(
        memory_store=store,
        candidates=candidates,
        session_id=None,
        user_id="user_1",
        org_id=None,
        text_query="where is the repo root?",
        selector_model="openai/gpt-4.1-mini",
    )

    assert selected == (candidates[1],)
    assert "manifest" in str(captured["messages"])
    assert "Repo root" in str(captured["messages"])
