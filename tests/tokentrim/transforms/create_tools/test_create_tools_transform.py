from __future__ import annotations

import pytest

from tokentrim.errors.base import TokentrimError
from tokentrim.transforms.create_tools import CreateTools
from tokentrim.pipeline.requests import ToolsRequest


def _request(*, task_hint: str | None) -> ToolsRequest:
    return ToolsRequest(
        tools=tuple(),
        task_hint=task_hint,
        token_budget=None,
        steps=(CreateTools(),),
    )


def test_creator_drops_duplicate_names(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CreateTools(model="creator-model")
    tools = [
        {
            "name": "search",
            "description": "search docs",
            "input_schema": {"type": "object"},
        }
    ]

    monkeypatch.setattr(
        "tokentrim.transforms.create_tools.transform.generate_text",
        lambda **kwargs: (
            '{"tools": ['
            '{"name": "search", "description": "duplicate", "input_schema": {}}, '
            '{"name": "lookup", "description": "new tool", "input_schema": {"type": "object"}}'
            "]}"
        ),
    )

    result = step.run(tools, _request(task_hint="investigate"))

    assert result == [
        {
            "name": "search",
            "description": "search docs",
            "input_schema": {"type": "object"},
        },
        {
            "name": "lookup",
            "description": "new tool",
            "input_schema": {"type": "object"},
        },
    ]


def test_creator_is_noop_when_no_tools_and_no_task_hint() -> None:
    step = CreateTools(model="creator-model")

    result = step.run([], _request(task_hint=None))

    assert result == []


def test_creator_raises_when_model_is_missing() -> None:
    step = CreateTools(model=None)

    with pytest.raises(TokentrimError) as exc_info:
        step.run([], _request(task_hint="investigate"))

    assert "no tool creation model" in str(exc_info.value)


def test_creator_raises_for_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    step = CreateTools(model="creator-model")

    monkeypatch.setattr(
        "tokentrim.transforms.create_tools.transform.generate_text",
        lambda **kwargs: "not json",
    )

    with pytest.raises(TokentrimError):
        step.run([], _request(task_hint="investigate"))


def test_creator_raises_for_invalid_top_level_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CreateTools(model="creator-model")

    monkeypatch.setattr(
        "tokentrim.transforms.create_tools.transform.generate_text",
        lambda **kwargs: '{"tools": "not-a-list"}',
    )

    with pytest.raises(TokentrimError):
        step.run([], _request(task_hint="investigate"))


def test_creator_raises_for_invalid_tool_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CreateTools(model="creator-model")

    monkeypatch.setattr(
        "tokentrim.transforms.create_tools.transform.generate_text",
        lambda **kwargs: '{"tools": [{"name": "broken", "description": 1, "input_schema": {}}]}',
    )

    with pytest.raises(TokentrimError):
        step.run([], _request(task_hint="investigate"))


def test_creator_deduplicates_repeated_generated_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = CreateTools(model="creator-model")

    monkeypatch.setattr(
        "tokentrim.transforms.create_tools.transform.generate_text",
        lambda **kwargs: (
            '{"tools": ['
            '{"name": "lookup", "description": "first", "input_schema": {}}, '
            '{"name": "lookup", "description": "second", "input_schema": {}}'
            "]}"
        ),
    )

    result = step.run([], _request(task_hint="investigate"))

    assert result == [
        {
            "name": "lookup",
            "description": "first",
            "input_schema": {},
        }
    ]
