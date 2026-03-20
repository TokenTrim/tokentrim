from __future__ import annotations

from tokentrim._copy import clone_messages, clone_tools, freeze_messages, freeze_tools


def test_clone_messages_returns_independent_copies() -> None:
    original = [{"role": "user", "content": "hello"}]

    cloned = clone_messages(original)
    cloned[0]["content"] = "changed"

    assert original == [{"role": "user", "content": "hello"}]


def test_clone_tools_deep_copies_nested_schema() -> None:
    original = [
        {
            "name": "search",
            "description": "search docs",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]

    cloned = clone_tools(original)
    cloned[0]["input_schema"]["properties"]["query"]["type"] = "number"

    assert original[0]["input_schema"]["properties"]["query"]["type"] == "string"


def test_freeze_helpers_return_tuples() -> None:
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"name": "search", "description": "docs", "input_schema": {}}]

    frozen_messages = freeze_messages(messages)
    frozen_tools = freeze_tools(tools)

    assert isinstance(frozen_messages, tuple)
    assert isinstance(frozen_tools, tuple)

