from __future__ import annotations

import builtins
import sys
from math import ceil
from types import SimpleNamespace

import pytest

from tokentrim.core.token_counting import _get_encoding_for_model, count_message_tokens, count_tool_tokens


class FakeEncoding:
    def encode(self, text: str) -> list[int]:
        return [1, 2, 3, 4]


def test_count_message_tokens_returns_zero_for_empty_payload() -> None:
    assert count_message_tokens([], model=None) == 0


def test_count_message_tokens_uses_character_fallback_when_model_is_none() -> None:
    messages = [{"role": "user", "content": "hello world"}]

    result = count_message_tokens(messages, model=None)

    assert result == ceil(len("user: hello world") / 4)


def test_count_tool_tokens_uses_tiktoken_encoding_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = SimpleNamespace(encoding_for_model=lambda model: FakeEncoding())
    monkeypatch.setitem(sys.modules, "tiktoken", fake_module)

    result = count_tool_tokens(
        [{"name": "search", "description": "docs", "input_schema": {}}],
        model="token-model",
    )

    assert result == 4


def test_count_tool_tokens_falls_back_when_tokenizer_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def encoding_for_model(model: str) -> FakeEncoding:
        raise KeyError(model)

    fake_module = SimpleNamespace(encoding_for_model=encoding_for_model)
    monkeypatch.setitem(sys.modules, "tiktoken", fake_module)
    tools = [{"name": "search", "description": "docs", "input_schema": {}}]

    result = count_tool_tokens(tools, model="unknown-model")

    assert result == ceil(len('[{"description": "docs", "input_schema": {}, "name": "search"}]') / 4)


def test_get_encoding_for_model_returns_none_when_tiktoken_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "tiktoken", raising=False)

    assert _get_encoding_for_model("missing-model") is None

