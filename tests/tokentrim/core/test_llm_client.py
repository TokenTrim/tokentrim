from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from tokentrim.core.llm_client import generate_text
from tokentrim.errors.base import TokentrimError


def test_generate_text_raises_when_litellm_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "litellm", raising=False)

    with pytest.raises(TokentrimError) as exc_info:
        generate_text(model="test-model", messages=[])

    assert "litellm is required" in str(exc_info.value)


def test_generate_text_wraps_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completion(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    with pytest.raises(TokentrimError) as exc_info:
        generate_text(model="test-model", messages=[])

    assert "LiteLLM completion failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_generate_text_supports_dict_style_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completion(**kwargs):
        return {"choices": [{"message": {"content": "  hello  "}}]}

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    result = generate_text(model="test-model", messages=[])

    assert result == "hello"


def test_generate_text_supports_object_style_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  world  "))]
    )

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=lambda **kwargs: response),
    )

    result = generate_text(model="test-model", messages=[])

    assert result == "world"


def test_generate_text_raises_when_response_has_no_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {"choices": [{"message": {"content": []}}]}

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=lambda **kwargs: response),
    )

    with pytest.raises(TokentrimError) as exc_info:
        generate_text(model="test-model", messages=[])

    assert "did not contain text content" in str(exc_info.value)

