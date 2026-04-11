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
    assert not hasattr(sys.modules["litellm"], "suppress_debug_info")


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


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5.4-nano", "openai/gpt-5", "openai/gpt-5.4-nano"])
def test_generate_text_omits_temperature_for_gpt5_models(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    captured: dict[str, object] = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    result = generate_text(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
    )

    assert result == "ok"
    assert captured["model"] == model
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert "temperature" not in captured


def test_generate_text_forwards_completion_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    result = generate_text(
        model="openai/mercury-2",
        messages=[{"role": "user", "content": "hello"}],
        completion_options={
            "api_base": "https://api.inceptionlabs.ai/v1",
            "api_key": "test-key",
        },
    )

    assert result == "ok"
    assert captured["api_base"] == "https://api.inceptionlabs.ai/v1"
    assert captured["api_key"] == "test-key"


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


def test_generate_text_temporarily_suppresses_litellm_debug_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    litellm_module = SimpleNamespace(
        suppress_debug_info=False,
        set_verbose=True,
    )

    def completion(**kwargs):
        observed["kwargs"] = kwargs
        observed["suppress_debug_info"] = litellm_module.suppress_debug_info
        observed["set_verbose"] = litellm_module.set_verbose
        return {"choices": [{"message": {"content": "ok"}}]}

    litellm_module.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", litellm_module)

    result = generate_text(model="openai/gpt-5.4-nano", messages=[{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert observed["suppress_debug_info"] is True
    assert observed["set_verbose"] is False
    assert litellm_module.suppress_debug_info is False
    assert litellm_module.set_verbose is True
