from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tokentrim.errors.base import TokentrimError


def generate_text(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    response_format: dict[str, Any] | None = None,
) -> str:
    try:
        import litellm
    except ImportError as exc:
        raise TokentrimError(
            "litellm is required for model-backed Tokentrim features."
        ) from exc

    completion = litellm.completion
    debug_state = _capture_litellm_debug_state(litellm)
    try:
        completion_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if response_format is not None:
            completion_kwargs["response_format"] = response_format
        if not _should_omit_temperature(model=model, temperature=temperature):
            completion_kwargs["temperature"] = temperature

        _apply_litellm_debug_suppression(litellm)
        response = completion(**completion_kwargs)
    except Exception as exc:
        raise TokentrimError(
            f"LiteLLM completion failed for model '{model}'."
        ) from exc
    finally:
        _restore_litellm_debug_state(litellm, debug_state)

    try:
        return _extract_content(response).strip()
    except Exception as exc:
        raise TokentrimError("LiteLLM response did not contain text content.") from exc


def _should_omit_temperature(*, model: str, temperature: float) -> bool:
    if temperature != 0.0:
        return False

    bare_model = _normalize_temperature_model_name(model)
    return bare_model == "gpt-5" or bare_model.startswith(("gpt-5-", "gpt-5."))


def _normalize_temperature_model_name(model: str) -> str:
    normalized_model = model.strip().lower()
    if "/" in normalized_model:
        _, _, normalized_model = normalized_model.partition("/")
    return normalized_model


def _capture_litellm_debug_state(litellm: Any) -> dict[str, tuple[bool, Any]]:
    return {
        "suppress_debug_info": (hasattr(litellm, "suppress_debug_info"), getattr(litellm, "suppress_debug_info", None)),
        "set_verbose": (hasattr(litellm, "set_verbose"), getattr(litellm, "set_verbose", None)),
    }


def _apply_litellm_debug_suppression(litellm: Any) -> None:
    litellm.suppress_debug_info = True
    if hasattr(litellm, "set_verbose"):
        litellm.set_verbose = False


def _restore_litellm_debug_state(
    litellm: Any,
    debug_state: dict[str, tuple[bool, Any]],
) -> None:
    for name, (had_attr, value) in debug_state.items():
        if had_attr:
            setattr(litellm, name, value)
        elif hasattr(litellm, name):
            delattr(litellm, name)


def _extract_content(response: Any) -> str:
    choices = response["choices"] if isinstance(response, Mapping) else response.choices
    first_choice = choices[0]
    message = (
        first_choice["message"]
        if isinstance(first_choice, Mapping)
        else first_choice.message
    )
    content = message["content"] if isinstance(message, Mapping) else message.content
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    raise ValueError("No message content found.")
