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
    completion_options: Mapping[str, Any] | None = None,
) -> str:
    try:
        from litellm import completion
    except ImportError as exc:
        raise TokentrimError(
            "litellm is required for model-backed Tokentrim features."
        ) from exc

    try:
        completion_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if response_format is not None:
            completion_kwargs["response_format"] = response_format
        if not _should_omit_temperature(model=model, temperature=temperature):
            completion_kwargs["temperature"] = temperature
        if completion_options is not None:
            completion_kwargs.update(dict(completion_options))

        response = completion(**completion_kwargs)
    except Exception as exc:
        raise TokentrimError(
            f"LiteLLM completion failed for model '{model}'."
        ) from exc

    try:
        return _extract_content(response).strip()
    except Exception as exc:
        raise TokentrimError("LiteLLM response did not contain text content.") from exc


def _should_omit_temperature(*, model: str, temperature: float) -> bool:
    normalized_model = model.strip().lower()
    if temperature != 0.0:
        return False

    return normalized_model == "gpt-5" or normalized_model.startswith("gpt-5-")


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
