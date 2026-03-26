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
        from litellm import completion
    except ImportError as exc:
        raise TokentrimError(
            "litellm is required for model-backed Tokentrim features."
        ) from exc

    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
    except Exception as exc:
        raise TokentrimError(
            f"LiteLLM completion failed for model '{model}'."
        ) from exc

    try:
        return _extract_content(response).strip()
    except Exception as exc:
        raise TokentrimError("LiteLLM response did not contain text content.") from exc


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

