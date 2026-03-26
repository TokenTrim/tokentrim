from __future__ import annotations

import json
from math import ceil
from typing import Any

from tokentrim.types.message import Message
from tokentrim.types.tool import Tool


def count_message_tokens(messages: list[Message], model: str | None) -> int:
    serialized = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    return _count_text_tokens(serialized, model)


def count_tool_tokens(tools: list[Tool], model: str | None) -> int:
    serialized = json.dumps(tools, sort_keys=True)
    return _count_text_tokens(serialized, model)


def _count_text_tokens(text: str, model: str | None) -> int:
    if not text:
        return 0

    if model:
        encoding = _get_encoding_for_model(model)
        if encoding is not None:
            return len(encoding.encode(text))

    return ceil(len(text) / 4)


def _get_encoding_for_model(model: str) -> Any | None:
    try:
        import tiktoken
    except ImportError:
        return None

    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return None

