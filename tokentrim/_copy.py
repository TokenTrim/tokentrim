from __future__ import annotations

from copy import deepcopy
from typing import Sequence, cast

from tokentrim.types.message import Message
from tokentrim.types.tool import Tool


def clone_messages(messages: Sequence[Message]) -> list[Message]:
    return [cast(Message, deepcopy(message)) for message in messages]


def clone_tools(tools: Sequence[Tool]) -> list[Tool]:
    return [cast(Tool, deepcopy(tool)) for tool in tools]


def freeze_messages(messages: Sequence[Message]) -> tuple[Message, ...]:
    return tuple(clone_messages(messages))


def freeze_tools(tools: Sequence[Tool]) -> tuple[Tool, ...]:
    return tuple(clone_tools(tools))

