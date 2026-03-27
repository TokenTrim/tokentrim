from collections.abc import Sequence
from copy import deepcopy
from typing import TypeVar

from tokentrim.types.message import Message
from tokentrim.types.tool import Tool

T = TypeVar("T")


def clone_items(items: Sequence[T]) -> list[T]:
    return deepcopy(list(items))


def freeze_items(items: Sequence[T]) -> tuple[T, ...]:
    return tuple(clone_items(items))


def clone_messages(messages: Sequence[Message]) -> list[Message]:
    return clone_items(messages)


def clone_tools(tools: Sequence[Tool]) -> list[Tool]:
    return clone_items(tools)


def freeze_messages(messages: Sequence[Message]) -> tuple[Message, ...]:
    return freeze_items(messages)


def freeze_tools(tools: Sequence[Tool]) -> tuple[Tool, ...]:
    return freeze_items(tools)
