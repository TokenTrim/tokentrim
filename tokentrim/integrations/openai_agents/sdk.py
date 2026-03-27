from __future__ import annotations

import inspect
from typing import Any

from tokentrim.errors.base import TokentrimError


def requires_adapter(*, token_budget: int | None, steps: tuple[Any, ...]) -> bool:
    return token_budget is not None or bool(steps)


def load_agents_sdk() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from agents import RunConfig
        from agents.handoffs import HandoffInputData
        from agents.run import ModelInputData
    except ImportError as exc:
        raise TokentrimError(
            "openai-agents is required for tokentrim.integrations.openai_agents."
        ) from exc

    return RunConfig, ModelInputData, HandoffInputData


async def resolve_maybe_awaitable(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
