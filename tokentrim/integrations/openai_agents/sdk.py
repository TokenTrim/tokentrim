from __future__ import annotations

import inspect
from typing import Any

from tokentrim.errors.base import TokentrimError


def requires_trim_hooks(
    *,
    token_budget: int | None,
    steps: tuple[Any, ...],
    memory_store: Any | None = None,
    agent_aware_memory: bool = False,
) -> bool:
    return (
        token_budget is not None
        or bool(steps)
        or memory_store is not None
        or agent_aware_memory
    )


def requires_adapter(
    *,
    token_budget: int | None,
    steps: tuple[Any, ...],
    memory_store: Any | None,
    agent_aware_memory: bool,
    trace_store: Any | None,
) -> bool:
    return (
        requires_trim_hooks(
            token_budget=token_budget,
            steps=steps,
            memory_store=memory_store,
            agent_aware_memory=agent_aware_memory,
        )
        or trace_store is not None
    )


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
