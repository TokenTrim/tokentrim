from __future__ import annotations

from dataclasses import dataclass

from tokentrim.consolidator.agent import (
    AgenticConsolidatorAgent,
    ConsolidatorAgent,
    ModelConsolidatorAgent,
)
from tokentrim.consolidator.models import DurableMemoryWriteScope
from tokentrim.consolidator.orchestrator import ConsolidatorRunResult, OfflineMemoryConsolidator
from tokentrim.memory.store import MemoryStore
from tokentrim.tracing.store import TraceStore


@dataclass(frozen=True, slots=True)
class ConsolidationJobConfig:
    apply: bool = True
    write_scope: DurableMemoryWriteScope = "all"
    trace_limit: int | None = None
    session_memory_limit: int | None = None
    user_memory_limit: int | None = None
    org_memory_limit: int | None = None


class SessionConsolidationJob:
    """Production job wrapper for session-end or scheduled consolidation."""

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        trace_store: TraceStore,
        agent: ConsolidatorAgent,
        config: ConsolidationJobConfig | None = None,
    ) -> None:
        self._consolidator = OfflineMemoryConsolidator(
            memory_store=memory_store,
            trace_store=trace_store,
            agent=agent,
        )
        self._config = config or ConsolidationJobConfig()

    def run(
        self,
        *,
        session_id: str,
        user_id: str,
        org_id: str | None = None,
    ) -> ConsolidatorRunResult:
        return self._consolidator.run(
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            apply=self._config.apply,
            write_scope=self._config.write_scope,
            trace_limit=self._config.trace_limit,
            session_memory_limit=self._config.session_memory_limit,
            user_memory_limit=self._config.user_memory_limit,
            org_memory_limit=self._config.org_memory_limit,
        )


def build_model_session_consolidation_job(
    *,
    memory_store: MemoryStore,
    trace_store: TraceStore,
    model: str,
    backend: str = "openai",
    config: ConsolidationJobConfig | None = None,
) -> SessionConsolidationJob:
    return SessionConsolidationJob(
        memory_store=memory_store,
        trace_store=trace_store,
        agent=ModelConsolidatorAgent(model=model, backend=backend),
        config=config,
    )


def build_agentic_session_consolidation_job(
    *,
    memory_store: MemoryStore,
    trace_store: TraceStore,
    model: str,
    backend: str = "openai",
    config: ConsolidationJobConfig | None = None,
    max_iterations: int = 4,
    max_subcalls: int = 4,
) -> SessionConsolidationJob:
    return SessionConsolidationJob(
        memory_store=memory_store,
        trace_store=trace_store,
        agent=AgenticConsolidatorAgent(
            model=model,
            backend=backend,
            max_iterations=max_iterations,
            max_subcalls=max_subcalls,
        ),
        config=config,
    )
