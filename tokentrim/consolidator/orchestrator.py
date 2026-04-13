from __future__ import annotations

"""Offline durable-memory orchestration for completed sessions."""

from dataclasses import dataclass

from tokentrim.consolidator.agent import ConsolidatorAgent, DeterministicConsolidatorAgent
from tokentrim.consolidator.models import (
    ConsolidationApplyResult,
    ConsolidationInput,
    ConsolidationPlan,
    DurableMemoryWriteScope,
    apply_consolidation_plan,
    restrict_consolidation_plan,
)
from tokentrim.memory.records import MemoryRecord
from tokentrim.memory.store import MemoryStore
from tokentrim.tracing.store import TraceStore


@dataclass(frozen=True, slots=True)
class ConsolidatorRunResult:
    """Result of one offline consolidation run."""

    consolidation_input: ConsolidationInput
    plan: ConsolidationPlan
    apply_result: ConsolidationApplyResult | None


class OfflineMemoryConsolidator:
    """Offline durable-memory orchestrator for one completed session."""

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        trace_store: TraceStore,
        agent: ConsolidatorAgent | None = None,
    ) -> None:
        self._memory_store = memory_store
        self._trace_store = trace_store
        self._agent = agent or DeterministicConsolidatorAgent()

    def build_input(
        self,
        *,
        session_id: str,
        user_id: str,
        org_id: str | None = None,
        trace_limit: int | None = None,
        session_memory_limit: int | None = None,
        user_memory_limit: int | None = None,
        org_memory_limit: int | None = None,
    ) -> ConsolidationInput:
        traces = self._trace_store.list_traces(
            user_id=user_id,
            session_id=session_id,
            limit=trace_limit,
        )
        session_memories = self._list_memories(
            scope="session",
            subject_id=session_id,
            limit=session_memory_limit,
        )
        user_memories = self._list_memories(
            scope="user",
            subject_id=user_id,
            limit=user_memory_limit,
        )
        org_memories = (
            self._list_memories(
                scope="org",
                subject_id=org_id,
                limit=org_memory_limit,
            )
            if org_id is not None
            else ()
        )
        return ConsolidationInput(
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            traces=traces,
            session_memories=session_memories,
            user_memories=user_memories,
            org_memories=org_memories,
        )

    def run(
        self,
        *,
        session_id: str,
        user_id: str,
        org_id: str | None = None,
        apply: bool = False,
        write_scope: DurableMemoryWriteScope = "all",
        trace_limit: int | None = None,
        session_memory_limit: int | None = None,
        user_memory_limit: int | None = None,
        org_memory_limit: int | None = None,
    ) -> ConsolidatorRunResult:
        consolidation_input = self.build_input(
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            trace_limit=trace_limit,
            session_memory_limit=session_memory_limit,
            user_memory_limit=user_memory_limit,
            org_memory_limit=org_memory_limit,
        )
        plan = restrict_consolidation_plan(
            plan=self._agent.build_plan(consolidation_input),
            write_scope=write_scope,
        )
        apply_result = (
            apply_consolidation_plan(plan=plan, memory_store=self._memory_store) if apply else None
        )
        return ConsolidatorRunResult(
            consolidation_input=consolidation_input,
            plan=plan,
            apply_result=apply_result,
        )

    def _list_memories(
        self,
        *,
        scope: str,
        subject_id: str,
        limit: int | None,
    ) -> tuple[MemoryRecord, ...]:
        return self._memory_store.list_memories(
            scope=scope,
            subject_id=subject_id,
            limit=limit,
        )


def run_session_consolidation(
    *,
    memory_store: MemoryStore,
    trace_store: TraceStore,
    session_id: str,
    user_id: str,
    org_id: str | None = None,
    apply: bool = False,
    agent: ConsolidatorAgent | None = None,
    write_scope: DurableMemoryWriteScope = "all",
    trace_limit: int | None = None,
    session_memory_limit: int | None = None,
    user_memory_limit: int | None = None,
    org_memory_limit: int | None = None,
) -> ConsolidatorRunResult:
    """Run one session consolidation through the offline orchestrator."""
    consolidator = OfflineMemoryConsolidator(
        memory_store=memory_store,
        trace_store=trace_store,
        agent=agent,
    )
    return consolidator.run(
        session_id=session_id,
        user_id=user_id,
        org_id=org_id,
        apply=apply,
        write_scope=write_scope,
        trace_limit=trace_limit,
        session_memory_limit=session_memory_limit,
        user_memory_limit=user_memory_limit,
        org_memory_limit=org_memory_limit,
    )
