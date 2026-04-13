from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

from tokentrim.core.copy_utils import freeze_messages, freeze_tools
from tokentrim.errors.base import TokentrimError
from tokentrim.pipeline import PipelineRequest, UnifiedPipeline
from tokentrim.types.message import Message
from tokentrim.types.result import Result
from tokentrim.types.tool import Tool
from tokentrim.transforms import AgentAwareMemory, InjectMemory
from tokentrim.transforms.base import Transform

if TYPE_CHECKING:
    from agents import RunConfig

    from tokentrim.integrations.base import IntegrationAdapter
    from tokentrim.memory import MemoryStore, MemoryRecord, SessionMemoryWriter
    from tokentrim.tracing import PipelineTracer, TraceStore


AdapterConfigT = TypeVar("AdapterConfigT")


class ComposedPipeline:
    """
    Unified compose-first pipeline.

    A composed pipeline runs all steps against a shared request state.
    """

    def __init__(
        self,
        *,
        owner: Tokentrim,
        steps: tuple[Transform, ...],
        pipeline: UnifiedPipeline,
        default_token_budget: int | None,
    ) -> None:
        self._owner = owner
        self._steps = steps
        self._pipeline = pipeline
        self._default_token_budget = default_token_budget

    def apply(
        self,
        payload: list[Message] | list[Tool] | None = None,
        *,
        context: list[Message] | None = None,
        tools: list[Tool] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        org_id: str | None = None,
        task_hint: str | None = None,
        token_budget: int | None = None,
        memory_store: MemoryStore | None = None,
        agent_aware_memory: bool = False,
        trace_store: TraceStore | None = None,
        pipeline_tracer: PipelineTracer | None = None,
    ) -> Result:
        """
        Apply composed steps to the provided payload or payloads.

        `payload` preserves the single-input API.
        `context=` and `tools=` allow mixed pipelines in one call.
        """
        effective_budget = (
            token_budget if token_budget is not None else self._default_token_budget
        )
        default_memory_store = self._owner._resolve_default_memory_store(
            user_id=user_id,
            session_id=session_id,
            org_id=org_id,
        )
        default_trace_store = self._owner._resolve_default_trace_store(
            user_id=user_id,
            session_id=session_id,
        )
        normalized_context, normalized_tools = self._normalize_payloads(
            payload=payload,
            context=context,
            tools=tools,
        )
        effective_steps = self._effective_steps(
            memory_store=memory_store or default_memory_store,
            agent_aware_memory=agent_aware_memory,
        )
        effective_memory_store = memory_store or default_memory_store
        effective_trace_store = trace_store or default_trace_store
        request = PipelineRequest(
            messages=freeze_messages(normalized_context),
            tools=freeze_tools(normalized_tools),
            user_id=user_id,
            session_id=session_id,
            org_id=org_id,
            task_hint=task_hint,
            token_budget=effective_budget,
            memory_store=effective_memory_store,
            agent_aware_memory=agent_aware_memory,
            trace_store=effective_trace_store,
            pipeline_tracer=pipeline_tracer,
            steps=effective_steps,
        )
        return self._pipeline.run(request)

    def _effective_steps(
        self,
        *,
        memory_store: MemoryStore | None,
        agent_aware_memory: bool,
    ) -> tuple[Transform, ...]:
        """Materialize implicit memory steps for the current call.

        Memory injection is system-owned. If a memory store is available, the
        pipeline injects memory unless the caller already installed an explicit
        `inject_memory` step. Agent-aware memory is separate because it changes
        the tool surface exposed to the model.
        """
        effective_steps = self._steps
        if memory_store is None:
            return effective_steps
        if agent_aware_memory and not any(step.name == "agent_aware_memory" for step in effective_steps):
            effective_steps = (AgentAwareMemory(), *effective_steps)
        if any(step.name == "inject_memory" for step in effective_steps):
            return effective_steps
        return (InjectMemory(), *effective_steps)

    def _normalize_payloads(
        self,
        *,
        payload: list[Message] | list[Tool] | None,
        context: list[Message] | None,
        tools: list[Tool] | None,
    ) -> tuple[list[Message], list[Tool]]:
        if payload is not None and (context is not None or tools is not None):
            raise TokentrimError(
                "compose(...).apply(...) accepts either `payload` or `context=`/`tools=`, not both."
            )

        if payload is not None:
            kind = self._infer_kind_from_payload(payload)
            if kind == "tools":
                return [], cast(list[Tool], payload)
            return cast(list[Message], payload), []

        if context is not None or tools is not None:
            return list(context or []), list(tools or [])

        raise TokentrimError(
            "compose(...).apply(...) requires `payload` or at least one of `context=`/`tools=`."
        )

    def _infer_kind_from_payload(self, payload: list[Message] | list[Tool]) -> str:
        if not isinstance(payload, list):
            raise TokentrimError("compose(...).apply(...) payload must be a list.")
        if not payload:
            raise TokentrimError(
                "compose(...).apply(...) cannot infer payload kind from an empty list."
            )
        saw_message = False
        saw_tool = False
        for entry in payload:
            if not isinstance(entry, dict):
                raise TokentrimError("compose(...).apply(...) payload entries must be dicts.")
            is_message = (
                isinstance(entry.get("role"), str) and isinstance(entry.get("content"), str)
            )
            is_tool = (
                isinstance(entry.get("name"), str)
                and isinstance(entry.get("description"), str)
                and isinstance(entry.get("input_schema"), dict)
            )
            if is_message and not is_tool:
                saw_message = True
                continue
            if is_tool and not is_message:
                saw_tool = True
                continue
            raise TokentrimError(
                "compose(...).apply(...) payload entries must all be message-shaped or tool-shaped dicts."
            )

        if saw_message and not saw_tool:
            return "context"
        if saw_tool and not saw_message:
            return "tools"
        raise TokentrimError(
            "compose(...).apply(...) payload entries must not mix messages and tools in one positional payload."
        )

    def to_openai_agents(
        self,
        *,
        token_budget: int | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        org_id: str | None = None,
        apply_to_session_history: bool = False,
        apply_to_handoffs: bool = False,
        memory_store: MemoryStore | None = None,
        agent_aware_memory: bool = False,
        trace_store: TraceStore | None = None,
        config: RunConfig | None = None,
    ) -> RunConfig:
        """
        Build an OpenAI Agents `RunConfig` from this composed pipeline.

        By default this only installs `call_model_input_filter`. Session and
        handoff hooks are opt-in to keep behavior predictable.
        """
        effective_budget = (
            token_budget if token_budget is not None else self._default_token_budget
        )
        from tokentrim.integrations.openai_agents import (
            OpenAIAgentsAdapter,
            OpenAIAgentsOptions,
        )
        effective_memory_store = memory_store or self._owner._resolve_default_memory_store(
            user_id=user_id,
            session_id=session_id,
            org_id=org_id,
        )
        effective_trace_store = trace_store or self._owner._resolve_default_trace_store(
            user_id=user_id,
            session_id=session_id,
        )

        adapter = OpenAIAgentsAdapter(
            options=OpenAIAgentsOptions(
                user_id=user_id,
                session_id=session_id,
                org_id=org_id,
                token_budget=effective_budget,
                memory_store=effective_memory_store,
                agent_aware_memory=agent_aware_memory,
                trace_store=effective_trace_store,
                steps=self._steps,
                apply_to_session_history=apply_to_session_history,
                apply_to_handoffs=apply_to_handoffs,
            )
        )
        return adapter.wrap(self._owner, config=config)


class Tokentrim:
    """
    Local context and tool optimisation for LLM agents.

    The SDK runs inside the caller's process. Model-backed features use LiteLLM
    directly and can be configured independently per feature.
    """

    def __init__(
        self,
        tokenizer: str | None = None,
        *,
        token_budget: int | None = None,
        storage_root: str | Path = ".tokentrim",
        memory_root: str | Path | None = None,
        trace_root: str | Path | None = None,
    ) -> None:
        """Create a Tokentrim client with local-first default stores.

        The default filesystem stores are only activated for calls that carry
        enough scope information to make persistence meaningful: memory requires
        at least one scope id, and traces require both user and session ids.
        """
        from tokentrim.memory import FilesystemMemoryStore
        from tokentrim.tracing import FilesystemTraceStore

        self._default_token_budget = token_budget
        base_root = Path(storage_root)
        self._memory_root = Path(memory_root) if memory_root is not None else base_root / "memory"
        self._trace_root = Path(trace_root) if trace_root is not None else base_root / "traces"
        self._default_memory_store = FilesystemMemoryStore(root_dir=self._memory_root)
        self._default_trace_store = FilesystemTraceStore(root_dir=self._trace_root)
        self._pipeline = UnifiedPipeline(
            tokenizer_model=tokenizer,
        )

    @property
    def default_memory_store(self) -> MemoryStore:
        return self._default_memory_store

    @property
    def default_trace_store(self) -> TraceStore:
        return self._default_trace_store

    @property
    def memory_root(self) -> Path:
        return self._memory_root

    @property
    def trace_root(self) -> Path:
        return self._trace_root

    def _resolve_default_memory_store(
        self,
        *,
        user_id: str | None,
        session_id: str | None,
        org_id: str | None,
    ) -> MemoryStore | None:
        """Enable the default memory store only for scoped requests."""
        if user_id is None and session_id is None and org_id is None:
            return None
        return self._default_memory_store

    def _resolve_default_trace_store(
        self,
        *,
        user_id: str | None,
        session_id: str | None,
    ) -> TraceStore | None:
        """Enable trace capture only when the call is attributable to a session."""
        if user_id is None or session_id is None:
            return None
        return self._default_trace_store

    def compose(self, *steps: Transform) -> ComposedPipeline:
        """Create a composed pipeline and run it with `.apply(...)`."""
        return ComposedPipeline(
            owner=self,
            steps=tuple(steps),
            pipeline=self._pipeline,
            default_token_budget=self._default_token_budget,
        )

    def session_memory_writer(
        self,
        *,
        memory_store: MemoryStore | None = None,
        session_id: str,
    ) -> SessionMemoryWriter:
        """Return an explicit session-memory writer for the host runtime."""
        from tokentrim.memory import SessionMemoryWriter

        return SessionMemoryWriter(memory_store=memory_store or self._default_memory_store, session_id=session_id)

    def write_session_memory(
        self,
        *,
        memory_store: MemoryStore | None = None,
        session_id: str,
        content: str,
        kind: str,
        salience: float = 0.5,
        dedupe_key: str | None = None,
        metadata: dict[str, object] | None = None,
        source_refs: tuple[str, ...] = (),
    ) -> MemoryRecord:
        """Write session-scoped memory explicitly.

        This is the core live-runtime write surface. Tokentrim does not require
        integrations to expose it as an agent tool.
        """
        from tokentrim.memory import write_session_memory

        return write_session_memory(
            memory_store=memory_store or self._default_memory_store,
            session_id=session_id,
            content=content,
            kind=kind,
            salience=salience,
            dedupe_key=dedupe_key,
            metadata=metadata,
            source_refs=source_refs,
        )

    def wrap_integration(
        self,
        adapter: IntegrationAdapter[AdapterConfigT],
        *,
        config: AdapterConfigT | None = None,
    ) -> AdapterConfigT:
        return adapter.wrap(self, config=config)

    def openai_agents_config(
        self,
        *,
        steps: tuple[Transform, ...] = (),
        token_budget: int | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        org_id: str | None = None,
        apply_to_session_history: bool = False,
        apply_to_handoffs: bool = False,
        memory_store: MemoryStore | None = None,
        agent_aware_memory: bool = False,
        trace_store: TraceStore | None = None,
        config: RunConfig | None = None,
    ) -> RunConfig:
        """
        Build an OpenAI Agents `RunConfig` with Tokentrim hooks pre-wired.
        """
        return self.compose(*steps).to_openai_agents(
            token_budget=token_budget,
            user_id=user_id,
            session_id=session_id,
            org_id=org_id,
            apply_to_session_history=apply_to_session_history,
            apply_to_handoffs=apply_to_handoffs,
            memory_store=memory_store,
            agent_aware_memory=agent_aware_memory,
            trace_store=trace_store,
            config=config,
        )
