from __future__ import annotations

from dataclasses import dataclass

from tokentrim.memory import MemoryQuery
from tokentrim.memory.agent_aware import (
    build_agent_aware_memory_prompt,
    build_session_memory_tools,
)
from tokentrim.memory.selector import select_memory_candidates
from tokentrim.memory.formatting import render_injected_memory_message
from tokentrim.memory.records import MemoryRecord
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.types.message import Message, get_text_content
from tokentrim.types.state import PipelineState
from tokentrim.types.tool import Tool


@dataclass(frozen=True, slots=True)
class AgentAwareMemory(Transform):
    """Expose Tokentrim's standard session-memory capability to the agent."""

    @property
    def name(self) -> str:
        return "agent_aware_memory"

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        if request.memory_store is None or request.session_id is None:
            return state
        prompt = build_agent_aware_memory_prompt(
            memory_store=request.memory_store,
            session_id=request.session_id,
        )
        tools = build_session_memory_tools()
        return PipelineState(
            context=_insert_system_message(
                state.context,
                message_content=prompt,
                slot="secondary_system",
            ),
            tools=_upsert_tools(state.tools, tools=tools),
        )


@dataclass(frozen=True, slots=True)
class InjectMemory(Transform):
    """Runtime-owned memory injection.

    This transform performs bounded pre-turn retrieval and context injection.
    It is part of system-owned context construction and is not intended to be
    invoked by the agent as a tool.
    """

    max_memories: int = 5
    max_memory_tokens: int = 256
    tokenizer_model: str | None = None

    @property
    def name(self) -> str:
        return "inject_memory"

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
    ) -> Transform:
        return InjectMemory(
            max_memories=self.max_memories,
            max_memory_tokens=self.max_memory_tokens,
            tokenizer_model=self.tokenizer_model if self.tokenizer_model is not None else tokenizer_model,
        )

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        if request.memory_store is None:
            return state
        if self.max_memories < 1 or self.max_memory_tokens < 1:
            return state

        text_query = _build_text_query(messages=state.context, task_hint=request.task_hint)
        candidates = request.memory_store.query_memories(
            MemoryQuery(
                session_id=request.session_id,
                user_id=request.user_id,
                org_id=request.org_id,
                k=self.max_memories,
                text_query=text_query,
            )
        )
        if not candidates:
            return state

        selected_candidates = select_memory_candidates(
            memory_store=request.memory_store,
            candidates=tuple(candidate for candidate in candidates if isinstance(candidate, MemoryRecord)),
            session_id=request.session_id,
            user_id=request.user_id,
            org_id=request.org_id,
            text_query=text_query,
            selector_model=request.injector_model,
        )
        if not selected_candidates:
            return state

        memory_content = _build_memory_message_content(
            candidates=selected_candidates,
            current_messages=state.context,
            token_budget=request.token_budget,
            max_memory_tokens=self.max_memory_tokens,
            tokenizer_model=self.tokenizer_model,
        )
        if memory_content is None:
            return state

        return PipelineState(
            context=_insert_system_message(
                state.context,
                message_content=memory_content,
                slot="secondary_system",
            ),
            tools=state.tools,
        )


def _build_text_query(*, messages: list[Message], task_hint: str | None) -> str | None:
    latest_user_text = next(
        (
            get_text_content(message).strip()
            for message in reversed(messages)
            if message.get("role") == "user" and get_text_content(message).strip()
        ),
        "",
    )
    parts = [part.strip() for part in (task_hint or "", latest_user_text) if part and part.strip()]
    if not parts:
        return None
    return "\n".join(parts)


def _build_memory_message_content(
    *,
    candidates: tuple[object, ...],
    current_messages: list[Message],
    token_budget: int | None,
    max_memory_tokens: int,
    tokenizer_model: str | None,
) -> str | None:
    typed_candidates = tuple(
        candidate for candidate in candidates if isinstance(candidate, MemoryRecord)
    )
    return render_injected_memory_message(
        candidates=typed_candidates,
        current_messages=current_messages,
        token_budget=token_budget,
        max_memory_tokens=max_memory_tokens,
        tokenizer_model=tokenizer_model,
    )


def _insert_system_message(
    current_messages: list[Message],
    *,
    message_content: str,
    slot: str,
) -> list[Message]:
    """Insert a system-owned message without duplicating the primary system prompt."""
    prompt_message: Message = {"role": "system", "content": message_content}
    if not current_messages:
        return [prompt_message]
    if current_messages[0].get("role") != "system":
        return [prompt_message, *current_messages]
    if len(current_messages) > 1 and current_messages[1].get("role") == "system":
        if current_messages[1].get("content") == message_content:
            return current_messages
        return [current_messages[0], prompt_message, *current_messages[1:]]
    if slot == "primary_system" and current_messages[0].get("content") == message_content:
        return current_messages
    return [current_messages[0], prompt_message, *current_messages[1:]]


def _upsert_tools(current_tools: list[Tool], *, tools: tuple[Tool, ...]) -> list[Tool]:
    next_tools = list(current_tools)
    for tool in tools:
        next_tools = [existing for existing in next_tools if existing.get("name") != tool.get("name")]
        next_tools.append(tool)
    return next_tools
