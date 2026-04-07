from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tokentrim.memory.extraction import build_trace_memory_candidate
from tokentrim.memory.writer import MemoryWriteCandidate
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.types.message import Message
from tokentrim.types.state import PipelineState
from tokentrim.working_state import WorkingState, find_working_state

_REMEMBER_RE = re.compile(r"(?i)\bremember(?:\s+this|\s+that)?\b")


class MemoryWritePolicy(ABC):
    @abstractmethod
    def build_candidate(
        self,
        *,
        state: PipelineState,
        request: PipelineRequest,
    ) -> MemoryWriteCandidate | None:
        """Return a memory candidate to persist, or None if nothing should be written."""


@dataclass(frozen=True, slots=True)
class DefaultMemoryWritePolicy(MemoryWritePolicy):
    include_checkpoints: bool = True
    include_active_errors: bool = True
    include_trace_extraction: bool = True

    def build_candidate(
        self,
        *,
        state: PipelineState,
        request: PipelineRequest,
    ) -> MemoryWriteCandidate | None:
        explicit_candidate = _explicit_memory_candidate(state.context)
        if explicit_candidate is not None:
            return explicit_candidate

        working_state = find_working_state(state.context)
        current_candidate = self._build_current_state_candidate(working_state, request.task_hint)
        if current_candidate is not None:
            return current_candidate

        return self._build_trace_candidate(request)

    def _build_current_state_candidate(
        self,
        working_state: WorkingState | None,
        task_hint: str | None,
    ) -> MemoryWriteCandidate | None:
        if working_state is None:
            return None
        if self.include_active_errors and working_state.active_error:
            return _build_error_candidate(working_state, task_hint)
        if self.include_checkpoints:
            return _build_checkpoint_candidate(working_state, task_hint)
        return None

    def _build_trace_candidate(self, request: PipelineRequest) -> MemoryWriteCandidate | None:
        if not self.include_trace_extraction:
            return None
        if request.trace_store is None or not request.user_id or not request.session_id:
            return None
        return build_trace_memory_candidate(
            request.trace_store.list_traces(user_id=request.user_id, session_id=request.session_id, limit=8),
            task_hint=request.task_hint,
        )


def _explicit_memory_candidate(messages: list[Message]) -> MemoryWriteCandidate | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message["role"] != "user":
            continue
        content = message["content"].strip()
        if not content or not _REMEMBER_RE.search(content):
            continue
        normalized = _strip_remember_prefix(content)
        if normalized:
            return MemoryWriteCandidate(
                content=normalized,
                metadata={"kind": "explicit"},
            )
        for previous in range(index - 1, -1, -1):
            prior = messages[previous]
            if prior["role"] == "system":
                continue
            candidate = prior["content"].strip()
            if candidate:
                return MemoryWriteCandidate(
                    content=candidate,
                    metadata={"kind": "explicit", "source_role": prior["role"]},
                )
        return None
    return None


def _strip_remember_prefix(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return ""
    joined = " ".join(lines)
    stripped = re.sub(r"(?i)^remember(?:\s+this|\s+that)?[:,]?\s*", "", joined).strip()
    return "" if stripped.lower() in {"remember", "remember this", "remember that"} else stripped
def _build_error_candidate(working_state: WorkingState, task_hint: str | None) -> MemoryWriteCandidate | None:
    if not working_state.active_error:
        return None
    parts = [f"Active error: {working_state.active_error}"]
    if working_state.active_files:
        parts.append(f"Active files: {', '.join(working_state.active_files)}")
    if working_state.latest_command:
        parts.append(f"Latest command: {working_state.latest_command}")
    if task_hint:
        parts.append(f"Task hint: {task_hint}")
    if working_state.next_step:
        parts.append(f"Next step: {working_state.next_step}")
    return MemoryWriteCandidate(
        content="\n".join(parts),
        metadata={"kind": "active_error"},
    )


def _build_checkpoint_candidate(working_state: WorkingState, task_hint: str | None) -> MemoryWriteCandidate | None:
    lines: list[str] = []
    if working_state.goal:
        lines.append(f"Goal: {working_state.goal}")
    if working_state.current_step:
        lines.append(f"Current step: {working_state.current_step}")
    if working_state.active_files:
        lines.append(f"Active files: {', '.join(working_state.active_files)}")
    if working_state.latest_command:
        lines.append(f"Latest command: {working_state.latest_command}")
    if working_state.constraints:
        lines.append(f"Constraints: {' | '.join(working_state.constraints)}")
    if working_state.next_step:
        lines.append(f"Next step: {working_state.next_step}")
    if task_hint:
        lines.append(f"Task hint: {task_hint}")
    if not lines:
        return None
    return MemoryWriteCandidate(
        content="\n".join(lines),
        metadata={"kind": "checkpoint"},
    )
