"""Context editor for deterministic message pruning.

This module provides a fast, deterministic pruning pass that removes
stale context (resolved errors, completed tool rounds, redundant plans)
before model-backed compaction.

Supports multimodal messages by safely extracting text content for analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tokentrim.transforms.compaction.config import ContextEditConfig
from tokentrim.transforms.compaction.patterns import (
    BACKTICK_RE,
    CONSTRAINT_RE,
    ERROR_RE,
    PATH_RE,
    PLAN_RE,
    SUCCESS_RE,
    TERMINAL_RE,
)
from tokentrim.transforms.compaction.types import (
    ContextEditGroupKind,
    ContextEditMessageGroup,
    ContextEditResult,
    ContextEditStats,
)
from tokentrim.types.message import (
    Message,
    get_text_content,
    has_tool_calls,
    is_tool_result,
)


@dataclass(frozen=True, slots=True)
class ContextEditor:
    """Deterministically prune stale context before model-backed compaction."""

    config: ContextEditConfig = ContextEditConfig()

    def apply(self, messages: Sequence[Message]) -> list[Message]:
        return self.edit(messages).messages

    def edit(self, messages: Sequence[Message]) -> ContextEditResult:
        groups = self._build_groups(messages)
        latest_tool_index = self._find_latest_tool_group(groups)
        kept_groups: list[ContextEditMessageGroup] = []

        removed_messages = 0
        removed_tool_rounds = 0
        removed_resolved_errors = 0
        removed_redundant_plans = 0
        saw_later_success = False
        kept_recent_plan = False

        for index in range(len(groups) - 1, -1, -1):
            group = groups[index]
            drop_reason = self._drop_reason(
                group,
                group_index=index,
                latest_tool_index=latest_tool_index,
                saw_later_success=saw_later_success,
                kept_recent_plan=kept_recent_plan,
            )

            if drop_reason is None:
                kept_groups.append(group)
                if group.kind == "assistant_plan":
                    kept_recent_plan = True
            else:
                removed_messages += len(group.messages)
                if drop_reason == "tool_round":
                    removed_tool_rounds += 1
                elif drop_reason == "resolved_error":
                    removed_tool_rounds += 1
                    removed_resolved_errors += 1
                elif drop_reason == "assistant_plan":
                    removed_redundant_plans += 1

            if self._group_has_success(group):
                saw_later_success = True

        kept_groups.reverse()
        edited_messages = [dict(message) for group in kept_groups for message in group.messages]
        return ContextEditResult(
            messages=edited_messages,
            stats=ContextEditStats(
                removed_messages=removed_messages,
                removed_tool_rounds=removed_tool_rounds,
                removed_resolved_errors=removed_resolved_errors,
                removed_redundant_plans=removed_redundant_plans,
            ),
        )

    def _drop_reason(
        self,
        group: ContextEditMessageGroup,
        *,
        group_index: int,
        latest_tool_index: int | None,
        saw_later_success: bool,
        kept_recent_plan: bool,
    ) -> str | None:
        if group.kind == "protected":
            return None
        if self._group_has_constraints(group):
            return None
        # Never drop groups with tool calls/results that might orphan counterparts
        if self._group_has_tool_calls(group):
            return None

        if group.kind == "assistant_plan" and self.config.collapse_repeated_assistant_plans:
            if kept_recent_plan and not self._group_has_artifacts(group):
                return "assistant_plan"
            return None

        if group.kind != "tool_round":
            return None

        if (
            self.config.keep_latest_tool_round
            and latest_tool_index == group_index
            and not saw_later_success
        ):
            return None

        has_error = self._group_has_error(group)
        if has_error and self.config.drop_resolved_errors and saw_later_success:
            return "resolved_error"
        if self.config.drop_completed_tool_rounds and not has_error:
            return "tool_round"
        return None

    def _build_groups(self, messages: Sequence[Message]) -> list[ContextEditMessageGroup]:
        groups: list[ContextEditMessageGroup] = []
        index = 0
        while index < len(messages):
            current = messages[index]
            if self._is_protected_message(current):
                groups.append(ContextEditMessageGroup(messages=(current,), kind="protected"))
                index += 1
                continue

            next_message = messages[index + 1] if index + 1 < len(messages) else None
            if next_message is not None and self._should_pair(current, next_message):
                group_messages = (current, next_message)
                groups.append(
                    ContextEditMessageGroup(messages=group_messages, kind=self._classify_group(group_messages))
                )
                index += 2
                continue

            groups.append(
                ContextEditMessageGroup(messages=(current,), kind=self._classify_group((current,)))
            )
            index += 1

        return groups

    def _find_latest_tool_group(self, groups: Sequence[ContextEditMessageGroup]) -> int | None:
        for index in range(len(groups) - 1, -1, -1):
            if groups[index].kind == "tool_round":
                return index
        return None

    def _should_pair(self, current: Message, next_message: Message) -> bool:
        if self._is_protected_message(next_message):
            return False
        current_role = current["role"]
        next_role = next_message["role"]

        if current_role == "assistant" and next_role == "user":
            if CONSTRAINT_RE.search(get_text_content(next_message)):
                return False
            return True
        if current_role == "user" and next_role == "assistant":
            if CONSTRAINT_RE.search(get_text_content(current)):
                return False
            if self._looks_like_tool_or_terminal_content(get_text_content(current)):
                return True
            return not self._looks_like_tool_or_terminal_content(get_text_content(next_message))
        return False

    def _classify_group(self, messages: tuple[Message, ...]) -> ContextEditGroupKind:
        if any(self._looks_like_tool_or_terminal_content(get_text_content(message)) for message in messages):
            return "tool_round"
        if len(messages) == 1 and messages[0]["role"] == "assistant":
            if self._looks_like_assistant_plan(get_text_content(messages[0])):
                return "assistant_plan"
        return "message"

    def _group_has_error(self, group: ContextEditMessageGroup) -> bool:
        return any(ERROR_RE.search(get_text_content(message)) for message in group.messages)

    def _group_has_success(self, group: ContextEditMessageGroup) -> bool:
        return any(SUCCESS_RE.search(get_text_content(message)) for message in group.messages)

    def _group_has_constraints(self, group: ContextEditMessageGroup) -> bool:
        return any(
            message["role"] == "user" and CONSTRAINT_RE.search(get_text_content(message))
            for message in group.messages
        )

    def _group_has_artifacts(self, group: ContextEditMessageGroup) -> bool:
        return any(
            PATH_RE.search(get_text_content(message)) or BACKTICK_RE.search(get_text_content(message))
            for message in group.messages
        )

    def _group_has_tool_calls(self, group: ContextEditMessageGroup) -> bool:
        """Check if group contains tool calls or tool results.

        These should not be dropped as they may orphan linked messages.
        """
        return any(
            has_tool_calls(message) or is_tool_result(message)
            for message in group.messages
        )

    def _looks_like_assistant_plan(self, content: str) -> bool:
        return bool(
            PLAN_RE.search(content)
            and not self._looks_like_tool_or_terminal_content(content)
        )

    def _looks_like_tool_or_terminal_content(self, content: str) -> bool:
        return bool(TERMINAL_RE.search(content))

    def _is_protected_message(self, message: Message) -> bool:
        return message["role"] == "system"
