from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from tokentrim.salience import extract_query_terms, score_text_salience
from tokentrim.types.message import Message

_ERROR_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^.*(?:error|exception|traceback|failed|failure|enoent|permission denied).*$"
)
_SUCCESS_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:fixed|resolved|success|succeeded|works now|passed|green|done)\b"
)
_TERMINAL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)(?:\[command\]|\[terminal\]|\[output\]|\[exit_code\]|traceback|stderr|stdout|^\$\s+)"
)
_CONSTRAINT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:do not|don't|must|only|avoid|never|without)\b"
)
_PLAN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:i(?:'ll| will)|let me|next|plan|first|then|going to|review|inspect|update|rerun|check|implement)\b"
)
_PATH_RE: Final[re.Pattern[str]] = re.compile(r"(?:^|[\s(])((?:~|/|\.\.?/)[^\s:;,)\]]+)")
_BACKTICK_RE: Final[re.Pattern[str]] = re.compile(r"`([^`\n]+)`")

GroupKind = Literal["protected", "tool_round", "assistant_plan", "message"]


@dataclass(frozen=True, slots=True)
class ContextEditConfig:
    keep_latest_tool_round: bool = True
    collapse_repeated_assistant_plans: bool = True
    drop_resolved_errors: bool = True
    drop_completed_tool_rounds: bool = True


@dataclass(frozen=True, slots=True)
class ContextEditStats:
    removed_messages: int = 0
    removed_tool_rounds: int = 0
    removed_resolved_errors: int = 0
    removed_redundant_plans: int = 0


@dataclass(frozen=True, slots=True)
class ContextEditResult:
    messages: list[Message]
    stats: ContextEditStats


@dataclass(frozen=True, slots=True)
class MessageGroup:
    messages: tuple[Message, ...]
    kind: GroupKind


@dataclass(frozen=True, slots=True)
class ContextEditor:
    config: ContextEditConfig = ContextEditConfig()

    def apply(self, messages: Sequence[Message]) -> list[Message]:
        return self.edit(messages).messages

    def edit(self, messages: Sequence[Message]) -> ContextEditResult:
        groups = self._build_groups(messages)
        query_terms = self._build_query_terms(messages)
        latest_tool_index = self._find_latest_tool_group(groups)
        kept_groups: list[MessageGroup] = []

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
                query_terms=query_terms,
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
        group: MessageGroup,
        *,
        group_index: int,
        latest_tool_index: int | None,
        saw_later_success: bool,
        kept_recent_plan: bool,
        query_terms: Sequence[str],
    ) -> str | None:
        group_salience = self._group_salience(
            group,
            group_index=group_index,
            total_groups=max(group_index + 1, 1),
            query_terms=query_terms,
        )
        if group.kind == "protected":
            return None
        if self._group_has_constraints(group):
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

    def _build_query_terms(self, messages: Sequence[Message]) -> tuple[str, ...]:
        parts: list[str] = []
        for message in messages[-4:]:
            parts.append(message["content"])
        return extract_query_terms(" ".join(parts))

    def _group_salience(
        self,
        group: MessageGroup,
        *,
        group_index: int,
        total_groups: int,
        query_terms: Sequence[str],
    ) -> int:
        text = "\n".join(message["content"] for message in group.messages)
        recency_rank = max(0, total_groups - group_index - 1)
        return score_text_salience(text, query_terms=query_terms, recency_rank=recency_rank)

    def _build_groups(self, messages: Sequence[Message]) -> list[MessageGroup]:
        groups: list[MessageGroup] = []
        index = 0
        while index < len(messages):
            current = messages[index]
            if self._is_protected_message(current):
                groups.append(MessageGroup(messages=(current,), kind="protected"))
                index += 1
                continue

            next_message = messages[index + 1] if index + 1 < len(messages) else None
            if next_message is not None and self._should_pair(current, next_message):
                group_messages = (current, next_message)
                groups.append(
                    MessageGroup(messages=group_messages, kind=self._classify_group(group_messages))
                )
                index += 2
                continue

            groups.append(
                MessageGroup(messages=(current,), kind=self._classify_group((current,)))
            )
            index += 1

        return groups

    def _find_latest_tool_group(self, groups: Sequence[MessageGroup]) -> int | None:
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
            if _CONSTRAINT_RE.search(next_message["content"]):
                return False
            return True
        if current_role == "user" and next_role == "assistant":
            if _CONSTRAINT_RE.search(current["content"]):
                return False
            if self._looks_like_tool_or_terminal_content(current["content"]):
                return True
            return not self._looks_like_tool_or_terminal_content(next_message["content"])
        return False

    def _classify_group(self, messages: tuple[Message, ...]) -> GroupKind:
        if any(self._looks_like_tool_or_terminal_content(message["content"]) for message in messages):
            return "tool_round"
        if len(messages) == 1 and messages[0]["role"] == "assistant":
            if self._looks_like_assistant_plan(messages[0]["content"]):
                return "assistant_plan"
        return "message"

    def _group_has_error(self, group: MessageGroup) -> bool:
        return any(_ERROR_RE.search(message["content"]) for message in group.messages)

    def _group_has_success(self, group: MessageGroup) -> bool:
        return any(_SUCCESS_RE.search(message["content"]) for message in group.messages)

    def _group_has_constraints(self, group: MessageGroup) -> bool:
        return any(
            message["role"] == "user" and _CONSTRAINT_RE.search(message["content"])
            for message in group.messages
        )

    def _group_has_artifacts(self, group: MessageGroup) -> bool:
        return any(
            _PATH_RE.search(message["content"]) or _BACKTICK_RE.search(message["content"])
            for message in group.messages
        )

    def _looks_like_assistant_plan(self, content: str) -> bool:
        return bool(
            _PLAN_RE.search(content)
            and not self._looks_like_tool_or_terminal_content(content)
        )

    def _looks_like_tool_or_terminal_content(self, content: str) -> bool:
        return bool(_TERMINAL_RE.search(content))

    def _is_protected_message(self, message: Message) -> bool:
        return message["role"] == "system"
