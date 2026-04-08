"""Microcompact transform for deterministic message compression.

This module provides a fast, deterministic compression pass that runs before
LLM-backed summarization. It groups messages by age and type, then compresses
older groups into compact system messages preserving key artifacts.

Supports multimodal messages by safely extracting text content and preserving
image references in the compacted output.

Integrates salience scoring to preserve high-value content (errors, constraints,
commands) even when it would otherwise be compressed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tokentrim.core.token_counting import count_message_tokens
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.salience import score_text_salience
from tokentrim.transforms.base import Transform
from tokentrim.transforms.compaction.config import (
    MICROCOMPACT_PREFIX,
    SUMMARY_SYSTEM_PREFIX,
    MicrocompactConfig,
)
from tokentrim.transforms.compaction.patterns import (
    BACKTICK_RE,
    COMMAND_RE,
    ERROR_RE,
    EXIT_CODE_RE,
    PATH_RE,
    WHITESPACE_RE,
)
from tokentrim.transforms.compaction.types import (
    AgeBand,
    GroupKind,
    MessageGroup,
    MicrocompactPlan,
    MicrocompactPressure,
)
from tokentrim.types.message import (
    Message,
    extract_image_refs,
    get_text_content,
    has_images,
    has_tool_calls,
    is_tool_result,
)
from tokentrim.types.state import PipelineState


@dataclass(frozen=True, slots=True)
class MicrocompactOrchestrator:
    """Deterministically compress bulky older groups before summarization."""

    config: MicrocompactConfig = MicrocompactConfig()
    tokenizer_model: str | None = None

    def apply(
        self,
        messages: list[Message],
        *,
        token_budget: int | None = None,
        pressure: MicrocompactPressure | None = None,
    ) -> list[Message]:
        return self.plan(messages, token_budget=token_budget, pressure=pressure).messages

    def plan(
        self,
        messages: list[Message],
        *,
        token_budget: int | None = None,
        pressure: MicrocompactPressure | None = None,
    ) -> MicrocompactPlan:
        original_messages = [dict(message) for message in messages]
        original_tokens = count_message_tokens(original_messages, self.tokenizer_model)
        effective_pressure = self._resolve_pressure(
            original_messages,
            token_budget=token_budget,
            pressure=pressure,
        )

        if len(original_messages) < self.config.min_messages:
            return MicrocompactPlan(
                messages=original_messages,
                original_tokens=original_tokens,
                compacted_tokens=original_tokens,
                groups_seen=0,
                groups_compacted=0,
            )

        groups = self._build_groups(original_messages, pressure=effective_pressure)
        compacted_messages: list[Message] = []
        groups_compacted = 0

        for group in groups:
            if self._should_compact_group(group, pressure=effective_pressure):
                compacted_messages.append(self._compact_group(group))
                groups_compacted += 1
                continue
            compacted_messages.extend(dict(message) for message in group.messages)

        compacted_tokens = count_message_tokens(compacted_messages, self.tokenizer_model)
        if original_tokens - compacted_tokens < self.config.min_tokens_saved:
            return MicrocompactPlan(
                messages=original_messages,
                original_tokens=original_tokens,
                compacted_tokens=original_tokens,
                groups_seen=len(groups),
                groups_compacted=0,
            )

        return MicrocompactPlan(
            messages=compacted_messages,
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            groups_seen=len(groups),
            groups_compacted=groups_compacted,
        )

    def with_tokenizer(self, tokenizer_model: str | None) -> MicrocompactOrchestrator:
        return replace(self, tokenizer_model=tokenizer_model)

    def _resolve_pressure(
        self,
        messages: list[Message],
        *,
        token_budget: int | None,
        pressure: MicrocompactPressure | None,
    ) -> MicrocompactPressure:
        if pressure is not None:
            return pressure
        if token_budget is None:
            return "normal"
        if count_message_tokens(messages, self.tokenizer_model) > token_budget:
            return "high"
        return "normal"

    def _build_groups(
        self,
        messages: list[Message],
        *,
        pressure: MicrocompactPressure,
    ) -> list[MessageGroup]:
        raw_groups: list[tuple[Message, ...]] = []
        index = 0
        while index < len(messages):
            current = messages[index]

            if self._is_protected_message(current):
                raw_groups.append((current,))
                index += 1
                continue

            next_message = messages[index + 1] if index + 1 < len(messages) else None
            if next_message is not None and self._should_pair(current, next_message):
                raw_groups.append((current, next_message))
                index += 2
                continue

            raw_groups.append((current,))
            index += 1

        groups: list[MessageGroup] = []
        total_groups = len(raw_groups)
        recent_tool_groups_seen = 0
        for reverse_index, group_messages in enumerate(reversed(raw_groups)):
            distance_from_end = reverse_index
            kind = self._classify_group(group_messages)
            recent_tool_index: int | None = None
            if kind == "tool_round":
                recent_tool_index = recent_tool_groups_seen
                recent_tool_groups_seen += 1
            groups.append(
                MessageGroup(
                    messages=group_messages,
                    kind=kind,
                    age_band=self._classify_age_band(
                        distance_from_end,
                        kind=kind,
                        recent_tool_index=recent_tool_index,
                        pressure=pressure,
                    ),
                    token_count=count_message_tokens(list(group_messages), self.tokenizer_model),
                )
            )
        groups.reverse()
        assert len(groups) == total_groups
        return groups

    def _should_pair(self, current: Message, next_message: Message) -> bool:
        if self._is_protected_message(next_message):
            return False
        current_role = current["role"]
        next_role = next_message["role"]

        # Pair tool calls with their results
        if has_tool_calls(current) and is_tool_result(next_message):
            return True
        # Pair tool results that follow each other (multi-tool case)
        if is_tool_result(current) and is_tool_result(next_message):
            return True

        if current_role == "assistant" and next_role == "user":
            return True
        if current_role == "user" and next_role == "assistant":
            if self._looks_like_tool_or_terminal_content(get_text_content(current)):
                return True
            return not self._looks_like_tool_or_terminal_content(get_text_content(next_message))
        return False

    def _classify_group(self, messages: tuple[Message, ...]) -> GroupKind:
        if any(self._is_protected_message(message) for message in messages):
            return "protected"
        # Tool call/result pairs are tool rounds
        if any(has_tool_calls(message) or is_tool_result(message) for message in messages):
            return "tool_round"
        if any(self._looks_like_tool_or_terminal_content(get_text_content(message)) for message in messages):
            return "tool_round"
        if len(messages) == 2:
            return "dialogue_round"
        return "single"

    def _classify_age_band(
        self,
        distance_from_end: int,
        *,
        kind: GroupKind,
        recent_tool_index: int | None,
        pressure: MicrocompactPressure,
    ) -> AgeBand:
        recent_groups_to_keep = (
            self.config.aggressive_recent_groups_to_keep
            if pressure == "high"
            else self.config.recent_groups_to_keep
        )
        if kind == "tool_round" and recent_tool_index is not None:
            if recent_tool_index < self.config.recent_tool_groups_to_keep:
                return "recent"
        elif distance_from_end < recent_groups_to_keep:
            return "recent"

        group_age = distance_from_end + 1
        if group_age >= self.config.mature_group_age:
            return "mature"
        return "old"

    def _should_compact_group(
        self,
        group: MessageGroup,
        *,
        pressure: MicrocompactPressure,
    ) -> bool:
        if group.kind == "protected":
            return False

        # Check salience score — preserve high-value content
        if self.config.use_salience_scoring:
            group_salience = self._compute_group_salience(group)
            # Under high pressure, still compact if salience is moderate
            salience_threshold = (
                self.config.min_salience_to_protect * 2
                if pressure == "high"
                else self.config.min_salience_to_protect
            )
            if group_salience >= salience_threshold:
                return False

        min_content_chars = (
            self.config.aggressive_min_content_chars
            if pressure == "high"
            else self.config.min_content_chars
        )
        total_chars = sum(len(get_text_content(message)) for message in group.messages)

        if group.kind == "tool_round":
            if group.age_band == "recent":
                return pressure == "high" and total_chars >= min_content_chars
            return total_chars >= max(80, min_content_chars // 2)

        if group.age_band == "mature":
            return total_chars >= min_content_chars or len(group.messages) > 1
        if group.age_band == "old":
            return total_chars >= min_content_chars
        return False

    def _compute_group_salience(self, group: MessageGroup) -> int:
        """Compute aggregate salience score for a message group.

        Higher scores indicate more important content that should be preserved.
        """
        total_salience = 0
        for i, message in enumerate(group.messages):
            content = get_text_content(message)
            # Use recency_rank based on position in group
            score = score_text_salience(content, recency_rank=len(group.messages) - i)
            total_salience = max(total_salience, score)
        return total_salience

    def _compact_group(self, group: MessageGroup) -> Message:
        commands = self._extract_commands(group.messages)
        exit_codes = self._extract_exit_codes(group.messages)
        errors = self._extract_errors(group.messages)
        artifacts = self._extract_artifacts(group.messages)
        images = self._extract_images(group.messages)
        tool_calls = self._extract_tool_call_info(group.messages)
        snippet = self._extract_text_snippet(group.messages)

        parts = [MICROCOMPACT_PREFIX, f"kind={group.kind}", f"age={group.age_band}"]
        if commands:
            parts.append("commands=" + " | ".join(commands[: self.config.max_commands]))
        if exit_codes:
            parts.append("exit_codes=" + " | ".join(exit_codes))
        if errors:
            parts.append("errors=" + " | ".join(errors[: self.config.max_errors]))
        if artifacts:
            parts.append("artifacts=" + " | ".join(artifacts[: self.config.max_artifacts]))
        if images:
            parts.append("images=" + " | ".join(images))
        if tool_calls:
            parts.append("tools=" + " | ".join(tool_calls))
        if snippet and (
            group.kind != "tool_round" or (not commands and not errors and not artifacts and not tool_calls)
        ):
            parts.append("snippet=" + snippet)

        return {"role": "system", "content": " ".join(parts)}

    def _extract_commands(self, messages: tuple[Message, ...]) -> list[str]:
        commands: list[str] = []
        for message in messages:
            content = get_text_content(message)
            commands.extend(match.group(1).strip() for match in COMMAND_RE.finditer(content))
        return self._dedupe(commands)

    def _extract_exit_codes(self, messages: tuple[Message, ...]) -> list[str]:
        codes: list[str] = []
        for message in messages:
            content = get_text_content(message)
            codes.extend(match.group(1) for match in EXIT_CODE_RE.finditer(content))
        return self._dedupe(codes)

    def _extract_errors(self, messages: tuple[Message, ...]) -> list[str]:
        errors: list[str] = []
        for message in messages:
            content = get_text_content(message)
            errors.extend(self._normalize_line(match.group(0)) for match in ERROR_RE.finditer(content))
        deduped = self._dedupe(errors)
        specific_errors = [
            error
            for error in deduped
            if "traceback" not in error.lower() and len(error) > 8
        ]
        if specific_errors:
            return specific_errors
        return deduped

    def _extract_artifacts(self, messages: tuple[Message, ...]) -> list[str]:
        artifacts: list[str] = []
        for message in messages:
            content = get_text_content(message)
            artifacts.extend(match.group(1).strip("()[]{}.,:;") for match in PATH_RE.finditer(content))
            artifacts.extend(match.group(1).strip("()[]{}.,:;") for match in BACKTICK_RE.finditer(content))
        return self._dedupe(artifacts)

    def _extract_images(self, messages: tuple[Message, ...]) -> list[str]:
        """Extract image references from multimodal messages."""
        images: list[str] = []
        for message in messages:
            images.extend(extract_image_refs(message))
        return self._dedupe(images)

    def _extract_tool_call_info(self, messages: tuple[Message, ...]) -> list[str]:
        """Extract tool call names and IDs from messages.

        Returns a list like: ["get_weather(id=call_123)", "search(id=call_456)"]
        """
        tool_info: list[str] = []
        for message in messages:
            # Extract from tool_calls array (assistant making calls)
            tool_calls = message.get("tool_calls", [])
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    func = tool_call.get("function", {})
                    name = func.get("name", "unknown")
                    call_id = tool_call.get("id", "")
                    if call_id:
                        tool_info.append(f"{name}(id={call_id})")
                    else:
                        tool_info.append(name)

            # Extract from tool result messages
            if is_tool_result(message):
                name = message.get("name", "")
                call_id = message.get("tool_call_id", "")
                if name:
                    if call_id:
                        tool_info.append(f"{name}→result(id={call_id})")
                    else:
                        tool_info.append(f"{name}→result")

        return self._dedupe(tool_info)

    def _extract_text_snippet(self, messages: tuple[Message, ...]) -> str:
        lines: list[str] = []
        for message in messages:
            content = get_text_content(message)
            for line in content.splitlines():
                raw_stripped = line.strip()
                if raw_stripped.startswith("[") and raw_stripped.endswith("]"):
                    continue
                normalized = self._normalize_line(line)
                if not normalized:
                    continue
                if normalized.startswith("$ "):
                    continue
                lines.append(normalized)
        if not lines:
            return ""
        text = " | ".join(lines[:3])
        if len(text) <= self.config.text_snippet_chars:
            return text
        return text[: self.config.text_snippet_chars - 3].rstrip() + "..."

    def _looks_like_tool_or_terminal_content(self, content: str) -> bool:
        lowered = content.lower()
        return any(
            marker in lowered
            for marker in (
                "[command]",
                "[terminal]",
                "[output]",
                "[exit_code]",
                "traceback",
                "stderr",
                "stdout",
                "$ ",
            )
        )

    def _is_protected_message(self, message: Message) -> bool:
        if message["role"] == "system":
            return True
        content = get_text_content(message)
        return content.startswith(MICROCOMPACT_PREFIX) or content.startswith(SUMMARY_SYSTEM_PREFIX)

    def _normalize_line(self, line: str) -> str:
        return WHITESPACE_RE.sub(" ", line).strip().strip("()[]{}.,:;")

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if not value:
                continue
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered


@dataclass(frozen=True, slots=True)
class MicrocompactMessages(Transform):
    config: MicrocompactConfig = MicrocompactConfig()
    tokenizer_model: str | None = None

    @property
    def name(self) -> str:
        return "microcompact"

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
    ) -> Transform:
        return replace(
            self,
            tokenizer_model=(
                self.tokenizer_model if self.tokenizer_model is not None else tokenizer_model
            ),
        )

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        orchestrator = MicrocompactOrchestrator(
            config=self.config,
            tokenizer_model=self.tokenizer_model,
        )
        return PipelineState(
            context=orchestrator.apply(
                state.context,
                token_budget=request.token_budget,
            ),
            tools=state.tools,
        )
