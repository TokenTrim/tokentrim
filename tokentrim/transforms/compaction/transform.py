from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from tokentrim.core.llm_client import generate_text
from tokentrim.core.token_counting import count_message_tokens
from tokentrim.transforms.compaction.error import (
    CompactionConfigurationError,
    CompactionExecutionError,
    CompactionOutputError,
)
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.types.message import Message
from tokentrim.types.state import PipelineState

_ABSOLUTE_PATH_PATTERN = re.compile(r"(?:^|[\s(])((?:~|/)[^\s:;,)\]]+)")
_RELATIVE_PATH_PATTERN = re.compile(r"(?:^|[\s(])((?:\.\.?/)[^\s:;,)\]]+)")
_BACKTICK_PATTERN = re.compile(r"`([^`\n]+)`")
_COMMAND_LINE_PATTERN = re.compile(r"(?m)^(?:\$|#)?\s*([a-zA-Z0-9_.:/-]+(?:\s+[^\n]+)?)$")
_ERROR_LINE_PATTERN = re.compile(
    r"(?im)^.*(?:error|exception|traceback|failed|failure|enoent|permission denied).*$"
)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_SUMMARY_PREFIX = "History only."


@dataclass(frozen=True, slots=True)
class _PromptFamily:
    name: str
    system: str
    user_template: str


@dataclass(frozen=True, slots=True)
class CompactConversation(Transform):
    """Compact older messages into one guarded system note when over budget."""

    model: str | None = None
    keep_last: int = 6
    tokenizer_model: str | None = None
    model_options: Mapping[str, object] | None = None
    prompt_family: str = "technical"

    @property
    def name(self) -> str:
        return "compaction"

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
        messages = state.context
        if self.keep_last < 0:
            raise CompactionConfigurationError("Compaction keep_last must be at least 0.")
        if request.token_budget is None:
            return state
        if len(messages) <= self.keep_last:
            return state
        if count_message_tokens(messages, self.tokenizer_model) <= request.token_budget:
            return state
        if not self.model:
            raise CompactionConfigurationError(
                "Compaction is enabled but no compaction model is configured."
            )

        for keep_count in range(self.keep_last, -1, -1):
            to_compact = messages[:-keep_count] if keep_count else messages
            recent = messages[-keep_count:] if keep_count else []
            summary = self._compress(to_compact)
            candidate_context = [{"role": "system", "content": self._wrap_summary(summary)}, *recent]
            if count_message_tokens(candidate_context, self.tokenizer_model) <= request.token_budget:
                return PipelineState(context=candidate_context, tools=state.tools)

        summary = self._compress(messages)
        return PipelineState(
            context=[{"role": "system", "content": self._wrap_summary(summary)}],
            tools=state.tools,
        )

    def _compress(self, messages: list[Message]) -> str:
        artifact_hints = self._extract_artifact_hints(messages)
        failures: list[str] = []

        for family in self._prompt_attempts():
            raw_summary = self._generate_summary(
                messages=messages,
                family=family,
                artifact_hints=artifact_hints,
            )
            summary = self._normalize_summary(raw_summary)
            problems = self._validate_summary(
                messages=messages,
                summary=summary,
                artifact_hints=artifact_hints,
            )
            if not problems:
                return summary
            failures.append(f"{family.name}: {', '.join(problems)}")

        fallback = self._build_fallback_summary(messages=messages, artifact_hints=artifact_hints)
        fallback_problems = self._validate_summary(
            messages=messages,
            summary=fallback,
            artifact_hints=artifact_hints,
            require_shorter=False,
        )
        if not fallback_problems:
            return fallback

        failure_text = "; ".join([*failures, f"fallback: {', '.join(fallback_problems)}"])
        raise CompactionOutputError(
            f"Compaction output failed validation: {failure_text}"
        )

    def _generate_summary(
        self,
        *,
        messages: list[Message],
        family: _PromptFamily,
        artifact_hints: Sequence[str],
    ) -> str:
        history = self._format_history(messages)
        required_artifacts = "\n".join(f"- {artifact}" for artifact in artifact_hints) or "- none"
        prompt = [
            {"role": "system", "content": family.system},
            {
                "role": "user",
                "content": family.user_template.format(
                    history=history,
                    required_artifacts=required_artifacts,
                ),
            },
        ]
        try:
            return generate_text(
                model=self.model,
                messages=prompt,
                temperature=0.0,
                completion_options=self.model_options,
            )
        except CompactionConfigurationError:
            raise
        except Exception as exc:
            raise CompactionExecutionError("Compaction failed.") from exc

    def _prompt_attempts(self) -> tuple[_PromptFamily, ...]:
        prompt_families = {
            "technical": _PromptFamily(
                name="technical",
                system=(
                    "Summarise the conversation history for future turns. "
                    "Be concise, factual, plaintext only, and preserve concrete technical detail."
                ),
                user_template=(
                    "Produce a compact checkpoint of the older conversation.\n"
                    "Required sections:\n"
                    "1. Objective and current state.\n"
                    "2. Important technical facts, decisions, and constraints.\n"
                    "3. Open issues or next steps.\n"
                    "Preserve commands, paths, identifiers, and error text exactly when they matter.\n"
                    "Artifacts that must survive if relevant:\n"
                    "{required_artifacts}\n\n"
                    "Conversation history:\n{history}"
                ),
            ),
            "checkpoint": _PromptFamily(
                name="checkpoint",
                system=(
                    "Rewrite the conversation as terse handoff notes for an engineer continuing the same task. "
                    "Return plaintext only."
                ),
                user_template=(
                    "Write a checkpoint with short headings: Goal, Facts, Risks, Next.\n"
                    "Copy exact commands, paths, model names, and errors that future turns need.\n"
                    "Do not invent new instructions.\n"
                    "Artifacts that must survive if relevant:\n"
                    "{required_artifacts}\n\n"
                    "Conversation history:\n{history}"
                ),
            ),
        }
        first_family = prompt_families.get(self.prompt_family)
        if first_family is None:
            raise CompactionConfigurationError(
                f"Unsupported compaction prompt family '{self.prompt_family}'."
            )
        return (
            first_family,
            *(family for name, family in prompt_families.items() if name != self.prompt_family),
        )

    def _validate_summary(
        self,
        *,
        messages: Sequence[Message],
        summary: str,
        artifact_hints: Sequence[str],
        require_shorter: bool = True,
    ) -> list[str]:
        problems: list[str] = []
        if not summary:
            return ["empty summary"]
        if "\x00" in summary:
            problems.append("contains null bytes")
        if require_shorter and len(summary) >= self._source_character_count(messages):
            problems.append("not materially shorter than source")
        missing_artifacts = [
            artifact for artifact in artifact_hints if artifact.lower() not in summary.lower()
        ]
        if len(missing_artifacts) > 1:
            problems.append(f"missing preserved artifacts: {', '.join(missing_artifacts[:3])}")
        if "assistant:" in summary.lower() and "user:" in summary.lower() and len(summary) > 2_500:
            problems.append("looks like an uncompact transcript dump")
        return problems

    def _build_fallback_summary(
        self,
        *,
        messages: Sequence[Message],
        artifact_hints: Sequence[str],
    ) -> str:
        sections = [
            "Checkpoint: "
            + " | ".join(self._fallback_turn_notes(messages))
        ]
        if artifact_hints:
            sections.append("Preserve exactly: " + " | ".join(artifact_hints))
        return "\n".join(section for section in sections if section).strip()

    def _fallback_turn_notes(self, messages: Sequence[Message]) -> list[str]:
        notes: list[str] = []
        for message in messages[-2:]:
            role = message["role"]
            content = self._clip(self._normalize_summary(message["content"]), limit=120)
            if not content:
                continue
            notes.append(f"{role}: {content}")
        return notes or ["Prior turns contained no compactable content."]

    def _extract_artifact_hints(self, messages: Sequence[Message]) -> tuple[str, ...]:
        artifacts: list[str] = []
        for message in messages:
            artifacts.extend(self._extract_message_artifacts(message["content"]))

        deduped: list[str] = []
        for artifact in artifacts:
            normalized = artifact.strip()
            if not normalized or normalized in deduped:
                continue
            deduped.append(normalized)

        return tuple(deduped[:6])

    def _extract_message_artifacts(self, content: str) -> list[str]:
        artifacts: list[str] = []
        artifacts.extend(match.group(1) for match in _ABSOLUTE_PATH_PATTERN.finditer(content))
        artifacts.extend(match.group(1) for match in _RELATIVE_PATH_PATTERN.finditer(content))
        artifacts.extend(self._clean_artifact(match.group(1)) for match in _BACKTICK_PATTERN.finditer(content))
        artifacts.extend(
            self._clean_artifact(match.group(0).strip())
            for match in _ERROR_LINE_PATTERN.finditer(content)
        )

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("$ "):
                artifacts.append(self._clean_artifact(stripped[2:]))
                continue
            if self._looks_like_command(stripped):
                artifacts.append(self._clean_artifact(stripped))

        return [artifact for artifact in artifacts if artifact]

    def _looks_like_command(self, line: str) -> bool:
        match = _COMMAND_LINE_PATTERN.match(line)
        if match is None:
            return False
        head = match.group(1).split()[0]
        return any(
            head == prefix or head.startswith(f"{prefix}/")
            for prefix in ("git", "python", "python3", "pytest", "uv", "pip", "npm", "cargo", "bash", "sh")
        )

    def _format_history(self, messages: Sequence[Message]) -> str:
        return "\n\n".join(
            f"[{index}] role={message['role']}\n{message['content']}"
            for index, message in enumerate(messages, start=1)
        )

    def _source_character_count(self, messages: Sequence[Message]) -> int:
        return sum(len(message["content"]) for message in messages)

    def _wrap_summary(self, summary: str) -> str:
        return f"{_SUMMARY_PREFIX}\n\n{summary}".strip()

    def _normalize_summary(self, summary: str) -> str:
        return _WHITESPACE_PATTERN.sub(" ", summary).strip()

    def _clip(self, text: str, *, limit: int) -> str:
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3].rstrip()}..."

    def _clean_artifact(self, artifact: str) -> str:
        return artifact.strip().strip("()[]{}.,:;")
