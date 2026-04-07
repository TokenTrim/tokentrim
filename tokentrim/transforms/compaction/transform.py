"""Compaction transform for summarizing conversation history.

This module provides the CompactConversation transform which compresses older
messages into a concise summary when the context exceeds a token budget.
The summary is injected as a system message, preserving recent messages.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from tokentrim.core.llm_client import generate_text
from tokentrim.core.token_counting import count_message_tokens
from tokentrim.working_state import extract_working_state, render_working_state_message
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.transforms.compaction.microcompact import (
    MicrocompactConfig,
    MicrocompactOrchestrator,
)
from tokentrim.transforms.compaction.error import (
    CompactionConfigurationError,
    CompactionExecutionError,
    CompactionOutputError,
)
from tokentrim.transforms.compaction.context_edit import ContextEditor
from tokentrim.types.message import Message
from tokentrim.types.state import PipelineState

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants
# =============================================================================

# Default number of recent messages to preserve without summarization.
DEFAULT_KEEP_LAST: Final[int] = 6

# Default amount of room to reserve for the model response.
DEFAULT_RESERVED_OUTPUT_TOKENS: Final[int] = 8_000

# Start compacting before the context window is actually full.
DEFAULT_AUTO_COMPACT_BUFFER_TOKENS: Final[int] = 4_000

# Maximum number of artifacts (paths, commands, errors) to preserve in summary.
MAX_PRESERVED_ARTIFACTS: Final[int] = 6

# Character limit for truncated content in fallback summaries.
FALLBACK_TRUNCATION_LIMIT: Final[int] = 120

# Summaries exceeding this length with transcript patterns are flagged as uncompacted.
UNCOMPACTED_TRANSCRIPT_THRESHOLD: Final[int] = 2_500

# Prefix for summary system messages, marking them as historical context.
SUMMARY_SYSTEM_PREFIX: Final[str] = "History only."

SUMMARY_SECTIONS: Final[tuple[str, ...]] = (
    "Goal",
    "Active State",
    "Critical Artifacts",
    "Open Risks",
    "Next Step",
    "Older Context",
)

_SUMMARY_SECTION_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "Goal": ("Goal",),
    "Active State": ("Active State", "Current State"),
    "Critical Artifacts": (
        "Critical Artifacts",
        "Important Facts",
        "Facts",
        "Commands / Paths / Identifiers",
    ),
    "Open Risks": ("Open Risks", "Errors / Risks", "Risks", "Risk"),
    "Next Step": ("Next Step", "Next Steps"),
    "Older Context": ("Older Context",),
}


# =============================================================================
# Artifact Extraction Patterns
# =============================================================================

# Matches absolute paths: /usr/bin/python, ~/Documents/file.txt
_ABSOLUTE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[\s(])((?:~|/)[^\s:;,)\]]+)"
)

# Matches relative paths: ./src/main.py, ../config.yaml
_RELATIVE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[\s(])((?:\.\.?/)[^\s:;,)\]]+)"
)

# Matches inline code spans: `variable_name`, `some_command`
_BACKTICK_CODE_RE: Final[re.Pattern[str]] = re.compile(r"`([^`\n]+)`")

# Matches shell command line prefixes: $ git status, # pip install
_SHELL_COMMAND_RE: Final[re.Pattern[str]] = re.compile(
    r"(?m)^(?:\$|#)?\s*([a-zA-Z0-9_.:/-]+(?:\s+[^\n]+)?)$"
)

# Matches error/exception lines (case-insensitive)
_ERROR_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^.*(?:error|exception|traceback|failed|failure|enoent|permission denied).*$"
)

# Whitespace normalization pattern
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

# Known command prefixes for shell command detection
_KNOWN_COMMAND_PREFIXES: Final[frozenset[str]] = frozenset(
    ("git", "python", "python3", "pytest", "uv", "pip", "npm", "cargo", "bash", "sh")
)

_MODEL_CONTEXT_WINDOW_PATTERNS: Final[tuple[tuple[str, int], ...]] = (
    ("gpt-4.1", 1_000_000),
    ("o4-mini", 200_000),
    ("o3", 200_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 128_000),
    ("claude", 200_000),
    ("mercury-2", 200_000),
)


# =============================================================================
# Prompt Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Configuration for a summarization prompt style.

    Attributes:
        name: Identifier for this prompt style (e.g., "technical", "checkpoint").
        system_prompt: System message instructing the LLM how to summarize.
        user_template: User message template with {history} and {required_artifacts}.
    """

    name: str
    system_prompt: str
    user_template: str


# Available summarization prompt styles, keyed by name
_PROMPT_TEMPLATES: Final[dict[str, PromptTemplate]] = {
    "structured": PromptTemplate(
        name="structured",
        system_prompt=(
            "Summarise the conversation into a compact engineering handoff. "
            "Return plaintext only. Keep exact technical details when they matter."
        ),
        user_template=(
            "Write a concise handoff using these exact headings:\n"
            "Goal:\n"
            "Active State:\n"
            "Critical Artifacts:\n"
            "Open Risks:\n"
            "Next Step:\n"
            "Older Context:\n"
            "Preserve commands, file paths, identifiers, and error strings exactly when relevant.\n"
            "Artifacts that must survive if relevant:\n"
            "{required_artifacts}\n\n"
            "Conversation history:\n{history}"
        ),
    ),
    "technical": PromptTemplate(
        name="technical",
        system_prompt=(
            "Summarise the conversation history for future turns. "
            "Be concise, factual, plaintext only, and preserve concrete technical detail."
        ),
        user_template=(
            "Produce a compact checkpoint of the older conversation.\n"
            "Use these exact headings:\n"
            "Goal:\n"
            "Active State:\n"
            "Critical Artifacts:\n"
            "Open Risks:\n"
            "Next Step:\n"
            "Older Context:\n"
            "Preserve commands, paths, identifiers, and error text exactly when they matter.\n"
            "Artifacts that must survive if relevant:\n"
            "{required_artifacts}\n\n"
            "Conversation history:\n{history}"
        ),
    ),
    "checkpoint": PromptTemplate(
        name="checkpoint",
        system_prompt=(
            "Rewrite the conversation as terse handoff notes for an engineer "
            "continuing the same task. Return plaintext only."
        ),
        user_template=(
            "Write a checkpoint with these exact headings:\n"
            "Goal:\n"
            "Active State:\n"
            "Critical Artifacts:\n"
            "Open Risks:\n"
            "Next Step:\n"
            "Older Context:\n"
            "Copy exact commands, paths, model names, and errors that future turns need.\n"
            "Do not invent new instructions.\n"
            "Artifacts that must survive if relevant:\n"
            "{required_artifacts}\n\n"
            "Conversation history:\n{history}"
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class CompactConversation(Transform):
    """Compacts older messages into a summary when context exceeds token budget.

    This transform monitors the conversation token count and, when it exceeds
    the budget, summarizes older messages while preserving recent ones. The
    summary is injected as a system message at the start of the context.

    The compaction process:
    1. Tries to keep `keep_last` recent messages, summarizing older ones
    2. If still over budget, progressively reduces preserved messages
    3. Validates summaries preserve important artifacts (paths, commands, errors)
    4. Falls back to local extraction if LLM summaries fail validation

    Attributes:
        model: LLM model identifier for generating summaries.
        keep_last: Target number of recent messages to preserve (minimum 0).
        tokenizer_model: Model for token counting (falls back to pipeline default).
        model_options: Additional options for LLM client (e.g., api_base, api_key).
        prompt_family: Summarization style - "technical" or "checkpoint".

    Example:
        >>> transform = CompactConversation(
        ...     model="gpt-4o-mini",
        ...     keep_last=4,
        ...     prompt_family="checkpoint",
        ... )
    """

    model: str | None = None
    keep_last: int = DEFAULT_KEEP_LAST
    tokenizer_model: str | None = None
    model_options: Mapping[str, object] | None = None
    prompt_family: str = "structured"
    auto_budget: bool = True
    context_window: int | None = None
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS
    auto_compact_buffer_tokens: int = DEFAULT_AUTO_COMPACT_BUFFER_TOKENS
    enable_microcompact: bool = True
    microcompact_config: MicrocompactConfig = MicrocompactConfig()

    @property
    def name(self) -> str:
        """Stable identifier for this transform, used in tracing and selection."""
        return "compaction"

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
    ) -> Transform:
        """Resolve configuration defaults from pipeline settings."""
        return replace(
            self,
            tokenizer_model=(
                self.tokenizer_model if self.tokenizer_model is not None else tokenizer_model
            ),
        )

    def resolve_token_budget(
        self,
        token_budget: int | None,
    ) -> int | None:
        return self._resolve_effective_token_budget(token_budget)

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        """Execute compaction on the pipeline state.

        Args:
            state: Current pipeline state with context messages and tools.
            request: Pipeline request containing token budget.

        Returns:
            Updated state, potentially with compacted context.

        Raises:
            CompactionConfigurationError: If keep_last < 0 or model missing.
            CompactionOutputError: If all summarization attempts fail validation.
        """
        messages = state.context

        if self.keep_last < 0:
            raise CompactionConfigurationError(
                f"keep_last must be >= 0, got {self.keep_last}"
            )

        effective_budget = self._resolve_effective_token_budget(request.token_budget)

        if not self._requires_compaction(messages, effective_budget):
            return state

        if not self.model:
            raise CompactionConfigurationError(
                "Compaction is enabled but no compaction model is configured."
            )

        return self._compact_with_budget_fit(messages, effective_budget, state.tools)

    # -------------------------------------------------------------------------
    # Compaction Strategy
    # -------------------------------------------------------------------------

    def _requires_compaction(
        self,
        messages: Sequence[Message],
        token_budget: int | None,
    ) -> bool:
        """Check if compaction is needed based on budget and message count."""
        if token_budget is None:
            return False
        if len(messages) <= self.keep_last:
            return False
        current_tokens = count_message_tokens(messages, self.tokenizer_model)
        return current_tokens > token_budget

    def _resolve_effective_token_budget(self, explicit_budget: int | None) -> int | None:
        if explicit_budget is not None:
            return explicit_budget
        if not self.auto_budget:
            return None

        effective_context_window = self._resolve_effective_context_window()
        if effective_context_window is None:
            return None
        return max(1, effective_context_window - self.auto_compact_buffer_tokens)

    def _resolve_effective_context_window(self) -> int | None:
        context_window = self.context_window
        if context_window is None:
            context_window = self._infer_context_window_from_model()
        if context_window is None:
            return None
        reserved_output_tokens = min(self.reserved_output_tokens, context_window - 1)
        return max(1, context_window - reserved_output_tokens)

    def _infer_context_window_from_model(self) -> int | None:
        candidates = [self.model, self.tokenizer_model]
        for candidate in candidates:
            if not candidate:
                continue
            normalized = candidate.lower()
            for pattern, context_window in _MODEL_CONTEXT_WINDOW_PATTERNS:
                if pattern in normalized:
                    return context_window
        return None

    def _compact_with_budget_fit(
        self,
        messages: Sequence[Message],
        token_budget: int,
        tools: Sequence[object],
    ) -> PipelineState:
        """Try progressively aggressive compaction until context fits budget.

        Starts by preserving keep_last messages, then reduces if needed.
        """
        working_state_message = self._build_working_state_message(messages)

        for preserved_count in range(self.keep_last, -1, -1):
            messages_to_summarize = messages[:-preserved_count] if preserved_count else messages
            preserved_messages = messages[-preserved_count:] if preserved_count else []

            logger.debug(
                "Attempting compaction: summarizing %d messages, preserving %d",
                len(messages_to_summarize),
                len(preserved_messages),
            )

            summary = self._generate_validated_summary(messages_to_summarize)
            summary_message = self._create_summary_message(summary)
            candidate_context = self._build_candidate_context(
                working_state_message,
                summary_message,
                preserved_messages,
            )

            if count_message_tokens(candidate_context, self.tokenizer_model) <= token_budget:
                logger.info(
                    "Compaction successful: %d messages → summary + %d preserved",
                    len(messages),
                    len(preserved_messages),
                )
                return PipelineState(context=candidate_context, tools=tools)

        # Final attempt: summarize everything
        summary = self._generate_validated_summary(messages)
        return PipelineState(
            context=self._build_candidate_context(
                working_state_message,
                self._create_summary_message(summary),
                (),
            ),
            tools=tools,
        )

    def _build_working_state_message(self, messages: Sequence[Message]) -> Message | None:
        return render_working_state_message(extract_working_state(messages))

    def _build_candidate_context(
        self,
        working_state_message: Message | None,
        summary_message: Message,
        preserved_messages: Sequence[Message],
    ) -> list[Message]:
        candidate_context: list[Message] = []
        if working_state_message is not None:
            candidate_context.append(working_state_message)
        candidate_context.append(summary_message)
        candidate_context.extend(preserved_messages)
        return candidate_context

    # -------------------------------------------------------------------------
    # Summary Generation & Validation
    # -------------------------------------------------------------------------

    def _generate_validated_summary(self, messages: Sequence[Message]) -> str:
        """Generate and validate a summary, with fallback on failure.

        Tries each prompt template in order, falling back to local extraction
        if all LLM-generated summaries fail validation.

        Raises:
            CompactionOutputError: If all attempts (including fallback) fail.
        """
        edited_messages = self._apply_context_edit(messages)
        microcompacted_messages = self._apply_microcompact(edited_messages, pressure="high")
        preserved_artifacts = self._extract_preserved_artifacts(microcompacted_messages)
        validation_failures: list[str] = []

        # Try each prompt template
        for template in self._get_ordered_templates():
            raw_summary = self._call_llm_for_summary(
                microcompacted_messages,
                template,
                preserved_artifacts,
            )
            summary = self._normalize_summary_output(raw_summary, preserved_artifacts)

            problems = self._validate_summary(
                microcompacted_messages,
                summary,
                preserved_artifacts,
            )
            if not problems:
                return summary

            validation_failures.append(f"{template.name}: {', '.join(problems)}")
            logger.debug("Summary from '%s' failed validation: %s", template.name, problems)

        # Try local fallback extraction
        fallback = self._build_fallback_summary(microcompacted_messages, preserved_artifacts)
        fallback_problems = self._validate_summary(
            microcompacted_messages,
            fallback,
            preserved_artifacts,
            require_shorter=False,
        )
        if not fallback_problems:
            logger.info("Using fallback summary after LLM validation failures")
            return fallback

        validation_failures.append(f"fallback: {', '.join(fallback_problems)}")
        failure_details = "; ".join(validation_failures)
        raise CompactionOutputError(f"Compaction output failed validation: {failure_details}")

    def _call_llm_for_summary(
        self,
        messages: Sequence[Message],
        template: PromptTemplate,
        preserved_artifacts: Sequence[str],
    ) -> str:
        """Call the LLM to generate a summary using the given template."""
        formatted_history = self._format_messages_as_history(messages)
        artifacts_text = (
            "\n".join(f"- {artifact}" for artifact in preserved_artifacts)
            or "- none"
        )

        prompt: list[Message] = [
            {"role": "system", "content": template.system_prompt},
            {
                "role": "user",
                "content": template.user_template.format(
                    history=formatted_history,
                    required_artifacts=artifacts_text,
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
            logger.exception("LLM call failed during compaction")
            raise CompactionExecutionError(
                f"Compaction failed while summarizing {len(messages)} messages"
            ) from exc

    def _get_ordered_templates(self) -> tuple[PromptTemplate, ...]:
        """Get prompt templates ordered by preference (configured family first)."""
        primary = _PROMPT_TEMPLATES.get(self.prompt_family)
        if primary is None:
            raise CompactionConfigurationError(
                f"Unknown prompt_family '{self.prompt_family}'. "
                f"Available: {', '.join(_PROMPT_TEMPLATES.keys())}"
            )
        return (
            primary,
            *(t for name, t in _PROMPT_TEMPLATES.items() if name != self.prompt_family),
        )

    def _validate_summary(
        self,
        messages: Sequence[Message],
        summary: str,
        preserved_artifacts: Sequence[str],
        *,
        require_shorter: bool = True,
    ) -> list[str]:
        """Validate a summary meets quality requirements.

        Returns:
            List of validation problems (empty if valid).
        """
        problems: list[str] = []

        if not summary:
            return ["empty summary"]

        if "\x00" in summary:
            problems.append("contains null bytes")

        source_length = self._total_content_length(messages)
        if require_shorter and len(summary) >= source_length:
            problems.append("not materially shorter than source")

        missing_sections = [
            heading for heading in SUMMARY_SECTIONS if f"{heading}:" not in summary
        ]
        if missing_sections:
            problems.append(f"missing sections: {', '.join(missing_sections)}")

        # Check that most artifacts are preserved (allow 1 missing)
        missing = [a for a in preserved_artifacts if a.lower() not in summary.lower()]
        if len(missing) > 1:
            problems.append(f"missing artifacts: {', '.join(missing[:3])}")

        # Detect uncompacted transcript dumps
        if self._looks_like_transcript_dump(summary):
            problems.append("appears to be uncompacted transcript")

        return problems

    def _looks_like_transcript_dump(self, summary: str) -> bool:
        """Detect if summary is just a copy of the conversation."""
        has_role_markers = "assistant:" in summary.lower() and "user:" in summary.lower()
        return has_role_markers and len(summary) > UNCOMPACTED_TRANSCRIPT_THRESHOLD

    def _build_fallback_summary(
        self,
        messages: Sequence[Message],
        preserved_artifacts: Sequence[str],
    ) -> str:
        """Build a simple summary from recent messages when LLM fails."""
        turn_notes = self._extract_recent_turn_notes(messages)
        return self._render_summary_sections(
            {
                "Goal": self._extract_fallback_goal(messages),
                "Active State": " | ".join(turn_notes),
                "Critical Artifacts": " | ".join(preserved_artifacts) if preserved_artifacts else "- none",
                "Open Risks": self._extract_fallback_risks(messages),
                "Next Step": self._extract_fallback_next_step(messages),
                "Older Context": "- compacted from earlier turns",
            }
        )

    def _extract_recent_turn_notes(self, messages: Sequence[Message]) -> list[str]:
        """Extract brief notes from the last 2 messages for fallback summary."""
        notes: list[str] = []
        for message in messages[-2:]:
            role = message["role"]
            content = self._truncate_with_ellipsis(
                self._normalize_whitespace(message["content"]),
                limit=FALLBACK_TRUNCATION_LIMIT,
            )
            if content:
                notes.append(f"{role}: {content}")
        return notes or ["Prior turns contained no compactable content."]

    def _extract_fallback_goal(self, messages: Sequence[Message]) -> str:
        for message in messages:
            if message["role"] != "user":
                continue
            content = self._truncate_with_ellipsis(
                self._normalize_whitespace(message["content"]),
                limit=FALLBACK_TRUNCATION_LIMIT,
            )
            if content:
                return content
        return "- not captured"

    def _extract_fallback_risks(self, messages: Sequence[Message]) -> str:
        risks = [
            self._strip_punctuation(match.group(0).strip())
            for message in messages
            for match in _ERROR_LINE_RE.finditer(message["content"])
        ]
        return " | ".join(risks[:2]) if risks else "- none"

    def _extract_fallback_next_step(self, messages: Sequence[Message]) -> str:
        for message in reversed(messages):
            if message["role"] != "assistant":
                continue
            content = self._truncate_with_ellipsis(
                self._normalize_whitespace(message["content"]),
                limit=FALLBACK_TRUNCATION_LIMIT,
            )
            if content:
                return content
        return "- review preserved recent messages"

    # -------------------------------------------------------------------------
    # Deterministic Microcompact
    # -------------------------------------------------------------------------

    def _apply_microcompact(
        self,
        messages: Sequence[Message],
        *,
        pressure: str | None = None,
    ) -> list[Message]:
        """Deterministically compress older messages before LLM summarization."""
        if not self.enable_microcompact:
            return list(messages)
        orchestrator = MicrocompactOrchestrator(
            config=self.microcompact_config,
            tokenizer_model=self.tokenizer_model,
        )
        return orchestrator.apply(list(messages), pressure=pressure)

    def _apply_context_edit(self, messages: Sequence[Message]) -> list[Message]:
        editor = ContextEditor()
        return editor.apply(messages)

    # -------------------------------------------------------------------------
    # Artifact Extraction
    # -------------------------------------------------------------------------

    def _extract_preserved_artifacts(
        self,
        messages: Sequence[Message],
    ) -> tuple[str, ...]:
        """Extract important artifacts (paths, commands, errors) from messages.

        These are passed to the LLM as hints for what to preserve in the summary.
        """
        artifacts: list[str] = []
        for message in messages:
            artifacts.extend(self._extract_artifacts_from_content(message["content"]))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for artifact in artifacts:
            normalized = artifact.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)

        return tuple(unique[:MAX_PRESERVED_ARTIFACTS])

    def _extract_artifacts_from_content(self, content: str) -> list[str]:
        """Extract artifact strings from message content."""
        artifacts: list[str] = []

        # File paths
        artifacts.extend(m.group(1) for m in _ABSOLUTE_PATH_RE.finditer(content))
        artifacts.extend(m.group(1) for m in _RELATIVE_PATH_RE.finditer(content))

        # Inline code
        artifacts.extend(
            self._strip_punctuation(m.group(1))
            for m in _BACKTICK_CODE_RE.finditer(content)
        )

        # Error lines
        artifacts.extend(
            self._strip_punctuation(m.group(0).strip())
            for m in _ERROR_LINE_RE.finditer(content)
        )

        # Shell commands
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("$ "):
                artifacts.append(self._strip_punctuation(stripped[2:]))
            elif self._is_shell_command(stripped):
                artifacts.append(self._strip_punctuation(stripped))

        return [a for a in artifacts if a]

    def _is_shell_command(self, line: str) -> bool:
        """Check if a line looks like a shell command."""
        match = _SHELL_COMMAND_RE.match(line)
        if not match:
            return False
        first_word = match.group(1).split()[0]
        return any(
            first_word == prefix or first_word.startswith(f"{prefix}/")
            for prefix in _KNOWN_COMMAND_PREFIXES
        )

    # -------------------------------------------------------------------------
    # Formatting Helpers
    # -------------------------------------------------------------------------

    def _format_messages_as_history(self, messages: Sequence[Message]) -> str:
        """Format messages as numbered history for summarization prompt."""
        return "\n\n".join(
            f"[{i}] role={msg['role']}\n{msg['content']}"
            for i, msg in enumerate(messages, start=1)
        )

    def _normalize_summary_output(
        self,
        raw_summary: str,
        preserved_artifacts: Sequence[str],
    ) -> str:
        normalized = raw_summary.replace("\r\n", "\n").strip()
        parsed = self._parse_summary_sections(normalized)
        if parsed:
            parsed.setdefault(
                "Critical Artifacts",
                " | ".join(preserved_artifacts) if preserved_artifacts else "- none",
            )
            parsed.setdefault("Open Risks", "- none")
            parsed.setdefault("Next Step", "- not captured")
            parsed.setdefault("Older Context", "- none")
            parsed.setdefault("Active State", "- none")
            parsed.setdefault("Goal", "- not captured")
            return self._render_summary_sections(parsed)

        return self._render_summary_sections(
            {
                "Goal": "- not captured",
                "Active State": "- not captured",
                "Critical Artifacts": " | ".join(preserved_artifacts) if preserved_artifacts else "- none",
                "Open Risks": "- none",
                "Next Step": "- not captured",
                "Older Context": self._normalize_whitespace(normalized) or "- none",
            }
        )

    def _parse_summary_sections(self, summary: str) -> dict[str, str]:
        alias_to_canonical = {
            alias.lower(): canonical
            for canonical, aliases in _SUMMARY_SECTION_ALIASES.items()
            for alias in aliases
        }
        parsed: dict[str, str] = {}
        current_heading: str | None = None
        buffer: list[str] = []

        for raw_line in summary.splitlines():
            line = raw_line.strip()
            if line.endswith(":"):
                canonical = alias_to_canonical.get(line[:-1].strip().lower())
                if canonical is not None:
                    if current_heading is not None:
                        parsed[current_heading] = self._normalize_section_body(buffer)
                    current_heading = canonical
                    buffer = []
                    continue
            if current_heading is not None:
                buffer.append(raw_line)

        if current_heading is not None:
            parsed[current_heading] = self._normalize_section_body(buffer)

        return parsed

    def _normalize_section_body(self, lines: Sequence[str]) -> str:
        cleaned = [line.strip() for line in lines if line.strip()]
        if not cleaned:
            return "- none"
        return "\n".join(cleaned)

    def _render_summary_sections(self, sections: Mapping[str, str]) -> str:
        rendered: list[str] = []
        for heading in SUMMARY_SECTIONS:
            body = sections.get(heading, "- none").strip() or "- none"
            rendered.append(f"{heading}:\n{body}")
        return "\n\n".join(rendered)

    def _create_summary_message(self, summary: str) -> Message:
        """Wrap summary in a system message with the history prefix."""
        return {
            "role": "system",
            "content": f"{SUMMARY_SYSTEM_PREFIX}\n\n{summary}".strip(),
        }

    def _total_content_length(self, messages: Sequence[Message]) -> int:
        """Calculate total character count of message contents."""
        return sum(len(msg["content"]) for msg in messages)

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse consecutive whitespace and trim."""
        return _WHITESPACE_RE.sub(" ", text).strip()

    def _truncate_with_ellipsis(self, text: str, *, limit: int) -> str:
        """Truncate text with ellipsis if it exceeds the limit."""
        if len(text) <= limit:
            return text
        return f"{text[:limit - 3].rstrip()}..."

    def _strip_punctuation(self, text: str) -> str:
        """Remove surrounding punctuation from artifact strings."""
        return text.strip().strip("()[]{}.,:;")
