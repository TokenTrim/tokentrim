"""Compaction transform for summarizing conversation history.

This module provides the CompactConversation transform which compresses older
messages into a concise summary when the context exceeds a token budget.
The summary is injected as a system message, preserving recent messages.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from tokentrim.core.llm_client import generate_text
from tokentrim.core.token_counting import count_message_tokens
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.transforms.compaction.config import (
    get_context_edit_config,
    get_microcompact_config,
    DEFAULT_AUTO_COMPACT_BUFFER_TOKENS,
    DEFAULT_KEEP_LAST,
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    MAX_PRESERVED_ARTIFACTS,
    MODEL_CONTEXT_WINDOW_PATTERNS,
    SUMMARY_SYSTEM_PREFIX,
)
from tokentrim.transforms.compaction.context_edit import ContextEditor
from tokentrim.transforms.compaction.error import (
    CompactionConfigurationError,
    CompactionExecutionError,
)
from tokentrim.transforms.compaction.microcompact import MicrocompactOrchestrator
from tokentrim.transforms.compaction.patterns import (
    ABSOLUTE_PATH_RE,
    BACKTICK_RE,
    ERROR_RE,
    KNOWN_COMMAND_PREFIXES,
    RELATIVE_PATH_RE,
    SHELL_COMMAND_RE,
)
from tokentrim.transforms.compaction.prompts import PromptTemplate, build_prompt_template
from tokentrim.transforms.compaction.types import CompactionStrategy
from tokentrim.types.message import Message, get_text_content
from tokentrim.types.state import PipelineState
from tokentrim.working_state import extract_working_state, render_working_state_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompactionLLM:
    """Thin boundary around the model call used for compaction."""

    model: str
    model_options: Mapping[str, object] | None = None

    def generate(
        self,
        *,
        messages: Sequence[Message],
        template: PromptTemplate,
        preserved_artifacts: Sequence[str],
    ) -> str:
        formatted_history = "\n\n".join(
            f"[{i}] role={msg['role']}\n{get_text_content(msg)}"
            for i, msg in enumerate(messages, start=1)
        )
        artifacts_text = "\n".join(f"- {artifact}" for artifact in preserved_artifacts) or "- none"
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
        return generate_text(
            model=self.model,
            messages=prompt,
            temperature=0.0,
            completion_options=self.model_options,
        )

@dataclass(frozen=True, slots=True)
class CompactConversation(Transform):
    """Compacts older messages into a summary when context exceeds token budget.

    This transform monitors the conversation token count and, when it exceeds
    the budget, summarizes older messages while preserving recent ones. The
    summary is injected as a system message at the start of the context.

    The compaction process:
    1. Tries to keep `keep_last` recent messages, summarizing older ones
    2. If still over budget, progressively reduces preserved messages
    3. Applies deterministic context editing and microcompaction
    4. Calls the compaction model once and returns its summary text

    Attributes:
        model: LLM model identifier for generating summaries.
        keep_last: Target number of recent messages to preserve (minimum 0).
        strategy: Compression policy preset for deterministic pruning/compression.
        model_options: Additional options for LLM client (e.g., api_base, api_key).
        instructions: Optional custom compaction instructions for the summary prompt.

    Example:
        >>> transform = CompactConversation(
        ...     model="gpt-4o-mini",
        ...     keep_last=4,
        ...     strategy="balanced",
        ...     instructions="Preserve exact commands, paths, and unresolved errors.",
        ... )
    """

    model: str | None = None
    keep_last: int = DEFAULT_KEEP_LAST
    strategy: CompactionStrategy = "balanced"
    model_options: Mapping[str, object] | None = None
    instructions: str | None = None
    auto_budget: bool = True
    context_window: int | None = None
    _tokenizer_model: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )

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
            _tokenizer_model=tokenizer_model,
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
            CompactionConfigurationError: If keep_last < 0 or model missing.
        """
        messages = state.context

        if self.keep_last < 0:
            raise CompactionConfigurationError(
                f"keep_last must be >= 0, got {self.keep_last}"
            )
        if self.strategy not in ("aggressive", "balanced", "minimal"):
            raise CompactionConfigurationError(
                f"strategy must be one of aggressive, balanced, minimal; got {self.strategy!r}"
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
        current_tokens = count_message_tokens(messages, self._tokenizer_model)
        return current_tokens > token_budget

    def _resolve_effective_token_budget(self, explicit_budget: int | None) -> int | None:
        if explicit_budget is not None:
            return explicit_budget
        if not self.auto_budget:
            return None

        effective_context_window = self._resolve_effective_context_window()
        if effective_context_window is None:
            return None
        auto_compact_buffer_tokens = min(
            DEFAULT_AUTO_COMPACT_BUFFER_TOKENS,
            max(1, effective_context_window // 4),
        )
        return max(1, effective_context_window - auto_compact_buffer_tokens)

    def _resolve_effective_context_window(self) -> int | None:
        context_window = self.context_window
        if context_window is None:
            context_window = self._infer_context_window_from_model()
        if context_window is None:
            return None
        reserved_output_tokens = min(
            DEFAULT_RESERVED_OUTPUT_TOKENS,
            max(1, context_window // 4),
        )
        return max(1, context_window - reserved_output_tokens)

    def _infer_context_window_from_model(self) -> int | None:
        candidates = [self.model, self._tokenizer_model]
        for candidate in candidates:
            if not candidate:
                continue
            normalized = candidate.lower()
            for pattern, context_window in MODEL_CONTEXT_WINDOW_PATTERNS:
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

            summary = self._generate_summary(messages_to_summarize)
            summary_message = self._create_summary_message(summary)
            candidate_context = self._build_candidate_context(
                working_state_message,
                summary_message,
                preserved_messages,
            )

            if count_message_tokens(candidate_context, self._tokenizer_model) <= token_budget:
                tokens_after = count_message_tokens(candidate_context, self._tokenizer_model)
                tokens_saved = count_message_tokens(list(messages), self._tokenizer_model) - tokens_after
                compression_ratio = tokens_saved / max(1, count_message_tokens(list(messages), self._tokenizer_model))
                logger.info(
                    "Compaction successful: %d messages → summary + %d preserved "
                    "(tokens: %d → %d, saved: %d, ratio: %.1f%%)",
                    len(messages),
                    len(preserved_messages),
                    count_message_tokens(list(messages), self._tokenizer_model),
                    tokens_after,
                    tokens_saved,
                    compression_ratio * 100,
                )
                return PipelineState(context=candidate_context, tools=tools)

        # Final attempt: summarize everything
        summary = self._generate_summary(messages)
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
    # Summary Generation
    # -------------------------------------------------------------------------

    def _generate_summary(self, messages: Sequence[Message]) -> str:
        """Generate a summary after deterministic preprocessing."""
        edited_messages = self._apply_context_edit(messages)
        microcompacted_messages = self._apply_microcompact(edited_messages, pressure="high")
        preserved_artifacts = self._extract_preserved_artifacts(microcompacted_messages)

        template = self._build_prompt_template()
        try:
            raw_summary = self._compaction_llm().generate(
                messages=microcompacted_messages,
                template=template,
                preserved_artifacts=preserved_artifacts,
            )
        except CompactionConfigurationError:
            raise
        except Exception as exc:
            logger.exception("LLM call failed during compaction")
            raise CompactionExecutionError(
                f"Compaction failed while summarizing {len(messages)} messages"
            ) from exc
        return self._finalize_summary_output(raw_summary)

    def _compaction_llm(self) -> CompactionLLM:
        if not self.model:
            raise CompactionConfigurationError(
                "Compaction is enabled but no compaction model is configured."
            )
        return CompactionLLM(
            model=self.model,
            model_options=self.model_options,
        )

    def _build_prompt_template(self) -> PromptTemplate:
        return build_prompt_template(self.instructions)

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
        orchestrator = MicrocompactOrchestrator(
            config=get_microcompact_config(self.strategy),
            tokenizer_model=self._tokenizer_model,
        )
        return orchestrator.apply(list(messages), pressure=pressure)

    def _apply_context_edit(self, messages: Sequence[Message]) -> list[Message]:
        editor = ContextEditor(config=get_context_edit_config(self.strategy))
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
            artifacts.extend(self._extract_artifacts_from_content(get_text_content(message)))

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
        artifacts.extend(m.group(1) for m in ABSOLUTE_PATH_RE.finditer(content))
        artifacts.extend(m.group(1) for m in RELATIVE_PATH_RE.finditer(content))

        # Inline code
        artifacts.extend(
            self._strip_punctuation(m.group(1))
            for m in BACKTICK_RE.finditer(content)
        )

        # Error lines
        artifacts.extend(
            self._strip_punctuation(m.group(0).strip())
            for m in ERROR_RE.finditer(content)
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
        match = SHELL_COMMAND_RE.match(line)
        if not match:
            return False
        first_word = match.group(1).split()[0]
        return any(
            first_word == prefix or first_word.startswith(f"{prefix}/")
                for prefix in KNOWN_COMMAND_PREFIXES
        )

    def _finalize_summary_output(self, raw_summary: str) -> str:
        finalized = raw_summary.replace("\r\n", "\n").strip()
        return finalized or "- no summary returned"

    def _create_summary_message(self, summary: str) -> Message:
        """Wrap summary in a system message with the history prefix."""
        return {
            "role": "system",
            "content": f"{SUMMARY_SYSTEM_PREFIX}\n\n{summary}".strip(),
        }

    def _strip_punctuation(self, text: str) -> str:
        """Remove surrounding punctuation from artifact strings."""
        return text.strip().strip("()[]{}.,:;")
