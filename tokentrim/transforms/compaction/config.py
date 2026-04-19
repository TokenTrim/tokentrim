"""Configuration constants and dataclasses for compaction transforms.

This module centralizes all default values and configuration classes
used across the compaction module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tokentrim.transforms.compaction.types import CompactionStrategy

# =============================================================================
# String Prefixes & Markers
# =============================================================================

MICROCOMPACT_PREFIX: Final[str] = "[microcompact]"
SUMMARY_SYSTEM_PREFIX: Final[str] = "History only."

# =============================================================================
# Microcompact Defaults
# =============================================================================

DEFAULT_MIN_CONTENT_CHARS: Final[int] = 280
DEFAULT_RECENT_GROUPS_TO_KEEP: Final[int] = 2
DEFAULT_RECENT_TOOL_GROUPS_TO_KEEP: Final[int] = 0
DEFAULT_MATURE_GROUP_AGE: Final[int] = 4
DEFAULT_MAX_COMMANDS: Final[int] = 2
DEFAULT_MAX_ERRORS: Final[int] = 2
DEFAULT_MAX_ARTIFACTS: Final[int] = 4
DEFAULT_TEXT_SNIPPET_CHARS: Final[int] = 96
DEFAULT_MIN_MESSAGES: Final[int] = 2
DEFAULT_MIN_TOKENS_SAVED: Final[int] = 1
DEFAULT_AGGRESSIVE_MIN_CONTENT_CHARS: Final[int] = 120
DEFAULT_AGGRESSIVE_RECENT_GROUPS_TO_KEEP: Final[int] = 1
DEFAULT_OVERSIZED_TOOL_RESULT_CHARS: Final[int] = 4_000
DEFAULT_OVERSIZED_TOOL_RESULT_TOKENS: Final[int] = 1_000

# =============================================================================
# CompactConversation Defaults
# =============================================================================

# Default number of recent messages to preserve without summarization
DEFAULT_KEEP_LAST: Final[int] = 6

# Default amount of room to reserve for the model response
DEFAULT_RESERVED_OUTPUT_TOKENS: Final[int] = 8_000

# Start compacting before the context window is actually full
DEFAULT_AUTO_COMPACT_BUFFER_TOKENS: Final[int] = 4_000

# Maximum number of artifacts (paths, commands, errors) to preserve in summary
MAX_PRESERVED_ARTIFACTS: Final[int] = 6

# =============================================================================
# Model Context Windows
# =============================================================================

MODEL_CONTEXT_WINDOW_PATTERNS: Final[tuple[tuple[str, int], ...]] = (
    ("gpt-4.1", 1_000_000),
    ("o4-mini", 200_000),
    ("o3", 200_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 128_000),
    ("claude", 200_000),
    ("mercury-2", 128_000),
)

# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass(frozen=True, slots=True)
class MicrocompactConfig:
    """Configuration for the microcompact deterministic compression pass."""

    min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS
    recent_groups_to_keep: int = DEFAULT_RECENT_GROUPS_TO_KEEP
    recent_tool_groups_to_keep: int = DEFAULT_RECENT_TOOL_GROUPS_TO_KEEP
    mature_group_age: int = DEFAULT_MATURE_GROUP_AGE
    max_commands: int = DEFAULT_MAX_COMMANDS
    max_errors: int = DEFAULT_MAX_ERRORS
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS
    text_snippet_chars: int = DEFAULT_TEXT_SNIPPET_CHARS
    min_messages: int = DEFAULT_MIN_MESSAGES
    min_tokens_saved: int = DEFAULT_MIN_TOKENS_SAVED
    aggressive_min_content_chars: int = DEFAULT_AGGRESSIVE_MIN_CONTENT_CHARS
    aggressive_recent_groups_to_keep: int = DEFAULT_AGGRESSIVE_RECENT_GROUPS_TO_KEEP
    oversized_tool_result_chars: int = DEFAULT_OVERSIZED_TOOL_RESULT_CHARS
    oversized_tool_result_tokens: int = DEFAULT_OVERSIZED_TOOL_RESULT_TOKENS
    # Salience scoring settings
    use_salience_scoring: bool = True
    min_salience_to_protect: int = 30  # Don't compact groups above this salience score


@dataclass(frozen=True, slots=True)
class ContextEditConfig:
    """Configuration for the context editor deterministic pruning pass."""

    keep_latest_tool_round: bool = True
    collapse_repeated_assistant_plans: bool = True
    drop_resolved_errors: bool = True
    drop_completed_tool_rounds: bool = True


# =============================================================================
# Strategy Presets
# =============================================================================

# Pre-configured settings for different compression strategies

BALANCED_MICROCOMPACT_CONFIG: Final[MicrocompactConfig] = MicrocompactConfig()
"""Balanced microcompact settings: default tradeoff between compression and fidelity."""

BALANCED_CONTEXT_EDIT_CONFIG: Final[ContextEditConfig] = ContextEditConfig()
"""Balanced context edit settings: default pruning policy."""

AGGRESSIVE_MICROCOMPACT_CONFIG: Final[MicrocompactConfig] = MicrocompactConfig(
    min_content_chars=80,
    recent_groups_to_keep=1,
    recent_tool_groups_to_keep=0,
    mature_group_age=2,
    aggressive_min_content_chars=40,
    aggressive_recent_groups_to_keep=0,
)
"""Aggressive microcompact settings: compress more, preserve less."""

AGGRESSIVE_CONTEXT_EDIT_CONFIG: Final[ContextEditConfig] = ContextEditConfig(
    keep_latest_tool_round=True,
    collapse_repeated_assistant_plans=True,
    drop_resolved_errors=True,
    drop_completed_tool_rounds=True,
)
"""Aggressive context edit settings: drop more aggressively."""

MINIMAL_MICROCOMPACT_CONFIG: Final[MicrocompactConfig] = MicrocompactConfig(
    min_content_chars=500,
    recent_groups_to_keep=4,
    recent_tool_groups_to_keep=2,
    mature_group_age=8,
    aggressive_min_content_chars=300,
    aggressive_recent_groups_to_keep=3,
)
"""Minimal microcompact settings: preserve more context."""

MINIMAL_CONTEXT_EDIT_CONFIG: Final[ContextEditConfig] = ContextEditConfig(
    keep_latest_tool_round=True,
    collapse_repeated_assistant_plans=False,
    drop_resolved_errors=False,
    drop_completed_tool_rounds=False,
)
"""Minimal context edit settings: keep more context."""


def get_microcompact_config(strategy: CompactionStrategy) -> MicrocompactConfig:
    if strategy == "aggressive":
        return AGGRESSIVE_MICROCOMPACT_CONFIG
    if strategy == "minimal":
        return MINIMAL_MICROCOMPACT_CONFIG
    return BALANCED_MICROCOMPACT_CONFIG


def get_context_edit_config(strategy: CompactionStrategy) -> ContextEditConfig:
    if strategy == "aggressive":
        return AGGRESSIVE_CONTEXT_EDIT_CONFIG
    if strategy == "minimal":
        return MINIMAL_CONTEXT_EDIT_CONFIG
    return BALANCED_CONTEXT_EDIT_CONFIG
