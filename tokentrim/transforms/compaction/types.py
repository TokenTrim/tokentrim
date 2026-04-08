"""Type definitions and result dataclasses for compaction transforms.

This module centralizes all shared type aliases and data structures
used across the compaction module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tokentrim.types.message import Message

# =============================================================================
# Type Aliases
# =============================================================================

AgeBand = Literal["recent", "old", "mature"]
"""How old a message group is relative to the conversation end."""

GroupKind = Literal["tool_round", "dialogue_round", "single", "protected"]
"""Classification of a message group for microcompact processing."""

ContextEditGroupKind = Literal["protected", "tool_round", "assistant_plan", "message"]
"""Classification of a message group for context edit processing."""

MicrocompactPressure = Literal["normal", "high"]
"""Compression aggressiveness level for microcompact."""

# =============================================================================
# Message Group Dataclasses
# =============================================================================


@dataclass(frozen=True, slots=True)
class MessageGroup:
    """A group of related messages for compaction processing.

    Attributes:
        messages: The messages in this group.
        kind: Classification of the group type.
        age_band: How old this group is relative to conversation end.
        token_count: Approximate token count for the group.
    """

    messages: tuple[Message, ...]
    kind: GroupKind
    age_band: AgeBand
    token_count: int


@dataclass(frozen=True, slots=True)
class ContextEditMessageGroup:
    """A group of related messages for context edit processing.

    Attributes:
        messages: The messages in this group.
        kind: Classification of the group type.
    """

    messages: tuple[Message, ...]
    kind: ContextEditGroupKind


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass(frozen=True, slots=True)
class MicrocompactPlan:
    """Result of planning microcompaction on a message list.

    Attributes:
        messages: The resulting messages after compaction.
        original_tokens: Token count before compaction.
        compacted_tokens: Token count after compaction.
        groups_seen: Total number of message groups processed.
        groups_compacted: Number of groups that were compacted.
    """

    messages: list[Message]
    original_tokens: int
    compacted_tokens: int
    groups_seen: int
    groups_compacted: int

    @property
    def tokens_saved(self) -> int:
        """Calculate tokens saved by compaction."""
        return self.original_tokens - self.compacted_tokens


@dataclass(frozen=True, slots=True)
class ContextEditStats:
    """Statistics from a context edit operation.

    Attributes:
        removed_messages: Total messages removed.
        removed_tool_rounds: Number of tool rounds removed.
        removed_resolved_errors: Number of resolved error rounds removed.
        removed_redundant_plans: Number of redundant assistant plans removed.
    """

    removed_messages: int = 0
    removed_tool_rounds: int = 0
    removed_resolved_errors: int = 0
    removed_redundant_plans: int = 0


@dataclass(frozen=True, slots=True)
class ContextEditResult:
    """Result of a context edit operation.

    Attributes:
        messages: The resulting messages after editing.
        stats: Statistics about what was removed.
    """

    messages: list[Message]
    stats: ContextEditStats


# =============================================================================
# Compaction Strategy
# =============================================================================

CompactionStrategy = Literal["aggressive", "balanced", "minimal"]
"""Pre-configured compression strategy.

- aggressive: Maximize compression, minimize preserved messages
- balanced: Default balance between compression and preservation
- minimal: Prefer keeping more context, compress less
"""


# =============================================================================
# Compaction Metrics
# =============================================================================


@dataclass(frozen=True, slots=True)
class CompactionMetrics:
    """Detailed metrics from a compaction operation.

    Provides observability into compression effectiveness and behavior.

    Attributes:
        original_tokens: Token count before compaction.
        compacted_tokens: Token count after compaction.
        original_messages: Message count before compaction.
        compacted_messages: Message count after compaction.
        groups_seen: Total message groups analyzed.
        groups_compacted: Number of groups that were compressed.
        context_edit_removed: Messages removed by context edit pass.
        microcompact_saved: Tokens saved by microcompact pass.
        strategy_used: The compression strategy that was applied.
    """

    original_tokens: int
    compacted_tokens: int
    original_messages: int
    compacted_messages: int
    groups_seen: int = 0
    groups_compacted: int = 0
    context_edit_removed: int = 0
    microcompact_saved: int = 0
    strategy_used: CompactionStrategy = "balanced"

    @property
    def tokens_saved(self) -> int:
        """Total tokens saved by compaction."""
        return self.original_tokens - self.compacted_tokens

    @property
    def compression_ratio(self) -> float:
        """Compression ratio (0.0 = no compression, 1.0 = 100% compression)."""
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compacted_tokens / self.original_tokens)

    @property
    def messages_removed(self) -> int:
        """Number of messages removed."""
        return self.original_messages - self.compacted_messages
