"""Compaction transforms for conversation history compression.

This package provides transforms for reducing conversation context size
while preserving important information. It includes both deterministic
pre-processing (microcompact, context edit) and LLM-backed summarization.

Public API:
    CompactConversation: Main transform for compacting conversation history.
"""

from tokentrim.transforms.compaction.transform import CompactConversation

__all__ = [
    "CompactConversation",
]
