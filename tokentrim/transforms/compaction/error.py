"""Custom exception classes for compaction transform errors.

This module defines a hierarchy of exceptions for different compaction
failure modes: configuration errors and execution errors.
"""

from tokentrim.errors.base import TokentrimError


class CompactionError(TokentrimError):
    """Base error for compaction transform failures."""


class CompactionConfigurationError(CompactionError):
    """Raised when compaction settings are invalid or incomplete."""


class CompactionExecutionError(CompactionError):
    """Raised when compaction generation fails unexpectedly."""
