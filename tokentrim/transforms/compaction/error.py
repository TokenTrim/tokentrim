from tokentrim.errors.base import TokentrimError


class CompactionError(TokentrimError):
    """Base error for compaction transform failures."""


class CompactionConfigurationError(CompactionError):
    """Raised when compaction settings are invalid or incomplete."""


class CompactionExecutionError(CompactionError):
    """Raised when compaction generation fails unexpectedly."""


class CompactionOutputError(CompactionError):
    """Raised when compaction output is unsafe or structurally invalid."""
