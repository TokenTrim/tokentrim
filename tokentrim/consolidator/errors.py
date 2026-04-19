from tokentrim.errors.base import TokentrimError


class ConsolidatorRuntimeBaseError(TokentrimError):
    """Base error for consolidator runtime transform failures."""


class ConsolidatorRuntimeConfigurationError(ConsolidatorRuntimeBaseError):
    """Raised when consolidator runtime configuration is invalid or incomplete."""


class ConsolidatorRuntimeError(ConsolidatorRuntimeBaseError):
    """Raised when consolidator runtime fails unexpectedly."""
