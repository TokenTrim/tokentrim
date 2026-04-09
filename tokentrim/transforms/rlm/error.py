from tokentrim.errors.base import TokentrimError


class RLMTransformError(TokentrimError):
    """Base error for RLM retrieval transform failures."""


class RLMConfigurationError(RLMTransformError):
    """Raised when RLM retrieval configuration is invalid or incomplete."""


class RLMExecutionError(RLMTransformError):
    """Raised when RLM retrieval fails unexpectedly."""
