from tokentrim.errors.base import TokentrimError


class RLMTransformError(TokentrimError):
    """Base error for recursive-memory transform failures."""


class RLMConfigurationError(RLMTransformError):
    """Raised when recursive-memory configuration is invalid or incomplete."""


class RLMExecutionError(RLMTransformError):
    """Raised when recursive-memory synthesis fails unexpectedly."""
