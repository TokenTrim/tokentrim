from tokentrim.errors.base import TokentrimError


class ToolsTransformError(TokentrimError):
    """Base error for tools transform failures."""


class ToolCreationConfigurationError(ToolsTransformError):
    """Raised when tool creation settings are invalid or incomplete."""


class ToolCreationOutputError(ToolsTransformError):
    """Raised when tool creation output is malformed or invalid."""


class ToolCreationExecutionError(ToolsTransformError):
    """Raised when tool creation generation fails unexpectedly."""
