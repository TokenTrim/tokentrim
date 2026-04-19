"""Compiled regex patterns for compaction transforms.

This module centralizes all regex patterns used across the compaction module
to avoid duplication and ensure consistency.
"""

from __future__ import annotations

import re
from typing import Final

# =============================================================================
# Command & Terminal Detection
# =============================================================================

# Matches shell command line prefixes: $ git status
COMMAND_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^\$\s+(.+)$")

# Matches terminal/tool content markers (case-insensitive, multiline)
TERMINAL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)(?:\[command\]|\[terminal\]|\[output\]|\[exit_code\]|traceback|stderr|stdout|^\$\s+)"
)

# Matches shell commands with known prefixes: git, python, pytest, etc.
SHELL_COMMAND_RE: Final[re.Pattern[str]] = re.compile(
    r"(?m)^(?:\$|#)?\s*([a-zA-Z0-9_.:/-]+(?:\s+[^\n]+)?)$"
)

# Known command prefixes for shell command detection
KNOWN_COMMAND_PREFIXES: Final[frozenset[str]] = frozenset(
    ("git", "python", "python3", "pytest", "uv", "pip", "npm", "cargo", "bash", "sh")
)

# =============================================================================
# Exit Code & Error Detection
# =============================================================================

# Matches exit code markers: [exit_code] 0, [exit_code] 1
EXIT_CODE_RE: Final[re.Pattern[str]] = re.compile(r"(?im)\[exit_code\]\s+(\d+)")

# Matches error/exception lines (case-insensitive, multiline)
ERROR_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^.*(?:error|exception|traceback|failed|failure|enoent|permission denied).*$"
)

# Matches success indicators (case-insensitive)
SUCCESS_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:fixed|resolved|success|succeeded|works now|passed|green|done)\b"
)

# =============================================================================
# Path & Artifact Extraction
# =============================================================================

# Matches absolute paths: /usr/bin/python, ~/Documents/file.txt
ABSOLUTE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[\s(])((?:~|/)[^\s:;,)\]]+)"
)

# Matches relative paths: ./src/main.py, ../config.yaml
RELATIVE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[\s(])((?:\.\.?/)[^\s:;,)\]]+)"
)

# Combined path pattern (absolute or relative)
PATH_RE: Final[re.Pattern[str]] = re.compile(r"(?:^|[\s(])((?:~|/|\.\.?/)[^\s:;,)\]]+)")

# Matches inline code spans: `variable_name`, `some_command`
BACKTICK_RE: Final[re.Pattern[str]] = re.compile(r"`([^`\n]+)`")

# =============================================================================
# Content Classification
# =============================================================================

# Matches constraint keywords in user messages
CONSTRAINT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:do not|don't|must|only|avoid|never|without)\b"
)

# Matches assistant planning language
PLAN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:i(?:'ll| will)|let me|next|plan|first|then|going to|review|inspect|update|rerun|check|implement)\b"
)

# =============================================================================
# Whitespace Normalization
# =============================================================================

# Collapses consecutive whitespace
WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
