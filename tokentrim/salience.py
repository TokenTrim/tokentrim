from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_./-]{3,}")
_PATH_RE: Final[re.Pattern[str]] = re.compile(r"(?:~|/|\.\.?/)[^\s]+")
_COMMAND_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^\$\s+(.+)$")
_ERROR_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^.*(?:error|exception|traceback|failed|failure|enoent|permission denied).*$"
)
_SUCCESS_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:fixed|resolved|success|succeeded|works now|passed|green|done)\b"
)
_CONSTRAINT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:do not|don't|must|only|avoid|never|without)\b"
)


@dataclass(frozen=True, slots=True)
class SalienceSignals:
    overlap_terms: int
    has_error: bool
    has_constraint: bool
    has_command: bool
    has_path: bool
    has_success: bool
    artifact_count: int


def extract_query_terms(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in _iter_terms(text):
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return tuple(ordered)


def score_text_salience(
    text: str,
    *,
    query_terms: Sequence[str] = (),
    recency_rank: int = 0,
) -> int:
    signals = analyze_text_salience(text, query_terms=query_terms)
    score = 0
    score += signals.overlap_terms * 12
    score += 25 if signals.has_error else 0
    score += 18 if signals.has_constraint else 0
    score += 10 if signals.has_command else 0
    score += 8 if signals.has_path else 0
    score += min(signals.artifact_count, 4) * 2
    score -= 12 if signals.has_success and not signals.has_error else 0
    score += max(0, 6 - recency_rank)
    return score


def analyze_text_salience(
    text: str,
    *,
    query_terms: Sequence[str] = (),
) -> SalienceSignals:
    lowered_terms = set(query_terms)
    terms = set(_iter_terms(text))
    overlap_terms = len(terms & lowered_terms) if lowered_terms else 0
    has_error = bool(_ERROR_RE.search(text))
    has_constraint = bool(_CONSTRAINT_RE.search(text))
    has_command = bool(_COMMAND_RE.search(text))
    has_path = bool(_PATH_RE.search(text))
    has_success = bool(_SUCCESS_RE.search(text))
    artifact_count = sum(
        (
            len(_PATH_RE.findall(text)),
            len(_COMMAND_RE.findall(text)),
            len(_ERROR_RE.findall(text)),
        )
    )
    return SalienceSignals(
        overlap_terms=overlap_terms,
        has_error=has_error,
        has_constraint=has_constraint,
        has_command=has_command,
        has_path=has_path,
        has_success=has_success,
        artifact_count=artifact_count,
    )


def _iter_terms(text: str) -> Iterable[str]:
    for match in _TOKEN_RE.finditer(text):
        yield match.group(0).lower()
    for match in _PATH_RE.finditer(text):
        yield match.group(0).lower()
