from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from tokentrim.types.message import Message

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_ABSOLUTE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[\s(])((?:~|/)[^\s:;,)\]]+)"
)
_RELATIVE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[\s(])((?:\.\.?/)[^\s:;,)\]]+)"
)
_BACKTICK_CODE_RE: Final[re.Pattern[str]] = re.compile(r"`([^`\n]+)`")
_ERROR_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^.*(?:error|exception|traceback|failed|failure|enoent|permission denied).*$"
)
_SUCCESS_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^.*(?:fixed|resolved|success|succeeded|works now|passed|green|done).*$"
)
_CONSTRAINT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:do not|don't|must|only|avoid|never|without)\b"
)
_TASK_WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:fix|debug|check|run|rerun|update|edit|change|add|remove|inspect|open|review|implement|create|test|use|preserve)\b"
)
_NEXT_STEP_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:next|i(?:'ll| will)|then|rerun|retry|check|inspect|update|implement|edit|run)\b"
)
_COMMAND_PREFIXES: Final[frozenset[str]] = frozenset(
    ("git", "python", "python3", "pytest", "uv", "pip", "npm", "cargo", "bash", "sh", "ls", "cat", "sed", "rg")
)
_MAX_ACTIVE_FILES: Final[int] = 5
_MAX_CONSTRAINTS: Final[int] = 3
_LINE_LIMIT: Final[int] = 180
WORKING_STATE_PREFIX: Final[str] = "Working state only."


@dataclass(frozen=True, slots=True)
class WorkingState:
    goal: str | None
    current_step: str | None
    active_files: tuple[str, ...]
    latest_command: str | None
    active_error: str | None
    constraints: tuple[str, ...]
    next_step: str | None

    def is_empty(self) -> bool:
        return not any(
            (
                self.goal,
                self.current_step,
                self.active_files,
                self.latest_command,
                self.active_error,
                self.constraints,
                self.next_step,
            )
        )


def extract_working_state(messages: Sequence[Message]) -> WorkingState:
    return WorkingState(
        goal=_extract_goal(messages),
        current_step=_extract_current_step(messages),
        active_files=_extract_active_files(messages),
        latest_command=_extract_latest_command(messages),
        active_error=_extract_active_error(messages),
        constraints=_extract_constraints(messages),
        next_step=_extract_next_step(messages),
    )


def render_working_state_message(state: WorkingState) -> Message | None:
    if state.is_empty():
        return None

    lines = [WORKING_STATE_PREFIX]
    if state.goal:
        lines.append(f"Goal: {state.goal}")
    if state.current_step:
        lines.append(f"Current Step: {state.current_step}")
    if state.active_files:
        lines.append("Active Files: " + ", ".join(state.active_files))
    if state.latest_command:
        lines.append(f"Latest Command: {state.latest_command}")
    if state.active_error:
        lines.append(f"Active Error: {state.active_error}")
    if state.constraints:
        lines.append("Constraints: " + " | ".join(state.constraints))
    if state.next_step:
        lines.append(f"Next Step: {state.next_step}")
    return {"role": "system", "content": "\n".join(lines)}


def parse_working_state_message(message: Message) -> WorkingState | None:
    if message["role"] != "system":
        return None
    content = message["content"]
    if not content.startswith(WORKING_STATE_PREFIX):
        return None
    parsed: dict[str, str] = {}
    for line in content.splitlines()[1:]:
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        parsed[key.strip().lower().replace(" ", "_")] = value.strip()
    return WorkingState(
        goal=_empty_to_none(parsed.get("goal")),
        current_step=_empty_to_none(parsed.get("current_step")),
        active_files=_split_csv(parsed.get("active_files")),
        latest_command=_empty_to_none(parsed.get("latest_command")),
        active_error=_empty_to_none(parsed.get("active_error")),
        constraints=_split_pipe(parsed.get("constraints")),
        next_step=_empty_to_none(parsed.get("next_step")),
    )


def find_working_state(messages: Sequence[Message]) -> WorkingState | None:
    for message in reversed(messages):
        parsed = parse_working_state_message(message)
        if parsed is not None:
            return parsed
    return None


def _extract_goal(messages: Sequence[Message]) -> str | None:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        candidate = _normalize_text(message["content"])
        if candidate and _looks_task_oriented(candidate):
            return _truncate_line(candidate)
    return None


def _extract_current_step(messages: Sequence[Message]) -> str | None:
    for message in reversed(messages):
        if message["role"] != "assistant":
            continue
        candidate = _normalize_text(message["content"])
        if candidate and _NEXT_STEP_RE.search(candidate):
            return _truncate_line(candidate)
    return None


def _extract_active_files(messages: Sequence[Message]) -> tuple[str, ...]:
    seen: set[str] = set()
    files: list[str] = []
    for message in reversed(messages):
        for candidate in _extract_paths(message["content"]):
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            files.append(normalized)
            if len(files) >= _MAX_ACTIVE_FILES:
                return tuple(files)
    return tuple(files)


def _extract_latest_command(messages: Sequence[Message]) -> str | None:
    for message in reversed(messages):
        lines = list(_iter_nonempty_lines(message["content"]))
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.startswith("$ "):
                return _truncate_line(stripped[2:].strip())
            if _looks_like_command(stripped):
                return _truncate_line(stripped)
        for candidate in reversed(tuple(_BACKTICK_CODE_RE.findall(message["content"]))):
            normalized = _normalize_text(candidate)
            if _looks_like_command(normalized):
                return _truncate_line(normalized)
    return None


def _extract_active_error(messages: Sequence[Message]) -> str | None:
    saw_later_success = False
    for message in reversed(messages):
        lines = list(_iter_nonempty_lines(message["content"]))
        for line in reversed(lines):
            stripped = line.strip()
            if _SUCCESS_LINE_RE.search(stripped):
                saw_later_success = True
            if _ERROR_LINE_RE.search(stripped) and not saw_later_success:
                return _truncate_line(_normalize_text(stripped))
    return None


def _extract_constraints(messages: Sequence[Message]) -> tuple[str, ...]:
    seen: set[str] = set()
    constraints: list[str] = []
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        for line in _iter_nonempty_lines(message["content"]):
            normalized = _normalize_text(line)
            if not normalized or not _CONSTRAINT_RE.search(normalized):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            constraints.append(_truncate_line(normalized))
            if len(constraints) >= _MAX_CONSTRAINTS:
                return tuple(constraints)
    return tuple(constraints)


def _extract_next_step(messages: Sequence[Message]) -> str | None:
    for message in reversed(messages):
        candidate = _normalize_text(message["content"])
        if candidate and _NEXT_STEP_RE.search(candidate):
            return _truncate_line(candidate)
    return None


def _extract_paths(content: str) -> Iterable[str]:
    for match in _ABSOLUTE_PATH_RE.finditer(content):
        yield match.group(1)
    for match in _RELATIVE_PATH_RE.finditer(content):
        yield match.group(1)
    for candidate in _BACKTICK_CODE_RE.findall(content):
        normalized = candidate.strip()
        if "/" in normalized or normalized.endswith((".py", ".md", ".json", ".yaml", ".yml", ".txt")):
            yield normalized


def _iter_nonempty_lines(content: str) -> Iterable[str]:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            yield stripped


def _looks_task_oriented(text: str) -> bool:
    return bool(_TASK_WORD_RE.search(text) or _CONSTRAINT_RE.search(text))


def _looks_like_command(text: str) -> bool:
    if not text or " " not in text:
        return False
    return text.split()[0] in _COMMAND_PREFIXES


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _truncate_line(value: str) -> str:
    if len(value) <= _LINE_LIMIT:
        return value
    return value[: _LINE_LIMIT - 3].rstrip() + "..."


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _split_pipe(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split("|") if part.strip())
