from __future__ import annotations

from pathlib import Path


def build_session_memory_policy(
    *,
    session_dir: Path,
    entrypoint_path: Path,
    write_tool_name: str,
    read_tool_name: str,
) -> str:
    """Build the agent-facing session-memory policy text."""
    lines = [
        "# Session Memory",
        "",
        "You have access to Tokentrim session memory for this conversation.",
        f"You have a persistent, file-backed session memory directory at `{session_dir}`.",
        f"The session memory index is `{entrypoint_path}`.",
        "",
        "## How Memory Works",
        "",
        "Session memory is persistent across turns in the current session, but it is not durable user memory or org memory.",
        "The system decides what memories get injected into context. You decide what session memory is worth saving.",
        f"Use `{read_tool_name}` to inspect the current session memory index or a specific memory file.",
        f"Use `{write_tool_name}` to create or update one session memory file.",
        "",
        "## When To Save",
        "",
        "Save concrete information that is likely to matter again later in this same session.",
        "Good candidates include: active constraints, current repo or environment state, intermediate task facts, decisions already made, and stable preferences expressed during the session.",
        "Prefer saving memory when losing that fact would force you to rediscover it later.",
        "",
        "## What Not To Save",
        "",
        "Do not save generic turn summaries, chain-of-thought, temporary guesses, or noisy logs.",
        "Do not save facts that are already obvious from the current turn unless they will still matter later.",
        "Do not save durable cross-session user or org knowledge here.",
        "",
        "## How To Write",
        "",
        "Write one memory per useful fact cluster, not a dump of everything that happened.",
        "Use a short semantic title and description so the memory index stays readable.",
        "Update an existing memory instead of duplicating it when the same fact changes or becomes more precise.",
        "Prefer precise, factual content over narrative prose.",
        "",
        "## Trust And Freshness",
        "",
        "Memory is a hint, not ground truth.",
        "If current files, tool output, tests, logs, or user instructions conflict with memory, trust the current evidence.",
        "If recalled memory looks stale or contradicted, re-check reality and update the session memory.",
        "",
        "## Kinds",
        "",
        "Allowed kinds are: `constraint`, `active_state`, `task_fact`, `decision`, `preference`.",
        "",
        "Memory injection remains system-controlled. Save only what future turns should be able to reuse.",
    ]
    return "\n".join(lines)
