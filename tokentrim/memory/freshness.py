from __future__ import annotations

from datetime import UTC, datetime


def memory_freshness_bucket(timestamp: str) -> str:
    days = memory_age_days(timestamp)
    if days <= 1:
        return "fresh"
    if days <= 7:
        return "aging"
    return "stale"


def memory_age_days(timestamp: str) -> int:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return 0
    now = datetime.now(UTC)
    return max(0, (now - dt).days)


def memory_age(timestamp: str) -> str:
    days = memory_age_days(timestamp)
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def memory_freshness_text(timestamp: str) -> str:
    days = memory_age_days(timestamp)
    bucket = memory_freshness_bucket(timestamp)
    if bucket == "fresh":
        return ""
    if bucket == "aging":
        return (
            f"This memory is {days} days old. "
            "Treat it as a reusable hint. "
            "Verify code, files, flags, and current project state before relying on it."
        )
    return (
        f"This memory is {days} days old. "
        "Memories are point-in-time observations, not live state. "
        "Verify code, files, flags, and current project state before relying on it."
    )


def memory_freshness_note(timestamp: str) -> str:
    text = memory_freshness_text(timestamp)
    if not text:
        return ""
    return f"<system-reminder>{text}</system-reminder>"
