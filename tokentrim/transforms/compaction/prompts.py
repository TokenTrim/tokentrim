"""Prompt templates for LLM-backed conversation compaction.

This module provides the prompt templates used to generate summaries
from conversation history. Templates define the system instructions and
user message format for the summarization LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Configuration for the compaction prompt."""

    system_prompt: str
    user_template: str


STRUCTURED_PROMPT_TEMPLATE: Final[PromptTemplate] = PromptTemplate(
    system_prompt=(
        "Summarise the conversation into a compact engineering handoff. "
        "Return plaintext only. Keep exact technical details when they matter."
    ),
    user_template=(
        "Write a concise handoff using these exact headings:\n"
        "Goal:\n"
        "Active State:\n"
        "Critical Artifacts:\n"
        "Open Risks:\n"
        "Next Step:\n"
        "Older Context:\n"
        "Preserve commands, file paths, identifiers, and error strings exactly when relevant.\n"
        "Artifacts that must survive if relevant:\n"
        "{required_artifacts}\n\n"
        "Conversation history:\n{history}"
    ),
)


def build_prompt_template(instructions: str | None) -> PromptTemplate:
    normalized = instructions.strip() if instructions is not None else ""
    if not normalized:
        return STRUCTURED_PROMPT_TEMPLATE
    return PromptTemplate(
        system_prompt=normalized,
        user_template=STRUCTURED_PROMPT_TEMPLATE.user_template,
    )
