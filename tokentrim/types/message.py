"""Message types for conversation context.

This module defines the Message TypedDict and related content types
for representing conversation messages with support for:
- Text content (simple string or structured text part)
- Multimodal content (images via URL or base64)
- Tool calls and tool results
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


# =============================================================================
# Content Part Types (for multimodal messages)
# =============================================================================


class TextContentPart(TypedDict):
    """A text content part within a multimodal message."""

    type: str  # "text"
    text: str


class ImageUrlDetail(TypedDict, total=False):
    """URL details for an image content part."""

    url: str
    detail: str  # "auto", "low", "high"


class ImageUrlContentPart(TypedDict):
    """An image URL content part within a multimodal message."""

    type: str  # "image_url"
    image_url: ImageUrlDetail


# Union of possible content parts
ContentPart = TextContentPart | ImageUrlContentPart | dict[str, Any]

# Content can be a simple string or a list of content parts
MessageContent = str | list[ContentPart]


# =============================================================================
# Tool Call Types
# =============================================================================


class FunctionCall(TypedDict, total=False):
    """A function call within a tool call."""

    name: str
    arguments: str  # JSON string


class ToolCall(TypedDict, total=False):
    """A tool call made by the assistant."""

    id: str
    type: str  # "function"
    function: FunctionCall


# =============================================================================
# Message Type
# =============================================================================


class Message(TypedDict, total=False):
    """A conversation message with support for multimodal and tool content.

    Required fields:
        role: The role of the message sender ("system", "user", "assistant", "tool").

    Content fields (at least one typically present):
        content: Text content (str) or multimodal content (list of parts).

    Tool-related fields (for function calling):
        tool_calls: List of tool calls made by an assistant message.
        tool_call_id: ID linking a tool result to its call (for role="tool").
        name: Function/tool name (for role="tool" or role="function").

    The content field can be:
        - A simple string: "Hello, world!"
        - A list of content parts for multimodal:
            [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://..."}}
            ]
    """

    # Required
    role: str

    # Content (string or multimodal list)
    content: MessageContent

    # Tool calls (assistant making calls)
    tool_calls: list[ToolCall]

    # Tool result linkage (tool returning result)
    tool_call_id: str
    name: str  # function/tool name


# =============================================================================
# Helper Functions
# =============================================================================


def get_text_content(message: Message) -> str:
    """Extract text content from a message, handling multimodal format.

    For simple string content, returns the string.
    For multimodal content (list), extracts and joins all text parts.
    For None/missing content, returns empty string.

    Args:
        message: The message to extract text from.

    Returns:
        The text content as a string.
    """
    content = message.get("content")

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    # Handle multimodal content (list of parts)
    text_parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
    return "\n".join(text_parts)


def has_images(message: Message) -> bool:
    """Check if a message contains image content.

    Args:
        message: The message to check.

    Returns:
        True if the message contains image_url parts.
    """
    content = message.get("content")

    if not isinstance(content, list):
        return False

    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in content
    )


def extract_image_refs(message: Message) -> list[str]:
    """Extract image references from a multimodal message.

    Args:
        message: The message to extract images from.

    Returns:
        List of image URLs or "[base64 image]" placeholders.
    """
    content = message.get("content")

    if not isinstance(content, list):
        return []

    refs: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            image_url = part.get("image_url", {})
            url = image_url.get("url", "")
            if url.startswith("data:"):
                refs.append("[base64 image]")
            elif url:
                # Extract filename from URL if possible
                filename = url.rsplit("/", 1)[-1].split("?")[0]
                refs.append(f"[image: {filename}]" if filename else url)
    return refs


def has_tool_calls(message: Message) -> bool:
    """Check if a message contains tool calls.

    Args:
        message: The message to check.

    Returns:
        True if the message has non-empty tool_calls.
    """
    return bool(message.get("tool_calls"))


def is_tool_result(message: Message) -> bool:
    """Check if a message is a tool result.

    Args:
        message: The message to check.

    Returns:
        True if the message has role="tool" or has tool_call_id.
    """
    return message.get("role") == "tool" or bool(message.get("tool_call_id"))
