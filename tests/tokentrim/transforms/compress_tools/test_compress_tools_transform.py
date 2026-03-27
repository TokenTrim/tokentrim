from __future__ import annotations

from tokentrim.pipeline.requests import ToolsRequest
from tokentrim.transforms.compress_tools import CompressToolDescriptions
from tokentrim.types.state import PipelineState


def _request() -> ToolsRequest:
    return ToolsRequest(
        tools=tuple(),
        task_hint=None,
        token_budget=None,
        steps=(CompressToolDescriptions(),),
    )


def test_bpe_normalizes_whitespace_and_preserves_schema() -> None:
    step = CompressToolDescriptions()
    tool = {
        "name": "search",
        "description": "search   the   knowledge\nbase",
        "input_schema": {"type": "object"},
    }

    result = step.run(PipelineState(context=[], tools=[tool]), _request()).tools[0]

    assert result["name"] == "search"
    assert result["description"] == "search the knowledge base"
    assert result["input_schema"] == {"type": "object"}


def test_bpe_returns_new_objects_and_deep_copies_schema() -> None:
    step = CompressToolDescriptions()
    tool = {
        "name": "search",
        "description": "docs",
        "input_schema": {"properties": {"query": {"type": "string"}}},
    }

    result = step.run(PipelineState(context=[], tools=[tool]), _request()).tools[0]
    result["input_schema"]["properties"]["query"]["type"] = "number"

    assert result is not tool
    assert tool["input_schema"]["properties"]["query"]["type"] == "string"


def test_bpe_trims_long_descriptions() -> None:
    step = CompressToolDescriptions()
    tool = {
        "name": "search",
        "description": "x" * 220,
        "input_schema": {"type": "object"},
    }

    result = step.run(PipelineState(context=[], tools=[tool]), _request()).tools[0]

    assert len(result["description"]) == 200
    assert result["description"].endswith("...")


def test_bpe_leaves_short_description_text_unchanged_after_normalization() -> None:
    step = CompressToolDescriptions()
    tool = {
        "name": "search",
        "description": "already short",
        "input_schema": {"type": "object"},
    }

    result = step.run(PipelineState(context=[], tools=[tool]), _request()).tools[0]

    assert result["description"] == "already short"
