from __future__ import annotations

import os
import re
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tokentrim.core.token_counting import count_message_tokens
from tokentrim.pipeline.requests import ContextRequest
from tokentrim.transforms.compaction import CompactConversation
from tokentrim.types.message import get_text_content
from tokentrim.types.state import PipelineState


def _load_env_test() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[4]
    env_path = repo_root / ".env.test"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _live_settings() -> tuple[str | None, dict[str, str] | None]:
    loaded = _load_env_test()
    enabled = os.getenv("TOKENTRIM_LIVE_COMPACTION") or loaded.get("TOKENTRIM_LIVE_COMPACTION")
    if enabled not in {"1", "true", "TRUE", "yes", "YES"}:
        return None, None

    model = os.getenv("TOKENTRIM_TEST_MODEL") or loaded.get("TOKENTRIM_TEST_MODEL")
    if not model:
        return None, None

    options: dict[str, str] = {}
    api_key = os.getenv("TOKENTRIM_TEST_API_KEY") or loaded.get("TOKENTRIM_TEST_API_KEY")
    api_base = os.getenv("TOKENTRIM_TEST_API_BASE") or loaded.get("TOKENTRIM_TEST_API_BASE")
    if api_key:
        options["api_key"] = api_key
    if api_base:
        options["api_base"] = api_base
    return model, options or None


def _has_litellm() -> bool:
    try:
        import litellm  # noqa: F401
    except ImportError:
        return False
    return True


def _artifact_dir(test_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    path = repo_root / "tests" / "artifacts" / test_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_model_suffix(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "unknown_model"


def _render_messages(messages: list[dict[str, object]]) -> str:
    sections: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "unknown"))
        lines = [f"Message {index}", f"role: {role}"]

        name = message.get("name")
        if name:
            lines.append(f"name: {name}")

        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            lines.append(f"tool_call_id: {tool_call_id}")

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            rendered_tool_calls: list[str] = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if isinstance(function, dict):
                    function_name = function.get("name", "unknown")
                    rendered_tool_calls.append(str(function_name))
            if rendered_tool_calls:
                lines.append(f"tool_calls: {', '.join(rendered_tool_calls)}")

        lines.append("content:")
        text_content = get_text_content(message).strip()
        lines.append(text_content or "<empty>")
        sections.append("\n".join(lines))
    return "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(sections) + "\n"


def _write_artifacts(
    *,
    test_name: str,
    before_messages: list[dict[str, object]],
    after_messages: list[dict[str, object]],
    model: str,
    elapsed_seconds: float,
    before_tokens: int,
    after_tokens: int,
    llm_prompt: list[dict[str, object]] | None = None,
    llm_output: str | None = None,
) -> Path:
    artifact_dir = _artifact_dir(test_name)
    tokens_saved = before_tokens - after_tokens
    compression_ratio = (tokens_saved / before_tokens) if before_tokens else 0.0

    (artifact_dir / "before.txt").write_text(_render_messages(before_messages))
    (artifact_dir / "after.txt").write_text(_render_messages(after_messages))
    if llm_prompt is not None:
        (artifact_dir / "llm_input.txt").write_text(_render_messages(llm_prompt))
    if llm_output is not None:
        (artifact_dir / "llm_output.txt").write_text(llm_output.rstrip() + "\n")
    (artifact_dir / "metrics.txt").write_text(
        "\n".join(
            [
                f"model={model}",
                f"elapsed_seconds={elapsed_seconds:.4f}",
                f"before_tokens={before_tokens}",
                f"after_tokens={after_tokens}",
                f"saved_tokens={tokens_saved}",
                f"compression_percent={compression_ratio * 100:.2f}",
            ]
        )
        + "\n"
    )
    return artifact_dir


def _request(*, token_budget: int | None) -> ContextRequest:
    return ContextRequest(
        messages=tuple(),
        user_id=None,
        session_id=None,
        org_id=None,
        token_budget=token_budget,
        steps=(CompactConversation(),),
    )


@pytest.mark.live
@pytest.mark.skipif(
    _live_settings()[0] is None or not _has_litellm(),
    reason="Live compaction stress test requires opt-in settings and litellm installed.",
)
def test_live_compaction_stress_round_trip() -> None:
    model, model_options = _live_settings()
    assert model is not None
    test_name = f"test_live_compaction_stress_round_trip_{_safe_model_suffix(model)}"
    captured: dict[str, object] = {}

    step = CompactConversation(
        model=model,
        model_options=model_options,
        keep_last=8,
        strategy="balanced",
        instructions=(
            "Produce a compact engineering handoff. Preserve exact commands, "
            "paths, unresolved errors, and image/tool references when relevant."
        ),
    )

    messages: list[dict[str, object]] = [
        {"role": "system", "content": "System instruction: stay scoped to compaction."},
        {"role": "user", "content": "Do not widen scope. Keep the latest turns verbatim."},
    ]
    for index in range(18):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"$ pytest tests/case_{index}.py\n"
                                "Traceback (most recent call last):\n"
                                f"FileNotFoundError: fixture_{index}\n"
                                f"Inspect ./tokentrim/file_{index}.py\n"
                                + ("verbose tool output\n" * 8)
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"https://example.com/debug_{index}.png"},
                        },
                    ],
                    "tool_calls": [
                        {
                            "id": f"call_{index:04d}",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "[exit_code] 1\nstderr: failed\nstdout: rerun with --verbose",
                    "tool_call_id": f"call_{index:04d}",
                    "name": "search",
                },
                {"role": "assistant", "content": f"I'll fix fixture_{index} and rerun the targeted tests."},
                {"role": "user", "content": f"Continue and preserve ./tokentrim/file_{index}.py."},
            ]
        )

    before_tokens = count_message_tokens(messages, None)
    started_at = time.perf_counter()
    real_generate_text = (
        __import__("tokentrim.transforms.compaction.transform", fromlist=["generate_text"]).generate_text
    )

    def capture_generate_text(**kwargs):
        captured["llm_prompt"] = kwargs.get("messages")
        output = real_generate_text(**kwargs)
        captured["llm_output"] = output
        return output

    with patch("tokentrim.transforms.compaction.transform.generate_text", capture_generate_text):
        result = step.run(PipelineState(context=messages, tools=[]), _request(token_budget=2_400))
    elapsed_seconds = time.perf_counter() - started_at
    after_tokens = count_message_tokens(result.context, None)
    tokens_saved = before_tokens - after_tokens
    compression_ratio = (tokens_saved / before_tokens) if before_tokens else 0.0
    artifact_dir = _write_artifacts(
        test_name=test_name,
        before_messages=messages,
        after_messages=result.context,
        model=model,
        elapsed_seconds=elapsed_seconds,
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        llm_prompt=captured.get("llm_prompt"),
        llm_output=captured.get("llm_output"),
    )

    assert result.context[0]["role"] == "system"
    assert result.context[1]["role"] == "system"
    assert "History only." in str(result.context[1]["content"])
    assert "Goal:" in str(result.context[1]["content"])
    assert "Critical Artifacts:" in str(result.context[1]["content"])

    print(
        "live compaction: "
        f"model={model} "
        f"elapsed={elapsed_seconds:.2f}s "
        f"before_tokens={before_tokens} "
        f"after_tokens={after_tokens} "
        f"saved_tokens={tokens_saved} "
        f"compression={compression_ratio * 100:.2f}% "
        f"artifacts={artifact_dir}"
    )
