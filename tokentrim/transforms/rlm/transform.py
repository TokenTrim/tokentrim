from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tokentrim.core.token_counting import count_message_tokens
from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.transforms.rlm.error import RLMConfigurationError, RLMExecutionError
from tokentrim.transforms.rlm.runtime import LocalRLMRuntime, RLMContextView
from tokentrim.types.message import Message
from tokentrim.types.state import PipelineState

_RLM_INVOCATION_LOG_COLLECTOR: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "tokentrim_rlm_invocation_log_collector",
    default=None,
)

_CONTEXT_PREVIEW_CHAR_LIMIT = max(
    1, int(os.environ.get("TOKENTRIM_RLM_CONTEXT_PREVIEW_CHARS", "4000"))
)
_MEMORY_MESSAGE_PREFIX = "Retrieved memory:\n"


@dataclass(frozen=True, slots=True)
class RetrieveMemory(Transform):
    """Retrieve additive next-step memory from stored traces."""

    model: str
    backend: str = "openai"
    trace_limit: int = 8
    max_depth: int = 1
    max_iterations: int = 4
    max_subcalls: int = 4
    subcall_model: str | None = None
    max_memory_tokens: int = 512
    tokenizer_model: str | None = None

    @property
    def name(self) -> str:
        return "rlm"

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
    ) -> Transform:
        return replace(
            self,
            tokenizer_model=(
                self.tokenizer_model if self.tokenizer_model is not None else tokenizer_model
            ),
        )

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        if request.trace_store is None:
            return state
        if not request.user_id or not request.session_id:
            return state

        self._validate_configuration()
        traces = request.trace_store.list_traces(
            user_id=request.user_id,
            session_id=request.session_id,
            limit=self.trace_limit,
        )
        if not traces:
            return state

        ordered_traces = list(reversed(traces))
        retrieval_question = _build_retrieval_brief(
            messages=state.context,
            task_hint=request.task_hint,
        )
        structured_context = RLMContextView.from_history(
            messages=state.context,
            traces=ordered_traces,
            label="retrieve_memory",
        )
        synthesized_memory = self._generate_memory(
            context=structured_context,
            root_prompt=_build_memory_root_prompt(retrieval_question),
        ).strip()
        if not synthesized_memory:
            return state

        memory_content = _build_memory_message_content(
            synthesized_memory=synthesized_memory,
            current_messages=state.context,
            token_budget=request.token_budget,
            max_memory_tokens=self.max_memory_tokens,
            tokenizer_model=self.tokenizer_model,
        )
        if memory_content is None:
            return state
        return PipelineState(
            context=_insert_memory_message(
                current_messages=state.context,
                memory_content=memory_content,
            ),
            tools=state.tools,
        )

    def _validate_configuration(self) -> None:
        if not self.model.strip():
            raise RLMConfigurationError("RetrieveMemory requires a non-empty RLM model name.")
        if not self.backend.strip():
            raise RLMConfigurationError("RetrieveMemory requires a non-empty RLM backend name.")
        if self.trace_limit < 1:
            raise RLMConfigurationError("RetrieveMemory trace_limit must be at least 1.")
        if self.max_depth < 1:
            raise RLMConfigurationError("RetrieveMemory max_depth must be at least 1.")
        if self.max_depth != 1:
            raise RLMConfigurationError(
                "RetrieveMemory currently only supports max_depth=1. "
                "That means one depth-1 recursive subcall from the root runtime."
            )
        if self.max_iterations < 1:
            raise RLMConfigurationError("RetrieveMemory max_iterations must be at least 1.")
        if self.max_subcalls < 1:
            raise RLMConfigurationError("RetrieveMemory max_subcalls must be at least 1.")
        if self.max_memory_tokens < 1:
            raise RLMConfigurationError("RetrieveMemory max_memory_tokens must be at least 1.")

    def _generate_memory(self, *, context: RLMContextView, root_prompt: str) -> str:
        return _run_rlm_request(
            artifact_type="tokentrim_rlm_invocation",
            model=self.model,
            backend=self.backend,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            max_subcalls=self.max_subcalls,
            subcall_model=self.subcall_model,
            tokenizer_model=self.tokenizer_model,
            context=context,
            root_prompt=root_prompt,
            system_prompt=None,
            failure_message="RLM retrieval failed.",
            unresolved_message="RLM retrieval returned unresolved RLM control text",
            allow_empty_response=True,
            empty_response_status="no_memory",
        )


@contextmanager
def capture_rlm_invocation_logs() -> Iterator[list[dict[str, Any]]]:
    logs: list[dict[str, Any]] = []
    token = _RLM_INVOCATION_LOG_COLLECTOR.set(logs)
    try:
        yield logs
    finally:
        _RLM_INVOCATION_LOG_COLLECTOR.reset(token)


def _append_rlm_invocation_log(**payload: Any) -> None:
    collector = _RLM_INVOCATION_LOG_COLLECTOR.get()
    if collector is None:
        return
    try:
        serializable_payload = json.loads(json.dumps(payload, default=str))
    except Exception:
        serializable_payload = {
            "artifact_type": "tokentrim_rlm_invocation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "log_serialization_error",
            "error": "Failed to serialize RLM invocation log payload.",
        }
    collector.append(serializable_payload)


def _run_rlm_request(
    *,
    artifact_type: str,
    model: str,
    backend: str,
    max_depth: int,
    max_iterations: int,
    max_subcalls: int,
    subcall_model: str | None,
    tokenizer_model: str | None,
    context: RLMContextView,
    root_prompt: str,
    system_prompt: str | None,
    failure_message: str,
    unresolved_message: str,
    allow_empty_response: bool,
    empty_response_status: str,
) -> str:
    config = {
        "backend": backend,
        "model": model,
        "max_depth": max_depth,
        "max_iterations": max_iterations,
        "max_subcalls": max_subcalls,
        "subcall_model": subcall_model,
    }
    context_preview = _preview_context_payload(context)
    try:
        runtime = LocalRLMRuntime(
            model=model,
            backend=backend,
            max_iterations=max_iterations,
            tokenizer_model=tokenizer_model,
            max_depth=max_depth,
            max_subcalls=max_subcalls,
            subcall_model=subcall_model,
        )
    except Exception as exc:
        _append_rlm_invocation_log(
            artifact_type=artifact_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="runtime_init_error",
            config=config,
            context_preview=context_preview,
            root_prompt=root_prompt,
            response=None,
            synthesized_memory=None,
            trajectory=None,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise RLMExecutionError(failure_message) from exc

    try:
        response = runtime.run(context, root_prompt=root_prompt, system_prompt=system_prompt)
    except Exception as exc:
        error_message = str(exc) if isinstance(exc, RLMExecutionError) else failure_message
        _append_rlm_invocation_log(
            artifact_type=artifact_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="runtime_error",
            config=config,
            context_preview=context_preview,
            root_prompt=root_prompt,
            response=None,
            synthesized_memory=None,
            trajectory=runtime.trajectory,
            error=f"{type(exc).__name__}: {exc}",
        )
        if isinstance(exc, RLMExecutionError):
            raise
        raise RLMExecutionError(error_message) from exc

    if not isinstance(response, str):
        _append_rlm_invocation_log(
            artifact_type=artifact_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="invalid_response_type",
            config=config,
            context_preview=context_preview,
            root_prompt=root_prompt,
            response=response,
            synthesized_memory=None,
            trajectory=runtime.trajectory,
            error="RLM runtime did not return text content.",
        )
        raise RLMExecutionError("RLM runtime did not return text content.")

    if _looks_like_unresolved_rlm_control_output(response):
        debug_path = _write_unresolved_control_debug_artifact(
            prompt=context_preview,
            root_prompt=root_prompt,
            response=response,
            trajectory=runtime.trajectory,
            config=config,
        )
        _append_rlm_invocation_log(
            artifact_type=artifact_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="unresolved_control_output",
            config=config,
            context_preview=context_preview,
            root_prompt=root_prompt,
            response=response,
            synthesized_memory=None,
            trajectory=runtime.trajectory,
            debug_artifact_path=debug_path,
            error=unresolved_message,
        )
        debug_suffix = f" Debug artifact written to {debug_path}." if debug_path else ""
        raise RLMExecutionError(
            f"{unresolved_message} ({response.strip()!r}) instead of plain text.{debug_suffix}"
        )

    synthesized_memory = _unwrap_final_output(response)
    if not synthesized_memory.strip():
        if allow_empty_response:
            _append_rlm_invocation_log(
                artifact_type=artifact_type,
                created_at=datetime.now(timezone.utc).isoformat(),
                status=empty_response_status,
                config=config,
                context_preview=context_preview,
                root_prompt=root_prompt,
                response=response,
                synthesized_memory="",
                trajectory=runtime.trajectory,
                error=None,
            )
            return ""
        _append_rlm_invocation_log(
            artifact_type=artifact_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="empty_response",
            config=config,
            context_preview=context_preview,
            root_prompt=root_prompt,
            response=response,
            synthesized_memory=synthesized_memory,
            trajectory=runtime.trajectory,
            error="RLM retrieval returned empty memory.",
        )
        raise RLMExecutionError("RLM retrieval returned empty memory.")
    _append_rlm_invocation_log(
        artifact_type=artifact_type,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="ok",
        config=config,
        context_preview=context_preview,
        root_prompt=root_prompt,
        response=response,
        synthesized_memory=synthesized_memory,
        trajectory=runtime.trajectory,
        error=None,
    )
    return synthesized_memory


def _build_retrieval_brief(
    *,
    messages: list[Message],
    task_hint: str | None,
) -> str:
    top_level_task = _top_level_task(messages=messages, task_hint=task_hint)
    latest_explicit_user_request = _latest_explicit_user_request(messages)
    latest_assistant = _latest_message_content(messages, role="assistant")
    latest_runtime_observation = _latest_runtime_observation(messages)

    lines = [f"Top-level task:\n{_clip_text(top_level_task, 1600)}"]
    if latest_explicit_user_request and latest_explicit_user_request != top_level_task:
        lines.append(
            "Latest explicit user request:\n"
            f"{_clip_text(latest_explicit_user_request, 1600)}"
        )
    if latest_assistant:
        lines.append(f"Latest assistant action:\n{_clip_text(latest_assistant, 1600)}")
    if latest_runtime_observation:
        lines.append(
            "Latest observation/result:\n"
            f"{_clip_text(latest_runtime_observation, 2200)}"
        )
    lines.append(
        "Immediate retrieval objective:\n"
        "Infer the most likely next step from the latest state and retrieve only the run "
        "context needed to execute that step."
    )
    return "\n\n".join(lines)


def _looks_like_unresolved_rlm_control_output(text: str) -> bool:
    return bool(re.match(r"^\s*FINAL_VAR\s*\(", text))


def _write_unresolved_control_debug_artifact(
    *,
    prompt: str,
    root_prompt: str,
    response: str,
    trajectory: Any | None,
    config: dict[str, Any],
) -> str | None:
    try:
        artifact = _build_unresolved_control_debug_artifact(
            prompt=prompt,
            root_prompt=root_prompt,
            response=response,
            trajectory=trajectory,
            config=config,
        )
        debug_dir = Path(
            os.getenv(
                "TOKENTRIM_RLM_DEBUG_DIR",
                os.path.join(tempfile.gettempdir(), "tokentrim-rlm-debug"),
            )
        )
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        artifact_path = debug_dir / f"unresolved-final-var-{timestamp}-{uuid4().hex[:8]}.json"
        artifact_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return str(artifact_path)
    except Exception:
        return None


def _build_unresolved_control_debug_artifact(
    *,
    prompt: str,
    root_prompt: str,
    response: str,
    trajectory: Any | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    final_var_name = _extract_final_var_name(response)
    trajectory_json = _to_json(trajectory) if trajectory is not None else ""
    assignment_pattern = None
    has_assignment = False
    has_var_name = False
    if final_var_name:
        escaped_name = re.escape(final_var_name)
        assignment_pattern = rf"\b{escaped_name}\s*="
        has_assignment = bool(re.search(assignment_pattern, trajectory_json))
        has_var_name = bool(re.search(rf"\b{escaped_name}\b", trajectory_json))

    iteration_count = _extract_iteration_count(trajectory)
    hit_max_iterations = (
        iteration_count is not None and iteration_count >= int(config["max_iterations"])
    )

    return {
        "artifact_type": "tokentrim_rlm_unresolved_final_var",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "response": response,
        "final_var_name": final_var_name,
        "diagnosis": {
            "iteration_count": iteration_count,
            "hit_max_iterations": hit_max_iterations,
            "trajectory_mentions_final_var_name": has_var_name,
            "trajectory_mentions_assignment": has_assignment,
            "likely_reason": _classify_unresolved_final_var_failure(
                final_var_name=final_var_name,
                iteration_count=iteration_count,
                max_iterations=int(config["max_iterations"]),
                has_var_name=has_var_name,
                has_assignment=has_assignment,
            ),
        },
        "prompt": prompt,
        "root_prompt": root_prompt,
        "trajectory": trajectory,
    }


def _extract_final_var_name(text: str) -> str | None:
    match = re.match(r"^\s*FINAL_VAR\s*\((.*?)\)\s*$", text, re.DOTALL)
    if match is None:
        return None
    name = match.group(1).strip()
    return name or None


def _extract_iteration_count(trajectory: Any | None) -> int | None:
    if not isinstance(trajectory, dict):
        return None
    iterations = trajectory.get("iterations")
    if isinstance(iterations, list):
        return len(iterations)
    return None


def _classify_unresolved_final_var_failure(
    *,
    final_var_name: str | None,
    iteration_count: int | None,
    max_iterations: int,
    has_var_name: bool,
    has_assignment: bool,
) -> str:
    if iteration_count is not None and iteration_count >= max_iterations:
        return "likely_hit_max_iterations_before_final_var_resolution"
    if final_var_name is None:
        return "unknown_final_var_shape"
    if not has_var_name:
        return "model_referenced_final_var_missing_from_logged_trajectory"
    if not has_assignment:
        return "model_referenced_final_var_without_obvious_assignment"
    return "final_var_appears_assigned_but_runtime_returned_unresolved_control_text"


def _unwrap_final_output(text: str) -> str:
    match = re.match(r"^\s*FINAL\s*\((.*)\)\s*$", text, re.DOTALL)
    if match is None:
        return text

    inner = match.group(1).strip()
    if not inner:
        return ""

    try:
        value = ast.literal_eval(inner)
    except Exception:
        quote = inner[0]
        if len(inner) >= 2 and quote in {"\"", "'"} and inner[-1] == quote:
            body = inner[1:-1]
            try:
                return bytes(body, "utf-8").decode("unicode_escape")
            except Exception:
                return body
        return text

    if isinstance(value, str):
        return value
    return text


def _preview_context_payload(context: RLMContextView) -> str:
    return context.to_text(context, limit=_CONTEXT_PREVIEW_CHAR_LIMIT)


def _build_memory_root_prompt(retrieval_question: str) -> str:
    return (
        "Retrieve only older-run details that are missing from the current live context and "
        "materially help the agent's immediate next step.\n"
        "Use the structured `context` object to inspect live messages and stored traces.\n"
        "Infer the next likely action from the top-level task, latest assistant action, and "
        "latest observation/result.\n"
        "Return plain text only.\n"
        "If the live context already contains everything needed, return FINAL(\"\") immediately.\n"
        "Do not restate obvious recent live context unless a small reminder is needed for clarity.\n"
        "Do not mention traces, spans, observability metadata, or that this is a retrieval step.\n"
        "Focus on concrete facts, commands, paths, outputs, and decisions from earlier history.\n\n"
        f"{retrieval_question}"
    )


def _build_memory_message_content(
    *,
    synthesized_memory: str,
    current_messages: list[Message],
    token_budget: int | None,
    max_memory_tokens: int,
    tokenizer_model: str | None,
) -> str | None:
    stripped_memory = synthesized_memory.strip()
    if not stripped_memory:
        return None

    allowed_budget = max_memory_tokens
    if token_budget is not None:
        remaining_budget = max(
            0,
            token_budget - count_message_tokens(current_messages, tokenizer_model),
        )
        allowed_budget = min(allowed_budget, remaining_budget)
    if allowed_budget <= 0:
        return None

    full_content = f"{_MEMORY_MESSAGE_PREFIX}{stripped_memory}"
    fitted_content = _fit_message_content_to_budget(
        role="system",
        content=full_content,
        token_budget=allowed_budget,
        tokenizer_model=tokenizer_model,
    )
    if not fitted_content.startswith(_MEMORY_MESSAGE_PREFIX):
        return None
    if not fitted_content[len(_MEMORY_MESSAGE_PREFIX) :].strip():
        return None
    return fitted_content.rstrip()


def _insert_memory_message(
    *,
    current_messages: list[Message],
    memory_content: str,
) -> list[Message]:
    memory_message = {"role": "system", "content": memory_content}
    if current_messages and current_messages[0]["role"] == "system":
        return [current_messages[0], memory_message, *current_messages[1:]]
    return [memory_message, *current_messages]


def _latest_message_content(messages: list[Message], *, role: str) -> str | None:
    for message in reversed(messages):
        if message["role"] != role:
            continue
        content = message["content"].strip()
        if content:
            return content
    return None


def _top_level_task(*, messages: list[Message], task_hint: str | None) -> str:
    if task_hint and task_hint.strip():
        return task_hint.strip()

    first_user_message: str | None = None
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"].strip()
        if not content:
            continue
        if first_user_message is None:
            first_user_message = content
        if not _looks_like_runtime_output(content):
            return content

    if first_user_message:
        return first_user_message
    return "Continue the active task."


def _latest_explicit_user_request(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        content = message["content"].strip()
        if content and not _looks_like_runtime_output(content):
            return content
    return None


def _latest_runtime_observation(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        content = message["content"].strip()
        if content and _looks_like_runtime_output(content):
            return content
    return None


def _looks_like_runtime_output(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("$ ") or "[exit_code]" in stripped


def _clip_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated {omitted} chars]"


def _fit_message_content_to_budget(
    *,
    role: str,
    content: str,
    token_budget: int | None,
    tokenizer_model: str | None,
) -> str:
    stripped = content.strip()
    if token_budget is None or not stripped:
        return stripped

    if count_message_tokens([{"role": role, "content": stripped}], tokenizer_model) <= token_budget:
        return stripped

    low = 1
    high = len(stripped)
    best = stripped[:1]

    while low <= high:
        mid = (low + high) // 2
        candidate = stripped[:mid].rstrip()
        if not candidate:
            low = mid + 1
            continue

        if (
            count_message_tokens(
                [{"role": role, "content": candidate}],
                tokenizer_model,
            )
            <= token_budget
        ):
            best = candidate
            low = mid + 1
        else:
            high = mid - 1

    return best

def _to_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
