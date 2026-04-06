from __future__ import annotations

from dataclasses import dataclass

from tokentrim.pipeline.requests import PipelineRequest
from tokentrim.transforms.base import Transform
from tokentrim.transforms.rlm.store import MemoryStore
from tokentrim.types.message import Message
from tokentrim.types.state import PipelineState


@dataclass(frozen=True, slots=True)
class RetrieveMemory(Transform):
    """Retrieve prior state and inject it as a system message when available."""

    memory_store: MemoryStore | None = None

    @property
    def name(self) -> str:
        return "rlm"

    def run(self, state: PipelineState, request: PipelineRequest) -> PipelineState:
        messages = state.context
        if self.memory_store is None:
            return state
        if not request.user_id or not request.session_id:
            return state

        retrieved = self.memory_store.retrieve(
            user_id=request.user_id,
            session_id=request.session_id,
        )
        if not retrieved:
            return state

        messages = state.context
        if messages and messages[0]["role"] == "system":
            merged: Message = {
                "role": "system",
                "content": f"{synthesized_memory}\n\n{messages[0]['content']}".strip(),
            }
            return PipelineState(context=[merged, *messages[1:]], tools=state.tools)

        injection: Message = {
            "role": "system",
            "content": synthesized_memory,
        }
        return PipelineState(context=[injection, *messages], tools=state.tools)

    def _validate_configuration(self) -> None:
        if not self.model.strip():
            raise RLMConfigurationError("RetrieveMemory requires a non-empty RLM model name.")
        if not self.backend.strip():
            raise RLMConfigurationError("RetrieveMemory requires a non-empty RLM backend name.")
        if self.trace_limit < 1:
            raise RLMConfigurationError("RetrieveMemory trace_limit must be at least 1.")
        if self.max_depth < 1:
            raise RLMConfigurationError("RetrieveMemory max_depth must be at least 1.")
        if self.max_iterations < 1:
            raise RLMConfigurationError("RetrieveMemory max_iterations must be at least 1.")

    def _generate_memory(self, *, prompt: str, root_prompt: str) -> str:
        rlm_module = _load_rlm_module()
        rlm_cls = getattr(rlm_module, "RLM", None)
        if rlm_cls is None:
            raise RLMConfigurationError(
                "tokentrim[rlm] is installed, but the `rlm.RLM` runtime could not be found."
            )

        try:
            rlm = rlm_cls(
                backend=self.backend,
                backend_kwargs={"model_name": self.model},
                environment="local",
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
                verbose=False,
            )
        except Exception as exc:
            raise RLMExecutionError("Recursive memory synthesis failed.") from exc

        try:
            completion = rlm.completion(prompt, root_prompt=root_prompt)
        except Exception as exc:
            raise RLMExecutionError("Recursive memory synthesis failed.") from exc
        finally:
            close = getattr(rlm, "close", None)
            if callable(close):
                close()

        response = getattr(completion, "response", None)
        if not isinstance(response, str):
            raise RLMExecutionError("Recursive memory synthesis did not return text content.")
        if _looks_like_rlm_control_output(response):
            raise RLMExecutionError(
                "Recursive memory synthesis returned unresolved RLM control text "
                f"({response.strip()!r}) instead of a plain-text memory block."
            )
        return response


def _load_rlm_module() -> ModuleType:
    try:
        return import_module("rlm")
    except ImportError as exc:
        raise RLMConfigurationError(
            "RetrieveMemory requires the optional recursive-memory runtime. "
            "Install it with `pip install \"tokentrim[rlm]\"`."
        ) from exc


def _build_retrieval_question(*, messages: list[Message], task_hint: str | None) -> str:
    if task_hint:
        return f"Current task: {task_hint}"

    for message in reversed(messages):
        content = message["content"].strip()
        if message["role"] == "user" and content:
            return f"Current user request: {content}"

    if messages:
        return "Current conversation state: synthesize only memory relevant to the active context."
    return "Synthesize only memory relevant to the current conversation."


def _looks_like_rlm_control_output(text: str) -> bool:
    return bool(re.match(r"^\s*FINAL(?:_VAR)?\s*\(", text))


def _build_memory_prompt(
    *,
    retrieval_question: str,
    current_messages: list[Message],
    trace_history: str,
) -> str:
    return (
        "You are preparing one short system-memory message for an active agent turn.\n"
        "Read the recent session trace history and the current live context, then write only "
        "the facts, preferences, prior decisions, unresolved work, and constraints that are "
        "relevant to the current task.\n"
        "Do not mention traces, spans, or observability metadata. Return plain text only.\n"
        "If nothing in the stored history is relevant, return an empty string.\n\n"
        f"{retrieval_question}\n\n"
        "Current live context:\n"
        f"{_serialize_messages(current_messages)}\n\n"
        "Recent session trace history:\n"
        f"{trace_history}"
    )


def _serialize_messages(messages: list[Message]) -> str:
    if not messages:
        return "(no live messages)"
    return "\n".join(
        f"{index}. {message['role']}: {message['content']}"
        for index, message in enumerate(messages, start=1)
    )


def _serialize_trace_history(traces: list[TokentrimTraceRecord]) -> str:
    chunks = []
    for index, trace in enumerate(traces, start=1):
        lines = [
            f"Trace {index}",
            f"source: {trace.source}",
            f"workflow: {trace.workflow_name or '(unknown)'}",
        ]
        if trace.started_at is not None:
            lines.append(f"started_at: {trace.started_at}")
        if trace.ended_at is not None:
            lines.append(f"ended_at: {trace.ended_at}")
        if trace.metadata:
            lines.append(f"metadata: {_to_json(trace.metadata)}")
        if trace.spans:
            lines.append("spans:")
            for span_index, span in enumerate(trace.spans, start=1):
                lines.extend(_serialize_span(span_index, span))
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def _serialize_span(index: int, span: TokentrimSpanRecord) -> list[str]:
    lines = [
        f"  - span {index}",
        f"    kind: {span.kind}",
    ]
    if span.name:
        lines.append(f"    name: {span.name}")
    if span.started_at is not None:
        lines.append(f"    started_at: {span.started_at}")
    if span.ended_at is not None:
        lines.append(f"    ended_at: {span.ended_at}")
    if span.metrics:
        lines.append(f"    metrics: {_to_json(span.metrics)}")
    if span.data:
        lines.append(f"    data: {_to_json(span.data)}")
    return lines


def _to_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
