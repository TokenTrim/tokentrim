from __future__ import annotations

import ast
import builtins
import io
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

from tokentrim.core.llm_client import generate_text
from tokentrim.consolidator.errors import ConsolidatorRuntimeError
from tokentrim.tracing.records import TokentrimSpanRecord, TokentrimTraceRecord
from tokentrim.types.message import Message

_BLOCKED_BUILTINS = {"eval", "exec", "compile", "input", "globals", "locals"}
_RESERVED_NAMES = {
    "context",
    "bundle",
    "llm_query",
    "bundle_query",
    "FINAL",
    "FINAL_VAR",
    "SHOW_VARS",
    "peek",
    "to_text",
}
_SAFE_BUILTINS = {
    name: value
    for name, value in builtins.__dict__.items()
    if name not in _BLOCKED_BUILTINS
}
_PREVIEW_CHAR_LIMIT = max(1, int(os.environ.get("TOKENTRIM_CONSOLIDATOR_PREVIEW_CHARS", "2000")))
_TEXT_CHAR_LIMIT = max(1, int(os.environ.get("TOKENTRIM_CONSOLIDATOR_TEXT_CHARS", "12000")))
_GREP_MATCH_LIMIT = max(1, int(os.environ.get("TOKENTRIM_CONSOLIDATOR_GREP_MATCH_LIMIT", "5")))
_EXEC_RESULT_CHAR_LIMIT = max(
    1, int(os.environ.get("TOKENTRIM_CONSOLIDATOR_EXEC_RESULT_CHARS", "4000"))
)
_EXCERPT_CONTEXT = 180

_SYSTEM_PROMPT = (
    "You are operating a local offline consolidator runtime for an active agent. "
    "The offline bundle is available as the Python object `bundle`; it is not pasted into "
    "the prompt. Use ```repl``` blocks to inspect `bundle` before answering. "
    "Inside the REPL, use zero-based indices and these helpers: "
    "`bundle.latest_messages(n)`, `bundle.message_slice(start, end)`, "
    "`bundle.latest_traces(n)`, `bundle.trace(i)`, `bundle.grep(pattern, target=\"all\")`, "
    "`bundle.peek(obj, limit=None)`, `bundle.to_text(obj, limit=None)`, "
    "`llm_query(prompt)`, `bundle_query(query, target=None)`, `SHOW_VARS()`, "
    "`FINAL(...)`, `FINAL_VAR(variable_name)`, and `print()`. "
    "Use `print(bundle.peek(...))` when you need to inspect an object in detail. "
    "When you are done, return either `FINAL(\"...\")` or `FINAL_VAR(variable_name)`."
)
_SUBCALL_SYSTEM_PROMPT = (
    "You are answering a focused offline bundle subquery over the Python object `bundle`. "
    "Use the same REPL tools to inspect the selected sub-bundle and return only the "
    "relevant answer."
)

OperationRecorder = Callable[[dict[str, Any]], None]


def resolve_model_name(*, backend: str, model: str) -> str:
    normalized_backend = backend.strip().lower()
    normalized_model = model.strip()
    if "/" in normalized_model:
        return normalized_model
    return f"{normalized_backend}/{normalized_model}"


@dataclass(frozen=True, slots=True)
class BundleMessageView:
    index: int
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class BundleSpanView:
    trace_index: int
    index: int
    kind: str
    name: str | None
    started_at: str | None
    ended_at: str | None
    error: dict[str, Any] | None
    metrics: dict[str, int] | None
    data: dict[str, Any]

    @classmethod
    def from_record(
        cls,
        *,
        trace_index: int,
        index: int,
        record: TokentrimSpanRecord,
    ) -> BundleSpanView:
        return cls(
            trace_index=trace_index,
            index=index,
            kind=record.kind,
            name=record.name,
            started_at=record.started_at,
            ended_at=record.ended_at,
            error=record.error,
            metrics=record.metrics,
            data=record.data,
        )


@dataclass(frozen=True, slots=True)
class BundleTraceView:
    index: int
    trace_id: str
    source: str
    workflow_name: str
    started_at: str | None
    ended_at: str | None
    metadata: dict[str, Any] | None
    spans: tuple[BundleSpanView, ...]

    @classmethod
    def from_record(cls, *, index: int, record: TokentrimTraceRecord) -> BundleTraceView:
        spans = tuple(
            BundleSpanView.from_record(trace_index=index, index=span_index, record=span)
            for span_index, span in enumerate(record.spans)
        )
        return cls(
            index=index,
            trace_id=record.trace_id,
            source=record.source,
            workflow_name=record.workflow_name,
            started_at=record.started_at,
            ended_at=record.ended_at,
            metadata=record.metadata,
            spans=spans,
        )


@dataclass(frozen=True, slots=True)
class BundleSearchMatch:
    source: str
    target: str
    pattern: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class OfflineBundleView:
    messages: tuple[BundleMessageView, ...] = ()
    traces: tuple[BundleTraceView, ...] = ()
    label: str = "run_history"
    operation_recorder: OperationRecorder | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_history(
        cls,
        *,
        messages: Sequence[Message],
        traces: Sequence[TokentrimTraceRecord],
        label: str = "run_history",
    ) -> OfflineBundleView:
        message_views = tuple(
            BundleMessageView(
                index=index,
                role=str(message["role"]),
                content=str(message["content"]),
            )
            for index, message in enumerate(messages)
        )
        trace_views = tuple(
            BundleTraceView.from_record(index=index, record=trace)
            for index, trace in enumerate(traces)
        )
        return cls(messages=message_views, traces=trace_views, label=label)

    @classmethod
    def from_text(cls, text: str, *, label: str = "text_bundle") -> OfflineBundleView:
        stripped = text.strip()
        if not stripped:
            return cls(label=label)
        return cls(
            messages=(BundleMessageView(index=0, role="user", content=stripped),),
            traces=(),
            label=label,
        )

    def with_operation_recorder(self, recorder: OperationRecorder | None) -> OfflineBundleView:
        return replace(self, operation_recorder=recorder)

    def latest_messages(self, n: int) -> tuple[BundleMessageView, ...]:
        count = max(0, int(n))
        result = self.messages[-count:] if count else ()
        self._record_browser_call(
            name="latest_messages",
            arguments={"n": count},
            result=_summarize_object(result),
        )
        return result

    def message_slice(self, start: int | None, end: int | None) -> tuple[BundleMessageView, ...]:
        result = self.messages[slice(start, end)]
        self._record_browser_call(
            name="message_slice",
            arguments={"start": start, "end": end},
            result=_summarize_object(result),
        )
        return result

    def latest_traces(self, n: int) -> tuple[BundleTraceView, ...]:
        count = max(0, int(n))
        result = self.traces[-count:] if count else ()
        self._record_browser_call(
            name="latest_traces",
            arguments={"n": count},
            result=_summarize_object(result),
        )
        return result

    def trace(self, index: int) -> BundleTraceView:
        result = self.traces[index]
        self._record_browser_call(
            name="trace",
            arguments={"index": index},
            result=_summarize_object(result),
        )
        return result

    def grep(self, pattern: str, target: str = "all") -> tuple[BundleSearchMatch, ...]:
        compiled = _compile_pattern(pattern)
        normalized_target = target.strip().lower()
        if normalized_target not in {"all", "messages", "traces"}:
            normalized_target = "all"

        matches: list[BundleSearchMatch] = []
        if normalized_target in {"all", "messages"}:
            for message in self.messages:
                _append_matches(
                    matches=matches,
                    pattern=str(pattern),
                    compiled=compiled,
                    target="messages",
                    source=f"message[{message.index}]",
                    text=f"{message.role}: {message.content}",
                )
                if len(matches) >= _GREP_MATCH_LIMIT:
                    break

        if normalized_target in {"all", "traces"} and len(matches) < _GREP_MATCH_LIMIT:
            for trace in self.traces:
                _append_matches(
                    matches=matches,
                    pattern=str(pattern),
                    compiled=compiled,
                    target="traces",
                    source=f"trace[{trace.index}]",
                    text=_trace_summary_text(trace),
                )
                if len(matches) >= _GREP_MATCH_LIMIT:
                    break
                for span in trace.spans:
                    _append_matches(
                        matches=matches,
                        pattern=str(pattern),
                        compiled=compiled,
                        target="traces",
                        source=f"trace[{trace.index}].span[{span.index}]",
                        text=_span_text(span),
                    )
                    if len(matches) >= _GREP_MATCH_LIMIT:
                        break
                if len(matches) >= _GREP_MATCH_LIMIT:
                    break

        result = tuple(matches)
        self._record_browser_call(
            name="grep",
            arguments={"pattern": str(pattern), "target": normalized_target},
            result=_summarize_object(result),
        )
        return result

    def peek(self, obj: Any, limit: int | None = None) -> str:
        resolved_limit = _PREVIEW_CHAR_LIMIT if limit is None else max(1, int(limit))
        preview = _object_to_text(obj if obj is not None else self, limit=resolved_limit)
        self._record_browser_call(
            name="peek",
            arguments={"target": _summarize_object(obj), "limit": resolved_limit},
            result=_clip_text(preview, 400),
        )
        return preview

    def to_text(self, obj: Any, limit: int | None = None) -> str:
        resolved_limit = _TEXT_CHAR_LIMIT if limit is None else max(1, int(limit))
        text = _object_to_text(obj if obj is not None else self, limit=resolved_limit)
        self._record_browser_call(
            name="to_text",
            arguments={"target": _summarize_object(obj), "limit": resolved_limit},
            result=_clip_text(text, 400),
        )
        return text

    def as_subcontext(self, target: Any | None = None) -> OfflineBundleView:
        if target is None:
            return self.with_operation_recorder(None)
        return _coerce_subbundle(target)

    def _record_browser_call(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        if self.operation_recorder is None:
            return
        self.operation_recorder(
            {
                "type": "browser_call",
                "name": name,
                "arguments": arguments,
                "result": result,
            }
        )


@dataclass(slots=True)
class LocalConsolidatorRuntime:
    model: str
    backend: str
    max_iterations: int
    tokenizer_model: str | None = None
    max_depth: int = 1
    max_subcalls: int = 4
    subcall_model: str | None = None
    context_variable_name: str = "bundle"
    context_aliases: tuple[str, ...] = ("context",)
    current_depth: int = field(default=0, repr=False)
    resolved_model: str = field(init=False)
    resolved_subcall_model: str = field(init=False)
    trajectory: dict[str, Any] = field(init=False)
    _globals: dict[str, Any] = field(init=False)
    _locals: dict[str, Any] = field(init=False)
    _last_final_answer: str | None = field(init=False, default=None)
    _bundle_value: OfflineBundleView = field(init=False)
    _system_prompt_value: str = field(init=False, default=_SYSTEM_PROMPT)
    _current_operations: list[dict[str, Any]] = field(init=False, default_factory=list)
    _subcall_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.resolved_model = resolve_model_name(backend=self.backend, model=self.model)
        self.resolved_subcall_model = resolve_model_name(
            backend=self.backend,
            model=self.subcall_model or self.model,
        )
        self.trajectory = {
            "depth": self.current_depth,
            "iterations": [],
            "subcalls_used": 0,
        }
        self._globals = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
            "llm_query": self._llm_query,
            "bundle_query": self._bundle_query,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            "SHOW_VARS": self._show_vars,
            "peek": self._peek,
            "to_text": self._to_text,
        }
        self._locals = {}
        self._last_final_answer = None
        self._bundle_value = OfflineBundleView()

    def run(
        self,
        bundle_input: Any,
        root_prompt: str | None = None,
        *,
        system_prompt: str | None = None,
    ) -> str:
        self.trajectory = {
            "depth": self.current_depth,
            "iterations": [],
            "subcalls_used": 0,
        }
        self._system_prompt_value = system_prompt or _SYSTEM_PROMPT
        self._bundle_value = _coerce_bundle_input(bundle_input).with_operation_recorder(
            self._record_operation
        )
        self._locals = {}
        self._bind_bundle_variables()
        self._last_final_answer = None
        self._subcall_count = 0
        message_history: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt_value},
            {
                "role": "user",
                "content": _build_user_prompt(
                    root_prompt=root_prompt,
                    iteration=0,
                    bundle_variable_name=self.context_variable_name,
                ),
            },
        ]

        for iteration in range(self.max_iterations):
            response = self._model_completion(message_history)
            code_blocks = _find_code_blocks(response)
            block_results = [self._execute_code(code_block) for code_block in code_blocks]
            final_answer = self._consume_final_answer()
            if final_answer is None:
                final_answer = self._extract_final_answer(response)

            self.trajectory["iterations"].append(
                {
                    "iteration": iteration + 1,
                    "prompt": [dict(message) for message in message_history],
                    "response": response,
                    "code_blocks": block_results,
                    "final_answer": final_answer,
                }
            )
            self.trajectory["subcalls_used"] = self._subcall_count

            if final_answer is not None:
                return final_answer

            message_history.append({"role": "assistant", "content": response})
            for block_result in block_results:
                message_history.append(
                    {
                        "role": "user",
                        "content": _format_execution_result(block_result),
                    }
                )
            message_history.append(
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        root_prompt=root_prompt,
                        iteration=iteration + 1,
                        bundle_variable_name=self.context_variable_name,
                    ),
                }
            )

        self.trajectory["subcalls_used"] = self._subcall_count
        raise ConsolidatorRuntimeError(
            "consolidator runtime hit max iterations without producing a final answer."
        )

    def _model_completion(self, messages: list[dict[str, str]]) -> str:
        return generate_text(
            model=self.resolved_model,
            messages=messages,
            temperature=0.0,
        )

    def _llm_query(self, prompt: str) -> str:
        try:
            response = generate_text(
                model=self.resolved_model,
                messages=[{"role": "user", "content": str(prompt)}],
                temperature=0.0,
            )
        except Exception as exc:
            response = f"Error: {exc}"
        self._record_operation(
            {
                "type": "llm_query",
                "prompt": _clip_text(str(prompt), 400),
                "response": _clip_text(response, 400),
            }
        )
        return response

    def _bundle_query(self, query: str, target: Any | None = None) -> str:
        if self.current_depth >= self.max_depth:
            error = "Error: bundle_query depth limit reached."
            self._record_operation(
                {
                    "type": "subcall",
                    "query": _clip_text(str(query), 400),
                    "target": _summarize_object(target),
                    "status": "depth_limit",
                    "result": error,
                }
            )
            return error
        if self._subcall_count >= self.max_subcalls:
            error = "Error: bundle_query subcall limit reached."
            self._record_operation(
                {
                    "type": "subcall",
                    "query": _clip_text(str(query), 400),
                    "target": _summarize_object(target),
                    "status": "subcall_limit",
                    "result": error,
                }
            )
            return error

        self._subcall_count += 1
        subbundle = self._bundle_value.as_subcontext(target)
        child_runtime = LocalConsolidatorRuntime(
            model=self.subcall_model or self.model,
            backend=self.backend,
            max_iterations=self.max_iterations,
            tokenizer_model=self.tokenizer_model,
            max_depth=self.max_depth,
            max_subcalls=self.max_subcalls,
            subcall_model=self.subcall_model,
            context_variable_name=self.context_variable_name,
            context_aliases=self.context_aliases,
            current_depth=self.current_depth + 1,
        )
        try:
            result = child_runtime.run(
                subbundle,
                root_prompt=f"Subquery:\n{str(query).strip()}",
                system_prompt=_SUBCALL_SYSTEM_PROMPT,
            )
            status = "ok"
        except Exception as exc:
            result = f"Error: {type(exc).__name__}: {exc}"
            status = "runtime_error"

        self._record_operation(
            {
                "type": "subcall",
                "query": _clip_text(str(query), 400),
                "target": _summarize_object(subbundle),
                "status": status,
                "result": _clip_text(result, 400),
                "trajectory": child_runtime.trajectory,
            }
        )
        return result

    def _peek(self, obj: Any, limit: int | None = None) -> str:
        return self._bundle_value.peek(obj, limit=limit)

    def _to_text(self, obj: Any, limit: int | None = None) -> str:
        return self._bundle_value.to_text(obj, limit=limit)

    def _final(self, value: Any) -> str:
        answer = str(value)
        self._last_final_answer = answer
        return answer

    def _final_var(self, variable_name: str | Any) -> str:
        if not isinstance(variable_name, str):
            answer = str(variable_name)
            self._last_final_answer = answer
            return answer

        normalized_name = variable_name.strip().strip("\"'")
        if normalized_name in self._locals:
            answer = str(self._locals[normalized_name])
            self._last_final_answer = answer
            return answer

        available = sorted(
            name
            for name in self._locals
            if not name.startswith("_") and name not in self._hidden_local_names()
        )
        if available:
            return (
                f"Error: Variable '{normalized_name}' not found. "
                f"Available variables: {available}. "
                "You must create and assign a variable before calling FINAL_VAR on it."
            )
        return (
            f"Error: Variable '{normalized_name}' not found. "
            "No variables have been created yet. "
            "You must create and assign a variable before calling FINAL_VAR on it."
        )

    def _show_vars(self) -> str:
        available = sorted(
            name
            for name in self._locals
            if not name.startswith("_") and name not in self._hidden_local_names()
        )
        if not available:
            return "No variables created yet."
        return f"Available variables: {available}"

    def _execute_code(self, code: str) -> dict[str, Any]:
        self._last_final_answer = None
        self._current_operations = []
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            combined = {**self._globals, **self._locals}
            exec(code, combined, combined)
            for key, value in combined.items():
                if key in _RESERVED_NAMES or key in self._hidden_local_names() or key.startswith("_"):
                    continue
                self._locals[key] = value
        except Exception:
            traceback.print_exc(file=stderr_buffer)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self._restore_helpers()

        return {
            "code": code,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "final_answer": self._last_final_answer,
            "locals": sorted(
                name
                for name in self._locals
                if not name.startswith("_") and name not in self._hidden_local_names()
            ),
            "operations": list(self._current_operations),
        }

    def _restore_helpers(self) -> None:
        self._globals["llm_query"] = self._llm_query
        self._globals["bundle_query"] = self._bundle_query
        self._globals["FINAL"] = self._final
        self._globals["FINAL_VAR"] = self._final_var
        self._globals["SHOW_VARS"] = self._show_vars
        self._globals["peek"] = self._peek
        self._globals["to_text"] = self._to_text
        self._bind_bundle_variables()

    def _consume_final_answer(self) -> str | None:
        if self._last_final_answer is None:
            return None
        answer = self._last_final_answer
        self._last_final_answer = None
        return answer

    def _extract_final_answer(self, text: str) -> str | None:
        final_var_match = re.search(r"^\s*FINAL_VAR\((.*?)\)", text, re.MULTILINE | re.DOTALL)
        if final_var_match is not None:
            variable_name = final_var_match.group(1).strip().strip("\"'")
            if variable_name in self._locals:
                return str(self._locals[variable_name])
            return None

        final_match = re.search(r"^\s*FINAL\((.*)\)\s*$", text, re.MULTILINE | re.DOTALL)
        if final_match is None:
            direct_json = _extract_json_object(text)
            return direct_json
        return _unwrap_final_output(final_match.group(1).strip())

    def _record_operation(self, entry: dict[str, Any]) -> None:
        self._current_operations.append(entry)

    def _bind_bundle_variables(self) -> None:
        for name in (self.context_variable_name, *self.context_aliases):
            self._locals[name] = self._bundle_value

    def _hidden_local_names(self) -> set[str]:
        return {self.context_variable_name, *self.context_aliases}


def _coerce_bundle_input(bundle_input: Any) -> OfflineBundleView:
    if isinstance(bundle_input, OfflineBundleView):
        return bundle_input
    if isinstance(bundle_input, str):
        return OfflineBundleView.from_text(bundle_input)
    if isinstance(bundle_input, Sequence) and not isinstance(
        bundle_input, (bytes, bytearray, str)
    ):
        if all(_looks_like_message(item) for item in bundle_input):
            return OfflineBundleView.from_history(messages=bundle_input, traces=())
    return OfflineBundleView.from_text(_fallback_to_text(bundle_input))


def _coerce_subbundle(target: Any) -> OfflineBundleView:
    if isinstance(target, OfflineBundleView):
        return target.with_operation_recorder(None)
    if isinstance(target, BundleMessageView):
        return OfflineBundleView(messages=(target,), label=f"message[{target.index}]")
    if isinstance(target, BundleTraceView):
        return OfflineBundleView(traces=(target,), label=f"trace[{target.index}]")
    if isinstance(target, BundleSpanView):
        return OfflineBundleView.from_text(_span_text(target), label=f"trace[{target.trace_index}].span[{target.index}]")
    if isinstance(target, BundleSearchMatch):
        return OfflineBundleView.from_text(_search_match_text(target), label=target.source)
    if isinstance(target, str):
        return OfflineBundleView.from_text(target)
    if isinstance(target, Sequence) and not isinstance(target, (bytes, bytearray, str)):
        items = list(target)
        if not items:
            return OfflineBundleView(label="empty_selection")
        if all(isinstance(item, BundleMessageView) for item in items):
            return OfflineBundleView(messages=tuple(items), label="message_selection")
        if all(isinstance(item, BundleTraceView) for item in items):
            return OfflineBundleView(traces=tuple(items), label="trace_selection")
        synthetic_messages = tuple(
            BundleMessageView(
                index=index,
                role="user",
                content=_object_to_text(item, limit=_TEXT_CHAR_LIMIT),
            )
            for index, item in enumerate(items)
        )
        return OfflineBundleView(messages=synthetic_messages, label="selection")
    return OfflineBundleView.from_text(_fallback_to_text(target), label=type(target).__name__)


def _build_user_prompt(
    *,
    root_prompt: str | None,
    iteration: int,
    bundle_variable_name: str,
) -> str:
    task_line = root_prompt if root_prompt is not None else "Retrieve the relevant answer from `bundle`."
    if iteration == 0:
        return (
            f"{task_line}\n\n"
            f"The offline bundle is loaded in the REPL variable `{bundle_variable_name}` as a structured object. "
            "Inspect it with a `repl` block before answering."
        )
    return (
        f"{task_line}\n\n"
        f"Continue inspecting `{bundle_variable_name}` if needed. Use another `repl` block, or return "
        "the final JSON object directly, or use FINAL(...)/FINAL_VAR(...), when you have the answer. "
        "Prefer finishing now over unnecessary extra browsing."
    )


def _find_code_blocks(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"```repl\s*\n(.*?)\n```", text, re.DOTALL)]


def _format_execution_result(result: dict[str, Any]) -> str:
    parts = [
        "Code executed:",
        f"```python\n{result['code']}\n```",
    ]
    stdout = _clip_text(result["stdout"].strip(), _EXEC_RESULT_CHAR_LIMIT)
    stderr = _clip_text(result["stderr"].strip(), _EXEC_RESULT_CHAR_LIMIT)
    if stdout:
        parts.append(f"STDOUT:\n{stdout}")
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    operations = result["operations"]
    if operations:
        parts.append(
            "Operations:\n" + "\n".join(_format_operation(operation) for operation in operations)
        )
    locals_list = result["locals"]
    if locals_list:
        parts.append(f"Variables: {locals_list}")
    if not stdout and not stderr and not locals_list and not operations:
        parts.append("No output.")
    return "\n\n".join(parts)


def _format_operation(operation: dict[str, Any]) -> str:
    op_type = operation.get("type")
    if op_type == "browser_call":
        return (
            f"- {operation['name']}({operation['arguments']}) -> "
            f"{operation['result']}"
        )
    if op_type == "subcall":
        return (
            f"- bundle_query[{operation['status']}] query={operation['query']!r} "
            f"target={operation['target']} result={operation['result']!r}"
        )
    if op_type == "llm_query":
        return (
            f"- llm_query(prompt={operation['prompt']!r}) -> "
            f"{operation['response']!r}"
        )
    return f"- {operation}"


def _unwrap_final_output(text: str) -> str:
    if not text:
        return ""
    try:
        value = ast.literal_eval(text)
    except Exception:
        quote = text[0]
        if len(text) >= 2 and quote in {"\"", "'"} and text[-1] == quote:
            body = text[1:-1]
            try:
                return bytes(body, "utf-8").decode("unicode_escape")
            except Exception:
                return body
        return text
    if isinstance(value, str):
        return value
    return text


def _extract_json_object(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1).strip() if fenced is not None else stripped
    if not candidate.startswith("{") or not candidate.endswith("}"):
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return candidate


def _looks_like_message(item: Any) -> bool:
    return isinstance(item, dict) and "role" in item and "content" in item


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE)


def _append_matches(
    *,
    matches: list[BundleSearchMatch],
    pattern: str,
    compiled: re.Pattern[str],
    target: str,
    source: str,
    text: str,
) -> None:
    match = compiled.search(text)
    if match is None:
        return
    matches.append(
        BundleSearchMatch(
            source=source,
            target=target,
            pattern=pattern,
            excerpt=_excerpt(text, match.start(), match.end()),
        )
    )


def _excerpt(text: str, start: int, end: int) -> str:
    left = max(0, start - _EXCERPT_CONTEXT)
    right = min(len(text), end + _EXCERPT_CONTEXT)
    excerpt = text[left:right].strip()
    if left > 0:
        excerpt = "..." + excerpt
    if right < len(text):
        excerpt = excerpt + "..."
    return excerpt


def _object_to_text(obj: Any, *, limit: int) -> str:
    rendered = _render_object(obj)
    return _clip_text(rendered, limit)


def _render_object(obj: Any) -> str:
    if isinstance(obj, OfflineBundleView):
        return _bundle_text(obj)
    if isinstance(obj, BundleMessageView):
        return _message_text(obj)
    if isinstance(obj, BundleTraceView):
        return _trace_text(obj)
    if isinstance(obj, BundleSpanView):
        return _span_text(obj)
    if isinstance(obj, BundleSearchMatch):
        return _search_match_text(obj)
    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray, str)):
        if not obj:
            return "(empty selection)"
        return "\n\n".join(_render_object(item) for item in obj)
    return _fallback_to_text(obj)


def _bundle_text(bundle: OfflineBundleView) -> str:
    lines = [
        f"bundle: {bundle.label}",
        f"message_count: {len(bundle.messages)}",
        f"trace_count: {len(bundle.traces)}",
    ]
    if bundle.messages:
        lines.append("live_messages:")
        lines.extend(_indent_lines(_message_text(message) for message in bundle.messages))
    if bundle.traces:
        lines.append("stored_traces:")
        lines.extend(_indent_lines(_trace_text(trace) for trace in bundle.traces))
    return "\n".join(lines)


def _message_text(message: BundleMessageView) -> str:
    return f"message[{message.index}] {message.role}: {message.content}"


def _trace_summary_text(trace: BundleTraceView) -> str:
    parts = [
        f"workflow={trace.workflow_name}",
        f"source={trace.source}",
    ]
    if trace.metadata:
        parts.append(f"metadata={_to_json(trace.metadata)}")
    return " ".join(parts)


def _trace_text(trace: BundleTraceView) -> str:
    lines = [
        f"trace[{trace.index}]",
        f"trace_id: {trace.trace_id}",
        f"source: {trace.source}",
        f"workflow: {trace.workflow_name}",
    ]
    if trace.started_at is not None:
        lines.append(f"started_at: {trace.started_at}")
    if trace.ended_at is not None:
        lines.append(f"ended_at: {trace.ended_at}")
    if trace.metadata:
        lines.append(f"metadata: {_to_json(trace.metadata)}")
    if trace.spans:
        lines.append("spans:")
        for span in trace.spans:
            lines.extend(_indent_lines([_span_text(span)]))
    return "\n".join(lines)


def _span_text(span: BundleSpanView) -> str:
    lines = [
        f"span[{span.index}]",
        f"kind: {span.kind}",
    ]
    if span.name:
        lines.append(f"name: {span.name}")
    if span.started_at is not None:
        lines.append(f"started_at: {span.started_at}")
    if span.ended_at is not None:
        lines.append(f"ended_at: {span.ended_at}")
    if span.metrics:
        lines.append(f"metrics: {_to_json(span.metrics)}")
    if span.error:
        lines.append(f"error: {_to_json(span.error)}")
    if span.data:
        lines.append(f"data: {_to_json(span.data)}")
    return "\n".join(lines)


def _search_match_text(match: BundleSearchMatch) -> str:
    return (
        f"search_match source={match.source} target={match.target} "
        f"pattern={match.pattern!r}\n{match.excerpt}"
    )


def _fallback_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, indent=2, default=str)
    except Exception:
        return repr(value)


def _summarize_object(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, OfflineBundleView):
        return {
            "type": "bundle",
            "label": obj.label,
            "message_count": len(obj.messages),
            "trace_count": len(obj.traces),
        }
    if isinstance(obj, BundleMessageView):
        return {
            "type": "message",
            "label": f"message[{obj.index}]",
            "role": obj.role,
        }
    if isinstance(obj, BundleTraceView):
        return {
            "type": "trace",
            "label": f"trace[{obj.index}]",
            "workflow": obj.workflow_name,
            "span_count": len(obj.spans),
        }
    if isinstance(obj, BundleSpanView):
        return {
            "type": "span",
            "label": f"trace[{obj.trace_index}].span[{obj.index}]",
            "kind": obj.kind,
        }
    if isinstance(obj, BundleSearchMatch):
        return {
            "type": "search_match",
            "source": obj.source,
            "target": obj.target,
            "excerpt": _clip_text(obj.excerpt, 160),
        }
    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray, str)):
        preview = [_summarize_object(item) for item in list(obj)[:4]]
        return {"type": "sequence", "count": len(obj), "items": preview}
    if isinstance(obj, str):
        return _clip_text(obj, 200)
    return _clip_text(repr(obj), 200)


def _indent_lines(blocks: Iterable[str]) -> list[str]:
    lines: list[str] = []
    for block in blocks:
        for line in str(block).splitlines():
            lines.append(f"  {line}")
    return lines


def _clip_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated {omitted} chars]"


def _to_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
