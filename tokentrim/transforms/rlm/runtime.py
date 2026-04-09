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
from tokentrim.transforms.rlm.error import RLMExecutionError
from tokentrim.tracing.records import TokentrimSpanRecord, TokentrimTraceRecord
from tokentrim.types.message import Message

_BLOCKED_BUILTINS = {"eval", "exec", "compile", "input", "globals", "locals"}
_RESERVED_NAMES = {
    "context",
    "llm_query",
    "rlm_query",
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
_PREVIEW_CHAR_LIMIT = max(1, int(os.environ.get("TOKENTRIM_RLM_PREVIEW_CHARS", "2000")))
_TEXT_CHAR_LIMIT = max(1, int(os.environ.get("TOKENTRIM_RLM_TEXT_CHARS", "12000")))
_GREP_MATCH_LIMIT = max(1, int(os.environ.get("TOKENTRIM_RLM_GREP_MATCH_LIMIT", "5")))
_EXEC_RESULT_CHAR_LIMIT = max(
    1, int(os.environ.get("TOKENTRIM_RLM_EXEC_RESULT_CHARS", "4000"))
)
_EXCERPT_CONTEXT = 180

_SYSTEM_PROMPT = (
    "You are operating a local recursive language memory runtime for an active agent. "
    "The run history is available as the Python object `context`; it is not pasted into "
    "the prompt. Use ```repl``` blocks to inspect `context` before answering. "
    "Inside the REPL, use zero-based indices and these helpers: "
    "`context.latest_messages(n)`, `context.message_slice(start, end)`, "
    "`context.latest_traces(n)`, `context.trace(i)`, `context.grep(pattern, target=\"all\")`, "
    "`context.peek(obj, limit=None)`, `context.to_text(obj, limit=None)`, "
    "`llm_query(prompt)`, `rlm_query(query, target=None)`, `SHOW_VARS()`, "
    "`FINAL(...)`, `FINAL_VAR(variable_name)`, and `print()`. "
    "Use `print(context.peek(...))` when you need to inspect an object in detail. "
    "When you are done, return either `FINAL(\"...\")` or `FINAL_VAR(variable_name)`."
)
_SUBCALL_SYSTEM_PROMPT = (
    "You are answering a focused RLM subquery over the Python object `context`. "
    "Use the same REPL tools to inspect the selected subcontext and return only the "
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
class RLMMessageView:
    index: int
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class RLMSpanView:
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
    ) -> RLMSpanView:
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
class RLMTraceView:
    index: int
    trace_id: str
    source: str
    workflow_name: str
    started_at: str | None
    ended_at: str | None
    metadata: dict[str, Any] | None
    spans: tuple[RLMSpanView, ...]

    @classmethod
    def from_record(cls, *, index: int, record: TokentrimTraceRecord) -> RLMTraceView:
        spans = tuple(
            RLMSpanView.from_record(trace_index=index, index=span_index, record=span)
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
class RLMSearchMatch:
    source: str
    target: str
    pattern: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class RLMContextView:
    messages: tuple[RLMMessageView, ...] = ()
    traces: tuple[RLMTraceView, ...] = ()
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
    ) -> RLMContextView:
        message_views = tuple(
            RLMMessageView(
                index=index,
                role=str(message["role"]),
                content=str(message["content"]),
            )
            for index, message in enumerate(messages)
        )
        trace_views = tuple(
            RLMTraceView.from_record(index=index, record=trace)
            for index, trace in enumerate(traces)
        )
        return cls(messages=message_views, traces=trace_views, label=label)

    @classmethod
    def from_text(cls, text: str, *, label: str = "text_context") -> RLMContextView:
        stripped = text.strip()
        if not stripped:
            return cls(label=label)
        return cls(
            messages=(RLMMessageView(index=0, role="user", content=stripped),),
            traces=(),
            label=label,
        )

    def with_operation_recorder(self, recorder: OperationRecorder | None) -> RLMContextView:
        return replace(self, operation_recorder=recorder)

    def latest_messages(self, n: int) -> tuple[RLMMessageView, ...]:
        count = max(0, int(n))
        result = self.messages[-count:] if count else ()
        self._record_browser_call(
            name="latest_messages",
            arguments={"n": count},
            result=_summarize_object(result),
        )
        return result

    def message_slice(self, start: int | None, end: int | None) -> tuple[RLMMessageView, ...]:
        result = self.messages[slice(start, end)]
        self._record_browser_call(
            name="message_slice",
            arguments={"start": start, "end": end},
            result=_summarize_object(result),
        )
        return result

    def latest_traces(self, n: int) -> tuple[RLMTraceView, ...]:
        count = max(0, int(n))
        result = self.traces[-count:] if count else ()
        self._record_browser_call(
            name="latest_traces",
            arguments={"n": count},
            result=_summarize_object(result),
        )
        return result

    def trace(self, index: int) -> RLMTraceView:
        result = self.traces[index]
        self._record_browser_call(
            name="trace",
            arguments={"index": index},
            result=_summarize_object(result),
        )
        return result

    def grep(self, pattern: str, target: str = "all") -> tuple[RLMSearchMatch, ...]:
        compiled = _compile_pattern(pattern)
        normalized_target = target.strip().lower()
        if normalized_target not in {"all", "messages", "traces"}:
            normalized_target = "all"

        matches: list[RLMSearchMatch] = []
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

    def as_subcontext(self, target: Any | None = None) -> RLMContextView:
        if target is None:
            return self.with_operation_recorder(None)
        return _coerce_subcontext(target)

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
class LocalRLMRuntime:
    model: str
    backend: str
    max_iterations: int
    tokenizer_model: str | None = None
    max_depth: int = 1
    max_subcalls: int = 4
    subcall_model: str | None = None
    current_depth: int = field(default=0, repr=False)
    resolved_model: str = field(init=False)
    resolved_subcall_model: str = field(init=False)
    trajectory: dict[str, Any] = field(init=False)
    _globals: dict[str, Any] = field(init=False)
    _locals: dict[str, Any] = field(init=False)
    _last_final_answer: str | None = field(init=False, default=None)
    _context_value: RLMContextView = field(init=False)
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
            "rlm_query": self._rlm_query,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            "SHOW_VARS": self._show_vars,
            "peek": self._peek,
            "to_text": self._to_text,
        }
        self._locals = {}
        self._last_final_answer = None
        self._context_value = RLMContextView()

    def run(
        self,
        context_prompt: Any,
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
        self._context_value = _coerce_context_prompt(context_prompt).with_operation_recorder(
            self._record_operation
        )
        self._locals = {"context": self._context_value}
        self._last_final_answer = None
        self._subcall_count = 0
        message_history: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt_value},
            {"role": "user", "content": _build_user_prompt(root_prompt=root_prompt, iteration=0)},
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
                    ),
                }
            )

        self.trajectory["subcalls_used"] = self._subcall_count
        raise RLMExecutionError(
            "RLM runtime hit max iterations without producing a final answer."
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

    def _rlm_query(self, query: str, target: Any | None = None) -> str:
        if self.current_depth >= self.max_depth:
            error = "Error: rlm_query depth limit reached."
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
            error = "Error: rlm_query subcall limit reached."
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
        subcontext = self._context_value.as_subcontext(target)
        child_runtime = LocalRLMRuntime(
            model=self.subcall_model or self.model,
            backend=self.backend,
            max_iterations=self.max_iterations,
            tokenizer_model=self.tokenizer_model,
            max_depth=self.max_depth,
            max_subcalls=self.max_subcalls,
            subcall_model=self.subcall_model,
            current_depth=self.current_depth + 1,
        )
        try:
            result = child_runtime.run(
                subcontext,
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
                "target": _summarize_object(subcontext),
                "status": status,
                "result": _clip_text(result, 400),
                "trajectory": child_runtime.trajectory,
            }
        )
        return result

    def _peek(self, obj: Any, limit: int | None = None) -> str:
        return self._context_value.peek(obj, limit=limit)

    def _to_text(self, obj: Any, limit: int | None = None) -> str:
        return self._context_value.to_text(obj, limit=limit)

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
            name for name in self._locals if not name.startswith("_") and name != "context"
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
            name for name in self._locals if not name.startswith("_") and name != "context"
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
                if key in _RESERVED_NAMES or key.startswith("_"):
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
                name for name in self._locals if not name.startswith("_") and name != "context"
            ),
            "operations": list(self._current_operations),
        }

    def _restore_helpers(self) -> None:
        self._globals["llm_query"] = self._llm_query
        self._globals["rlm_query"] = self._rlm_query
        self._globals["FINAL"] = self._final
        self._globals["FINAL_VAR"] = self._final_var
        self._globals["SHOW_VARS"] = self._show_vars
        self._globals["peek"] = self._peek
        self._globals["to_text"] = self._to_text
        self._locals["context"] = self._context_value

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
            return None
        return _unwrap_final_output(final_match.group(1).strip())

    def _record_operation(self, entry: dict[str, Any]) -> None:
        self._current_operations.append(entry)


def _coerce_context_prompt(context_prompt: Any) -> RLMContextView:
    if isinstance(context_prompt, RLMContextView):
        return context_prompt
    if isinstance(context_prompt, str):
        return RLMContextView.from_text(context_prompt)
    if isinstance(context_prompt, Sequence) and not isinstance(
        context_prompt, (bytes, bytearray, str)
    ):
        if all(_looks_like_message(item) for item in context_prompt):
            return RLMContextView.from_history(messages=context_prompt, traces=())
    return RLMContextView.from_text(_fallback_to_text(context_prompt))


def _coerce_subcontext(target: Any) -> RLMContextView:
    if isinstance(target, RLMContextView):
        return target.with_operation_recorder(None)
    if isinstance(target, RLMMessageView):
        return RLMContextView(messages=(target,), label=f"message[{target.index}]")
    if isinstance(target, RLMTraceView):
        return RLMContextView(traces=(target,), label=f"trace[{target.index}]")
    if isinstance(target, RLMSpanView):
        return RLMContextView.from_text(_span_text(target), label=f"trace[{target.trace_index}].span[{target.index}]")
    if isinstance(target, RLMSearchMatch):
        return RLMContextView.from_text(_search_match_text(target), label=target.source)
    if isinstance(target, str):
        return RLMContextView.from_text(target)
    if isinstance(target, Sequence) and not isinstance(target, (bytes, bytearray, str)):
        items = list(target)
        if not items:
            return RLMContextView(label="empty_selection")
        if all(isinstance(item, RLMMessageView) for item in items):
            return RLMContextView(messages=tuple(items), label="message_selection")
        if all(isinstance(item, RLMTraceView) for item in items):
            return RLMContextView(traces=tuple(items), label="trace_selection")
        synthetic_messages = tuple(
            RLMMessageView(
                index=index,
                role="user",
                content=_object_to_text(item, limit=_TEXT_CHAR_LIMIT),
            )
            for index, item in enumerate(items)
        )
        return RLMContextView(messages=synthetic_messages, label="selection")
    return RLMContextView.from_text(_fallback_to_text(target), label=type(target).__name__)


def _build_user_prompt(*, root_prompt: str | None, iteration: int) -> str:
    task_line = root_prompt if root_prompt is not None else "Retrieve the relevant answer from `context`."
    if iteration == 0:
        return (
            f"{task_line}\n\n"
            "The run history is loaded in the REPL variable `context` as a structured object. "
            "Inspect it with a `repl` block before answering."
        )
    return (
        f"{task_line}\n\n"
        "Continue inspecting `context` if needed. Use another `repl` block, or return "
        "FINAL(...)/FINAL_VAR(...) when you have the answer."
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
            f"- rlm_query[{operation['status']}] query={operation['query']!r} "
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


def _looks_like_message(item: Any) -> bool:
    return isinstance(item, dict) and "role" in item and "content" in item


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE)


def _append_matches(
    *,
    matches: list[RLMSearchMatch],
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
        RLMSearchMatch(
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
    if isinstance(obj, RLMContextView):
        return _context_text(obj)
    if isinstance(obj, RLMMessageView):
        return _message_text(obj)
    if isinstance(obj, RLMTraceView):
        return _trace_text(obj)
    if isinstance(obj, RLMSpanView):
        return _span_text(obj)
    if isinstance(obj, RLMSearchMatch):
        return _search_match_text(obj)
    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray, str)):
        if not obj:
            return "(empty selection)"
        return "\n\n".join(_render_object(item) for item in obj)
    return _fallback_to_text(obj)


def _context_text(context: RLMContextView) -> str:
    lines = [
        f"context: {context.label}",
        f"message_count: {len(context.messages)}",
        f"trace_count: {len(context.traces)}",
    ]
    if context.messages:
        lines.append("live_messages:")
        lines.extend(_indent_lines(_message_text(message) for message in context.messages))
    if context.traces:
        lines.append("stored_traces:")
        lines.extend(_indent_lines(_trace_text(trace) for trace in context.traces))
    return "\n".join(lines)


def _message_text(message: RLMMessageView) -> str:
    return f"message[{message.index}] {message.role}: {message.content}"


def _trace_summary_text(trace: RLMTraceView) -> str:
    parts = [
        f"workflow={trace.workflow_name}",
        f"source={trace.source}",
    ]
    if trace.metadata:
        parts.append(f"metadata={_to_json(trace.metadata)}")
    return " ".join(parts)


def _trace_text(trace: RLMTraceView) -> str:
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


def _span_text(span: RLMSpanView) -> str:
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


def _search_match_text(match: RLMSearchMatch) -> str:
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
    if isinstance(obj, RLMContextView):
        return {
            "type": "context",
            "label": obj.label,
            "message_count": len(obj.messages),
            "trace_count": len(obj.traces),
        }
    if isinstance(obj, RLMMessageView):
        return {
            "type": "message",
            "label": f"message[{obj.index}]",
            "role": obj.role,
        }
    if isinstance(obj, RLMTraceView):
        return {
            "type": "trace",
            "label": f"trace[{obj.index}]",
            "workflow": obj.workflow_name,
            "span_count": len(obj.spans),
        }
    if isinstance(obj, RLMSpanView):
        return {
            "type": "span",
            "label": f"trace[{obj.trace_index}].span[{obj.index}]",
            "kind": obj.kind,
        }
    if isinstance(obj, RLMSearchMatch):
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
