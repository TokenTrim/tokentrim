# Tokentrim

Tokentrim is a local Python SDK for managing LLM context before model
execution. It runs entirely in-process. There is no Tokentrim service layer.

## What It Does

Tokentrim exposes a single compose-first API:

- `Tokentrim(...).compose(*steps).apply(...)`

The v1 wedge is conversation compaction plus durable memory for long-running
coding and agent workflows. The main promise is simple: preserve the details
that matter while reducing context size before model execution.

The same runtime supports both payload tracks:

- context messages
- tool definitions

## Install

```bash
pip install tokentrim
```

If you plan to use the OpenAI Agents integration:

```bash
pip install "tokentrim[openai-agents]"
```

For development:

```bash
pip install -e ".[dev,openai-agents]"
pre-commit install
```

## Usage

The most common transforms are re-exported from `tokentrim` directly.

### 5-Minute Start

If you are trying Tokentrim for the first time:

1. Start with `CompactConversation` only.
2. Add `RetrieveDurableMemory` and `RememberDurableMemory` once you have
   stable `user_id` and `session_id` values in your app.
3. Add `FileSystemTraceStore` only if you want on-disk trace artifacts under
   `.tokentrim/traces/`.

### Hero Path: Context Compaction

```python
from tokentrim import CompactConversation, Tokentrim


tt = Tokentrim(tokenizer="gpt-4o-mini", token_budget=8000)

result = tt.compose(
    CompactConversation(model="gpt-4o-mini", keep_last=8),
).apply(
    context=messages,
)
optimized_messages = result.context
```

What happens here:

- Tokentrim checks whether the current context is over budget.
- If it is, it extracts working state, edits stale context, and compacts older
  history.
- If it is already under budget, compaction becomes a no-op.
- The returned `result.context` is what you should send to the model.

Compaction can also target OpenAI-compatible non-OpenAI endpoints through LiteLLM:

```python
import os

from tokentrim import CompactConversation


step = CompactConversation(
    model="openai/mercury-2",
    keep_last=8,
    model_options={
        "api_base": "https://api.inceptionlabs.ai/v1",
        "api_key": os.environ["INCEPTION_API_KEY"],
    },
)
```

### Durable Memory

```python
from tokentrim import CompactConversation, RememberDurableMemory, RetrieveDurableMemory, Tokentrim
from tokentrim.memory import LocalDirectoryMemoryStore

tt = Tokentrim(tokenizer="gpt-4o-mini", token_budget=8000)
memory_store = LocalDirectoryMemoryStore(root_dir=".tokentrim/memory")

result = tt.compose(
    CompactConversation(model="gpt-4o-mini", keep_last=8),
    RetrieveDurableMemory(memory_store=memory_store),
    RememberDurableMemory(memory_store=memory_store),
).apply(
    context=messages,
    user_id="user-123",
    session_id="session-456",
    task_hint="debug the failing pytest command",
)
optimized_messages = result.context
```

What happens here:

- `RetrieveDurableMemory` reads relevant notes for the current
  `user_id + session_id` pair and injects them as a system message.
- `RememberDurableMemory` can persist new notes from the same run, such as
  explicit "remember this" instructions, checkpoints, or active errors.
- Memory does not write anything unless you include a write step like
  `RememberDurableMemory`.

### Planned Transforms

The earlier `filter`, `compress_tools`, and `create_tools` transforms were only
exploratory sketches and are intentionally not shipped in the public package.
They are still part of the product plan, but they need real design and
benchmark validation before returning as supported APIs.

Planned future areas:

- message filtering and cleanup policies beyond compaction
- tool description compression for large tool registries
- task-aware tool suggestion or generation

Use one `Tokentrim` client and one API pattern (`compose(...).apply(...)`) for
both payload kinds.

`context=[...]` and `tools=[...]` are the preferred explicit call shapes.
`apply(payload)` still works for non-empty lists, but the named arguments are
clearer and safer.

`tokenizer` is the shared model used for token counting only. Model-backed
transforms define their own model (for example
`CompactConversation(model=...)`).

`CompactConversation` can run in two budget modes:

- explicit: pass `token_budget=...` to `Tokentrim(...)` or `.apply(...)`
- automatic: omit `token_budget` and let compaction derive a threshold from the
  configured model/context window

When compaction runs, Tokentrim emits two system blocks before the live
messages: a deterministic working-state block for the current goal, active
files, latest command, active error, constraints, and next step; then the
compacted history block for older context.

The compacted history block is also schema-shaped now. It uses stable sections
for `Goal`, `Active State`, `Critical Artifacts`, `Open Risks`, `Next Step`,
and `Older Context` so downstream agents do not have to recover structure from
freeform summaries.

For longer-running sessions, you can also add a separate durable-memory step.
`RetrieveDurableMemory` reads a local durable-memory store, `.tokentrim/memory`
by default, and injects only the most relevant saved notes after the leading
system messages. The store keeps an `index.jsonl` for retrieval and markdown
entry files under `entries/` so the saved memory is inspectable and editable.

`RememberDurableMemory` is the write-side companion. It can persist explicit
"remember this" notes, working-state checkpoints, and active-error snapshots
for the current `user_id + session_id` scope.

If a `trace_store` is also attached to the run, the default write policy can
extract rule-based memory candidates from prior traces too, for example
repeated failures or later resolutions of repeated failures.

For simple usage, automatic mode is usually the right choice:

```python
from tokentrim import CompactConversation, Tokentrim


tt = Tokentrim(tokenizer="gpt-4o-mini")

result = tt.compose(
    CompactConversation(model="gpt-4o-mini", keep_last=8),
).apply(context=messages)
```

When automatic mode is active, the derived budget is used both for deciding
when to compact and for the pipeline's final hard budget enforcement.

If you need explicit control, `CompactConversation` also accepts:

- `context_window=...`
- `reserved_output_tokens=...`
- `auto_compact_buffer_tokens=...`
- `auto_budget=False`

`CompactConversation` also accepts `model_options` for provider-specific
LiteLLM arguments such as `api_base`, `api_key`, or similar completion
settings.

## On-Disk Layout

Tokentrim only creates on-disk artifacts if you opt into filesystem-backed
memory or tracing.

Durable memory with `LocalDirectoryMemoryStore`:

```text
.tokentrim/
  memory/
    users/<user_id>/sessions/<session_id>/
      index.jsonl
      entries/
        <timestamp>_<entry_id>.md
```

Stored traces with `FileSystemTraceStore`:

```text
.tokentrim/
  traces/
    users/<user_id>/sessions/<session_id>/
      <timestamp>_<trace_id>.atif.json
```

When files are written:

- memory markdown files are written only when `RememberDurableMemory` runs and
  produces a new durable-memory candidate
- trace files are written only when you attach a `FileSystemTraceStore`
- compaction itself does not create files

If you only use `CompactConversation`, Tokentrim stays entirely in-memory.

## Transform Contract

All transforms operate on one shared `PipelineState`:

```python
PipelineState(
    context=[...],
    tools=[...],
)
```

Each transform can read or modify either side and returns the next state.
Tracing and budget enforcement stay in the pipeline runtime.

## Result Shape

Every run returns one `Result` object:

- `result.trace`: run metadata and per-step trace entries
- `result.context`: final context payload as a tuple
- `result.tools`: final tools payload as a tuple

Both fields are always present. Unused sides are empty tuples.

`result.trace` is the synchronous pipeline trace for that one Tokentrim run.
It is not the same as the persisted identity trace store used by integrations.

## Stored Traces

Tokentrim also supports persisted canonical trace history through:

- `TraceStore`
- `FileSystemTraceStore`
- `InMemoryTraceStore`
- `TokentrimTraceRecord`
- `TokentrimSpanRecord`

This is currently used by the OpenAI Agents integration. Stored traces are
scoped by `user_id + session_id` and include canonicalized span records for:

- native OpenAI Agents spans
- Tokentrim transform spans emitted while a wrapped OpenAI run is active

Stored traces are additive. They do not replace `result.trace`.

If you want on-disk trajectory files, use `FileSystemTraceStore`. Completed
traces are written under `.tokentrim/traces/...` as ATIF-shaped JSON with the
canonical Tokentrim trace embedded in `extra` metadata for round-trip loading.

## Package Map

- `tokentrim/client.py`: `Tokentrim` facade + composed pipeline API
- `tokentrim/pipeline/`: requests + unified pipeline runtime
- `tokentrim/tracing/`: canonical persisted trace records, stores, and pipeline tracer interfaces
- `tokentrim/transforms/`: pipeline steps (`compaction`, durable-memory retrieval, durable-memory writing)
- `tokentrim/memory/`: durable-memory stores and retrieval utilities
- `tokentrim/core/`: shared helpers (`copy_utils`, `token_counting`, `llm_client`)
- `tokentrim/types/`: payload/result/trace datatypes
- `tokentrim/integrations/`: adapter boundary + OpenAI Agents integration
- `tokentrim/errors/`: shared SDK errors

## OpenAI Agents SDK

```python
from agents import Agent, Runner

from tokentrim import CompactConversation, Tokentrim


tt = Tokentrim(tokenizer="gpt-4o-mini")

run_config = tt.compose(
    CompactConversation(model="gpt-4o-mini", keep_last=8),
).to_openai_agents(
    token_budget=8000,
)

result = Runner.run_sync(
    Agent(name="Assistant", instructions="Answer concisely."),
    "Summarise the last discussion.",
    run_config=run_config,
)
```

This is the main first-class integration helper today. Other runtimes can use
the same compaction step through the generic `compose(...).apply(...)` API.

The current adapter trims plain-text message inputs. Rich Responses items such
as tool calls, images, and search results are preserved unchanged.

The current adapter is still message-oriented in practice. Tool transforms in a
pipeline will run, but OpenAI Agents hook inputs only provide message history to
Tokentrim.

If `trace_store` is provided, both `user_id` and `session_id` are required.
Stored traces include canonical OpenAI spans plus Tokentrim transform spans
such as `compaction` when those transforms run inside an active OpenAI trace.

## Design Notes

- results are frozen dataclasses
- message and tool schemas stay intentionally minimal in v0.1
- tool BPE is a deterministic heuristic in v0.1
