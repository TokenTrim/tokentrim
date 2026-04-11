# Tokentrim

Tokentrim is a local Python SDK for managing LLM context before model
execution. It runs entirely in-process. There is no Tokentrim service layer.

## What It Does

Tokentrim exposes a single compose-first API:

- `Tokentrim(...).compose(*steps).apply(...)`

The v1 wedge is conversation compaction for long-running coding and agent
workflows. The main promise is simple: preserve the details that matter while
reducing context size before model execution.

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

If you plan to use the in-repo RLM retrieval runtime:

```bash
pip install tokentrim
```

`tokentrim[rlm]` is still accepted as a compatibility alias, but it is now a
no-op extra because the barebones local RLM runtime ships in-repo.

For development:

```bash
pip install -e ".[dev,openai-agents,rlm]"
pre-commit install
```

## Usage

The most common transforms are re-exported from `tokentrim` directly.

### 5-Minute Start

If you are trying Tokentrim for the first time:

1. Start with `CompactConversation` only.

### Hero Path: Context Compaction

```python
from tokentrim import InMemoryTraceStore, Tokentrim
from tokentrim.transforms import CompactConversation, FilterMessages, RetrieveMemory


tt = Tokentrim(tokenizer="gpt-4o-mini", token_budget=8000)
trace_store = InMemoryTraceStore()

result = tt.compose(
    CompactConversation(model="gpt-4o-mini", keep_last=8),
    RetrieveMemory(model="gpt-4o-mini"),
).apply(
    context=messages,
    user_id="user-123",
    session_id="session-456",
    trace_store=trace_store,
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
    instructions="Preserve exact commands, file paths, and unresolved errors.",
    model_options={
        "api_base": "https://api.inceptionlabs.ai/v1",
        "api_key": os.environ["INCEPTION_API_KEY"],
    },
)
```

### Mixed Pipeline

```python
from tokentrim import InMemoryTraceStore, Tokentrim
from tokentrim.transforms import (
    CompactConversation,
    CompressToolDescriptions,
    CreateTools,
    FilterMessages,
    RetrieveMemory,
)


tt = Tokentrim(tokenizer="gpt-4o-mini", token_budget=8000)
trace_store = InMemoryTraceStore()

result = tt.compose(
    FilterMessages(),
    CompactConversation(model="gpt-4o-mini", keep_last=8),
    RetrieveMemory(model="gpt-4o-mini"),
    CompressToolDescriptions(max_description_chars=160),
    CreateTools(model="gpt-4o-mini"),
).apply(
    context=messages,
    tools=tools,
    user_id="user-123",
    session_id="session-456",
    task_hint="debug a failed database connection",
    trace_store=trace_store,
)

optimized_messages = result.context
optimized_tools = result.tools
```

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

The default compaction prompt asks the model for a structured engineering
handoff with sections such as `Goal`, `Active State`, `Critical Artifacts`,
`Open Risks`, `Next Step`, and `Older Context`. Tokentrim preserves that prompt
shape by default, but the returned compacted history is ultimately the model's
output after deterministic preprocessing, not a runtime-enforced schema.

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

- `strategy="balanced" | "aggressive" | "minimal"`
- `instructions=...`
- `context_window=...`
- `auto_budget=False`

`CompactConversation` also accepts `model_options` for provider-specific
LiteLLM arguments such as `api_base`, `api_key`, or similar completion
settings.

Use `instructions` when you want to override the default compaction prompt.
Use `strategy` to choose how aggressively deterministic pruning and
microcompaction should compress older context. The default is `balanced`.

## Live Compaction Tests

Tokentrim includes an opt-in live stress test for `CompactConversation` at:

- `tests/tokentrim/transforms/compaction/test_live_stress.py`

This test calls a real model through LiteLLM, measures elapsed time and token
compression, and writes readable artifacts under
`tests/artifacts/test_live_compaction_stress_round_trip/`.

The artifact directory contains:

- `before.txt`: the original conversation before compaction
- `after.txt`: the final compacted conversation
- `metrics.txt`: model, elapsed time, token counts, and compression rate

### Setup

Copy `.env.test.example` to `.env.test`:

```bash
cp .env.test.example .env.test
```

Then fill in one provider configuration.

GPT example:

```bash
TOKENTRIM_LIVE_COMPACTION=1
TOKENTRIM_TEST_MODEL=gpt-4o-mini
TOKENTRIM_TEST_API_KEY=sk-...
TOKENTRIM_TEST_API_BASE=https://api.openai.com/v1
```

Mercury example:

```bash
TOKENTRIM_LIVE_COMPACTION=1
TOKENTRIM_TEST_MODEL=openai/mercury-2
TOKENTRIM_TEST_API_KEY=your_mercury_key
TOKENTRIM_TEST_API_BASE=https://api.inceptionlabs.ai/v1
```

### Run

From the repo root:

```bash
PYTHONPATH=. pytest --no-cov tests/tokentrim/transforms/compaction/test_live_stress.py -q -s
```

Use `-s` so pytest shows the final summary line with:

- model
- elapsed time
- tokens before compaction
- tokens after compaction
- tokens saved
- compression percentage
- artifact directory path

You can also run all non-live tests normally:

```bash
PYTHONPATH=. pytest -q -m "not live"
```

### CI Behavior

GitHub Actions does not run live tests. The workflow excludes the `live`
pytest marker explicitly and also sets `TOKENTRIM_LIVE_COMPACTION=0`.

## On-Disk Layout

Compaction itself does not create files. If you only use
`CompactConversation`, Tokentrim stays entirely in-memory.

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
- `InMemoryTraceStore`
- `TokentrimTraceRecord`
- `TokentrimSpanRecord`

This is currently used by the OpenAI Agents integration. Stored traces are
scoped by `user_id + session_id` and include canonicalized span records for:

- native OpenAI Agents spans
- Tokentrim transform spans emitted while a wrapped OpenAI run is active

Stored traces are additive. They do not replace `result.trace`.
`RetrieveMemory(model=...)` reads this stored trace history and synthesizes an
additive memory block for the next step when earlier run details are useful.

## Recursive Memory

`RetrieveMemory(model=...)` is a trace-backed multi-step retrieval step:

- it requires `trace_store`, `user_id`, and `session_id`
- it runs whenever the transform is enabled and trace scope is available
- it reads stored traces for that user/session scope as persistent run history
- it builds a structured `context` object from live messages plus stored traces
- it uses Tokentrim's in-repo local RLM runtime to browse `context`, inspect slices, grep older history, and issue one depth-1 `rlm_query(...)` subcall over an arbitrary subcontext
- it injects one retrieved system-memory block ahead of live messages when older history is helpful for the immediate next step

Important behavior:

- it is context-only and does not modify tools
- it is a no-op when scope is missing, no traces exist, or the RLM returns empty output
- it uses a local in-process REPL loop with `max_depth=1`; that means root retrieval plus one depth-1 recursive subcall, with no deeper recursion or alternate sandboxes
- it rejects leaked RLM scaffold output such as `FINAL(...)` / `FINAL_VAR(...)` instead of injecting it into context
- blank output means “no additional memory needed this turn”; it is logged as `no_memory`, not treated as an error
- retrieved memory is trimmed to `max_memory_tokens`, and if a request budget is active it is also trimmed to remaining headroom or dropped entirely when no space remains

This transform now behaves like an RLM-backed multi-step retriever. It uses
stored trace history plus the current live context to decide what the
agent is most likely doing next, browse the relevant history, and return only
the additive memory that helps with that immediate next step.

## Package Map

- `tokentrim/client.py`: `Tokentrim` facade + composed pipeline API
- `tokentrim/pipeline/`: requests + unified pipeline runtime
- `tokentrim/tracing/`: canonical persisted trace records, stores, and pipeline tracer interfaces
- `tokentrim/transforms/`: pipeline steps (`compaction`)
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
    RetrieveMemory(model="gpt-4o-mini"),
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
- `RetrieveMemory` now uses Tokentrim's internal local-only RLM runtime for additive-memory synthesis
- `RetrieveMemory` prepends one bounded retrieved-memory system block instead of replacing live context
- tool BPE is a deterministic heuristic in v0.1
