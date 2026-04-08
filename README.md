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
    instructions="Preserve exact commands, file paths, and unresolved errors.",
    model_options={
        "api_base": "https://api.inceptionlabs.ai/v1",
        "api_key": os.environ["INCEPTION_API_KEY"],
    },
)
```

### Planned Transforms

The earlier `filter`, `compress_tools`, `create_tools`, and memory-oriented
experiments were exploratory only. They are intentionally not shipped in the
public package on this branch.

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
