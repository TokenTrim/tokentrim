# Tokentrim

Tokentrim is a local Python SDK for optimizing LLM context and tool payloads
before model execution. It runs entirely in-process. There is no Tokentrim
service layer.

## What It Does

Tokentrim exposes a single compose-first API:

- `Tokentrim(...).compose(*steps).apply(...)`

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

If you plan to use recursive memory synthesis:

```bash
pip install "tokentrim[rlm]"
```

For development:

```bash
pip install -e ".[dev,openai-agents,rlm]"
pre-commit install
```

## Usage

### Context Only

```python
from tokentrim import InMemoryTraceStore, Tokentrim
from tokentrim.transforms import CompactConversation, FilterMessages, RetrieveMemory


tt = Tokentrim(tokenizer="gpt-4o-mini", token_budget=8000)
trace_store = InMemoryTraceStore()

result = tt.compose(
    FilterMessages(),
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

### Tools Only

```python
from tokentrim import Tokentrim
from tokentrim.transforms import CompressToolDescriptions, CreateTools


tt = Tokentrim(tokenizer="gpt-4o-mini", token_budget=8000)

result = tt.compose(
    CompressToolDescriptions(max_description_chars=160),
    CreateTools(model="gpt-4o-mini"),
).apply(
    tools=tools,
    task_hint="debug a failed database connection",
)
optimized_tools = result.tools
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

For legacy single-payload calls, `apply(payload)` still works for non-empty
lists. For empty payloads, use `context=[]` or `tools=[]` explicitly.

`tokenizer` is the shared model used for token counting only. Model-backed
transforms define their own model (for example
`CompactConversation(model=...)` and `CreateTools(model=...)`).

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
`RetrieveMemory(model=...)` reads this stored trace history and synthesizes one
scoped system-memory message when relevant traces are available.

## Recursive Memory

`RetrieveMemory(model=...)` is a trace-backed memory enrichment step:

- it requires `trace_store`, `user_id`, and `session_id`
- it only runs when a `token_budget` is configured and the current live context plus serialized trace history would exceed that budget
- it reads recent stored traces for that user/session scope
- it builds a retrieval prompt from the current task plus live messages
- it calls the optional external `rlm` runtime to synthesize one short memory block
- it prepends that memory as a `system` message when synthesis succeeds

Important behavior:

- it is context-only and does not modify tools
- it is a no-op when scope is missing, no traces exist, no `token_budget` is set, the combined live context plus trace history already fits in budget, or the synthesized output is blank
- it currently uses the external runtime in `environment="local"`
- it raises a transform-specific configuration error if `tokentrim[rlm]` is not installed
- it rejects leaked RLM scaffold output such as `FINAL(...)` / `FINAL_VAR(...)` instead of injecting it into context

This transform is not a compaction step. It reduces stored historical trace data
into a smaller memory block, but it usually increases the final live prompt
because that memory is injected into the current context.

## Package Map

- `tokentrim/client.py`: `Tokentrim` facade + composed pipeline API
- `tokentrim/pipeline/`: requests + unified pipeline runtime
- `tokentrim/tracing/`: canonical persisted trace records, stores, and pipeline tracer interfaces
- `tokentrim/transforms/`: domain transforms (`filter`, `compaction`, `rlm`, `compress_tools`, `create_tools`)
- `tokentrim/core/`: shared helpers (`copy_utils`, `token_counting`, `llm_client`)
- `tokentrim/types/`: payload/result/trace datatypes
- `tokentrim/integrations/`: adapter boundary + OpenAI Agents integration
- `tokentrim/errors/`: shared SDK errors

## OpenAI Agents SDK

```python
from agents import Agent, Runner

from tokentrim import InMemoryTraceStore, Tokentrim
from tokentrim.transforms import CompactConversation, FilterMessages, RetrieveMemory


tt = Tokentrim(tokenizer="gpt-4o-mini")
trace_store = InMemoryTraceStore()

run_config = tt.compose(
    FilterMessages(),
    CompactConversation(model="gpt-4o-mini", keep_last=8),
    RetrieveMemory(model="gpt-4o-mini"),
).to_openai_agents(
    token_budget=8000,
    user_id="user-123",
    session_id="session-456",
    trace_store=trace_store,
)

result = Runner.run_sync(
    Agent(name="Assistant", instructions="Answer concisely."),
    "Summarise the last discussion.",
    run_config=run_config,
)

stored_traces = trace_store.list_traces(
    user_id="user-123",
    session_id="session-456",
)
```

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
- `RetrieveMemory` is TraceStore-backed memory synthesis using the optional external `rlm` runtime
- `RetrieveMemory` enriches live context by prepending one synthesized `system` memory block
- tool BPE is a deterministic heuristic in v0.1
