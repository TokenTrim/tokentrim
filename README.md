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

For development:

```bash
pip install -e ".[dev,openai-agents]"
pre-commit install
```

## Usage

### Context Only

```python
from tokentrim import Tokentrim
from tokentrim.transforms import CompactConversation, FilterMessages, RetrieveMemory


tt = Tokentrim(tokenizer="gpt-4o-mini", token_budget=8000)

result = tt.compose(
    FilterMessages(),
    CompactConversation(model="gpt-4o-mini", keep_last=8),
    RetrieveMemory(),
).apply(
    context=messages,
    user_id="user-123",
    session_id="session-456",
)
optimized_messages = result.context
```

Compaction can also target OpenAI-compatible non-OpenAI endpoints through LiteLLM:

```python
import os

from tokentrim.transforms import CompactConversation


step = CompactConversation(
    model="openai/mercury-2",
    keep_last=8,
    model_options={
        "api_base": "https://api.inceptionlabs.ai/v1",
        "api_key": os.environ["INCEPTION_API_KEY"],
    },
)
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
from tokentrim import Tokentrim
from tokentrim.transforms import (
    CompactConversation,
    CompressToolDescriptions,
    CreateTools,
    FilterMessages,
    RetrieveMemory,
)


tt = Tokentrim(tokenizer="gpt-4o-mini", token_budget=8000)

result = tt.compose(
    FilterMessages(),
    CompactConversation(model="gpt-4o-mini", keep_last=8),
    RetrieveMemory(),
    CompressToolDescriptions(max_description_chars=160),
    CreateTools(model="gpt-4o-mini"),
).apply(
    context=messages,
    tools=tools,
    user_id="user-123",
    session_id="session-456",
    task_hint="debug a failed database connection",
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

`CompactConversation` also accepts `model_options` for provider-specific
LiteLLM arguments such as `api_base`, `api_key`, or similar completion
settings.

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
    RetrieveMemory(),
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
- RLM is retrieval-only in v0.1
- tool BPE is a deterministic heuristic in v0.1
