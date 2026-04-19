# Tokentrim

Tokentrim is a local, in-process Python SDK for managing LLM context before
model execution. It does not run a hosted service and it does not require a
Tokentrim backend.

The current architecture has four main surfaces:

- runtime context management
- local-first memory
- persisted traces
- offline consolidation

## What It Does

Tokentrim exposes one compose-first API:

- `Tokentrim(...).compose(*steps).apply(...)`

The same runtime can operate on:

- `context`
- `tools`

The current wedge is long-running coding and agent workflows:

- compact old conversation safely
- inject useful memory into the next turn
- let an agent write bounded session memory
- persist traces for later offline consolidation

## Install

Base install:

```bash
pip install tokentrim
```

OpenAI Agents integration:

```bash
pip install "tokentrim[openai-agents]"
```

Development:

```bash
pip install -e ".[dev,openai-agents]"
pre-commit install
```

`tokentrim[rlm]` is still accepted as a compatibility alias, but it is now a
no-op extra kept only for backward compatibility.

## Usage

The most common types and transforms are re-exported from `tokentrim`.

### 5-Minute Start

```python
from tokentrim import Tokentrim
from tokentrim.transforms import CompactConversation, MicrocompactMessages


tt = Tokentrim(tokenizer="gpt-4o-mini", token_budget=8000)

result = tt.compose(
    MicrocompactMessages(),
    CompactConversation(model="gpt-4o-mini", keep_last=8),
).apply(context=messages)

optimized_messages = result.context
```

What happens here:

- Tokentrim microcompacts older tool-heavy traffic deterministically
- Tokentrim compacts older history only when needed
- the returned `result.context` is the payload to send to the model

### Memory Injection

If you provide a `memory_store`, Tokentrim owns memory injection at runtime.
The agent does not search memory directly. The system does that before the
model turn and prepends one bounded memory block when relevant.

```python
from tokentrim import FilesystemMemoryStore, Tokentrim
from tokentrim.transforms import CompactConversation


tt = Tokentrim(tokenizer="gpt-4o-mini")
memory_store = FilesystemMemoryStore(root_dir=".tokentrim/memory")

result = tt.compose(
    CompactConversation(model="gpt-4o-mini", keep_last=8),
).apply(
    context=messages,
    user_id="user-123",
    session_id="session-456",
    org_id="org-1",
    task_hint="debug a failed database connection",
    memory_store=memory_store,
)
```

By default:

- memory injection is system-owned
- injected memory is additive
- memory injection is context-only
- the agent itself does not rewrite durable memory

### Agent-Aware Session Memory

If you want the agent to be able to write memory, enable agent-aware mode.
Tokentrim exposes a standard session-memory tool and a system prompt that tells
the agent when to use it.

```python
from tokentrim import FilesystemMemoryStore, Tokentrim
from tokentrim.transforms import CompactConversation


tt = Tokentrim(tokenizer="gpt-4o-mini")
memory_store = FilesystemMemoryStore(root_dir=".tokentrim/memory")

result = tt.compose(
    CompactConversation(model="gpt-4o-mini", keep_last=8),
).apply(
    context=messages,
    user_id="user-123",
    session_id="session-456",
    org_id="org-1",
    memory_store=memory_store,
    agent_aware_memory=True,
)
```

Important rule:

- runtime agents write session memory only

Durable `user` and `org` memory are rewritten later by the offline
consolidator.

### Mixed Context + Tools Pipeline

```python
from tokentrim import FilesystemMemoryStore, FilesystemTraceStore, Tokentrim
from tokentrim.transforms import CompactConversation, MicrocompactMessages


tt = Tokentrim(tokenizer="gpt-4o-mini", token_budget=8000)
memory_store = FilesystemMemoryStore(root_dir=".tokentrim/memory")
trace_store = FilesystemTraceStore(root_dir=".tokentrim/traces")

result = tt.compose(
    MicrocompactMessages(),
    CompactConversation(model="gpt-4o-mini", keep_last=8),
).apply(
    context=messages,
    tools=tools,
    user_id="user-123",
    session_id="session-456",
    org_id="org-1",
    task_hint="debug a failed database connection",
    memory_store=memory_store,
    trace_store=trace_store,
    agent_aware_memory=True,
)

optimized_messages = result.context
optimized_tools = result.tools
```

`context=[...]` and `tools=[...]` are the preferred explicit call shapes.
`apply(payload)` still works for non-empty lists, but named arguments are
clearer and safer.

For real local usage, this should be the default mental model:

- memory is persisted under `.tokentrim/memory`
- traces are persisted under `.tokentrim/traces`
- in-memory stores are mainly for tests and short-lived experiments

If you want offline consolidation later, use `FilesystemTraceStore(...)` from
the beginning. The consolidator depends on durable trace history.

### Canonical Message Shape

Tokentrim understands structured tool traffic in `context`. The preferred
message model is:

- user turn: `{"role": "user", "content": "..."}`
- assistant turn: `{"role": "assistant", "content": "..."}`
- assistant tool call: `{"role": "assistant", "content": "...", "tool_calls": [...]}`
- tool result: `{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}`

If an integration has real tool calls, preserve them in this shape instead of
flattening them into fake user messages.

### Budgeting

`tokenizer` is the shared model used for token counting only. Model-backed
transforms define their own model, for example:

- `CompactConversation(model=...)`

`CompactConversation` can run in two budget modes:

- explicit: pass `token_budget=...`
- automatic: omit `token_budget` and let Tokentrim derive a threshold from the
  configured model/context window

When compaction runs, Tokentrim emits a deterministic working-state block plus
the compacted-history block before the live messages.

For explicit control, `CompactConversation` also accepts:

- `strategy="balanced" | "aggressive" | "minimal"`
- `instructions=...`
- `context_window=...`
- `auto_budget=False`
- `model_options={...}`

`model_options` is passed through to LiteLLM-compatible providers.

Example with a non-OpenAI endpoint:

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

## Traces

Tokentrim supports persisted canonical trace history through:

- `TraceStore`
- `FilesystemTraceStore`
- `InMemoryTraceStore`
- `TokentrimTraceRecord`
- `TokentrimSpanRecord`

Stored traces are additive. They do not replace `result.trace`.

`result.trace` is the synchronous pipeline trace for one Tokentrim run.
Persisted trace stores are the durable offline replay surface used by
integrations and consolidation.

In practice there are two different trace lifetimes:

- `result.trace`
  - synchronous
  - one pipeline invocation
  - returned immediately to the caller
- `TraceStore`
  - durable
  - survives process restarts
  - used for replay, debugging, and consolidation

For the current architecture, `FilesystemTraceStore(root_dir=".tokentrim/traces")`
should be the normal default. `InMemoryTraceStore()` is mostly useful for:

- unit tests
- examples that intentionally avoid filesystem setup
- one-process experiments where trace history does not need to survive

If you plan to run `tokentrim consolidate`, prefer `FilesystemTraceStore(...)`
immediately so traces already exist in the right offline shape.

For the current OpenAI Agents integration, stored traces are scoped by
`user_id + session_id` and include canonicalized span records for:

- native OpenAI Agents spans
- Tokentrim transform spans emitted while a wrapped run is active

## Memory Model

Tokentrim separates memory by scope:

- `session`: ephemeral working memory for one session
- `user`: durable memory shared across sessions for one user
- `org`: durable memory shared across users

For the first local iteration, filesystem-backed memory is intended to live
under `.tokentrim/`, for example:

```text
.tokentrim/
  memory/
  traces/
```

The storage format is intentionally abstracted behind store interfaces. The
runtime should not care whether memory lives in markdown, JSON, or something
else.

Recommended local-first defaults:

```python
from tokentrim import FilesystemMemoryStore, FilesystemTraceStore


memory_store = FilesystemMemoryStore(root_dir=".tokentrim/memory")
trace_store = FilesystemTraceStore(root_dir=".tokentrim/traces")
```

That layout matches the intended lifecycle:

- runtime reads from `session`, `user`, and `org` memory
- runtime writes `session` memory only
- tracing persists what happened during execution
- the offline consolidator later reads both stores and rewrites durable memory

## Offline Consolidation

The consolidator is a separate subsystem from runtime memory injection.

Runtime path:

- the system injects memory into context
- the agent can optionally write session memory
- traces are persisted durably

Offline path:

- the consolidator reads traces plus memory
- it analyzes what happened in finished sessions
- it rewrites durable `user` and `org` memory

CLI entrypoint:

```bash
tokentrim consolidate \
  --memory-dir .tokentrim/memory \
  --trace-dir .tokentrim/traces \
  --mode deterministic \
  --dry-run
```

Supported modes:

- `deterministic`
- `model`
- `agentic`

Supported write scopes:

- `--scope all`
- `--scope user`
- `--scope org`

Targeting options:

- `--user-id ...`
- `--session-id ...`
- `--org-id ...`

## OpenAI Agents SDK

```python
from agents import Agent, Runner

from tokentrim import (
    FilesystemMemoryStore,
    FilesystemTraceStore,
    CompactConversation,
    Tokentrim,
)


tt = Tokentrim(tokenizer="gpt-4o-mini")
memory_store = FilesystemMemoryStore(root_dir=".tokentrim/memory")
trace_store = FilesystemTraceStore(root_dir=".tokentrim/traces")

run_config = tt.compose(
    CompactConversation(model="gpt-4o-mini", keep_last=8),
).to_openai_agents(
    token_budget=8000,
    user_id="user-123",
    session_id="session-456",
    org_id="org-1",
    memory_store=memory_store,
    trace_store=trace_store,
    agent_aware_memory=True,
)

result = Runner.run_sync(
    Agent(name="Assistant", instructions="Answer concisely."),
    "Summarise the last discussion.",
    run_config=run_config,
)
```

This is the main first-class integration helper today. Other runtimes can use
the same core pipeline through `compose(...).apply(...)`.

The current adapter trims plain-text message inputs. Rich Responses items such
as tool calls, images, and search results are preserved unchanged.

If `trace_store` is provided, both `user_id` and `session_id` are required.

For a realistic local setup, pass both filesystem-backed stores:

- `FilesystemMemoryStore(root_dir=".tokentrim/memory")`
- `FilesystemTraceStore(root_dir=".tokentrim/traces")`

## Live Compaction Tests

Tokentrim includes an opt-in live stress test for `CompactConversation` at:

- `tests/tokentrim/transforms/compaction/test_live_stress.py`

This test calls a real model through LiteLLM, measures elapsed time and token
compression, and writes readable artifacts under:

- `tests/artifacts/test_live_compaction_stress_round_trip/`

Setup:

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

Run:

```bash
PYTHONPATH=. pytest --no-cov tests/tokentrim/transforms/compaction/test_live_stress.py -q -s
```

Non-live suite:

```bash
PYTHONPATH=. pytest -q -m "not live"
```

## Package Map

- `tokentrim/client.py`: public facade + composed-pipeline API
- `tokentrim/pipeline/`: request models and unified execution runtime
- `tokentrim/transforms/`: runtime transforms such as compaction and memory injection
- `tokentrim/memory/`: memory records, stores, querying, formatting, and agent-aware writes
- `tokentrim/tracing/`: canonical persisted trace records, stores, and pipeline tracer interfaces
- `tokentrim/consolidator/`: offline consolidation agents, planning/apply engine, and orchestration
- `tokentrim/core/`: shared helpers such as copy/freeze, token counting, and LiteLLM integration
- `tokentrim/types/`: payload/result/trace datatypes
- `tokentrim/integrations/`: adapter boundary + concrete integrations
- `tokentrim/errors/`: shared SDK errors

## Design Notes

- results are frozen dataclasses
- message and tool schemas stay intentionally minimal in v0.1
- memory injection is system-owned and additive
- agent-aware memory writes target session memory only
- offline consolidation is separate from runtime memory injection
- tool BPE is a deterministic heuristic in v0.1
