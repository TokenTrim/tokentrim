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

## Package Map

- `tokentrim/client.py`: `Tokentrim` facade + composed pipeline API
- `tokentrim/pipeline/`: requests + unified pipeline runtime
- `tokentrim/transforms/`: domain transforms (`filter`, `compaction`, `rlm`, `compress_tools`, `create_tools`)
- `tokentrim/core/`: shared helpers (`copy_utils`, `token_counting`, `llm_client`)
- `tokentrim/types/`: payload/result/trace datatypes
- `tokentrim/integrations/`: adapter boundary + OpenAI Agents integration
- `tokentrim/errors/`: shared SDK errors

## OpenAI Agents SDK

```python
from agents import Agent, Runner

from tokentrim import Tokentrim
from tokentrim.transforms import CompactConversation, FilterMessages, RetrieveMemory


tt = Tokentrim(tokenizer="gpt-4o-mini")

run_config = tt.compose(
    FilterMessages(),
    CompactConversation(model="gpt-4o-mini", keep_last=8),
    RetrieveMemory(),
).to_openai_agents(
    token_budget=8000,
    user_id="user-123",
    session_id="session-456",
)

result = Runner.run_sync(
    Agent(name="Assistant", instructions="Answer concisely."),
    "Summarise the last discussion.",
    run_config=run_config,
)
```

The current adapter trims plain-text message inputs. Rich Responses items such
as tool calls, images, and search results are preserved unchanged.

The current adapter is still message-oriented in practice. Tool transforms in a
pipeline will run, but OpenAI Agents hook inputs only provide message history to
Tokentrim.

## Design Notes

- results are frozen dataclasses
- message and tool schemas stay intentionally minimal in v0.1
- RLM is retrieval-only in v0.1
- tool BPE is a deterministic heuristic in v0.1
