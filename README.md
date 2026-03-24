# Tokentrim

Tokentrim is a local Python SDK for trimming LLM context and tool payloads before
you call your model. The library runs in your process. There is no Tokentrim
service layer, and model-backed features use LiteLLM directly with whichever
backend you configure.

## What It Does

Tokentrim exposes two synchronous entrypoints:

- `get_better_context(...)` to reduce message history cost
- `get_better_tools(...)` to reduce tool payload cost

The SDK applies explicit pipelines:

- context: `filter -> compaction -> rlm`
- tools: `bpe -> creator`

## Install

```bash
pip install tokentrim
```

## Usage

```python
from tokentrim import Tokentrim
from tokentrim.context import CompactConversation, FilterMessages, RetrieveMemory
from tokentrim.tools import CompressToolDescriptions, CreateTools

tt = Tokentrim(
    model="gpt-4o-mini",
    compaction_model="gpt-4o-mini",
    tool_creation_model="gpt-4o-mini",
    token_budget=8000,
)

context_result = tt.get_better_context(
    messages,
    steps=(
        FilterMessages(),
        CompactConversation(model="gpt-4o-mini", keep_last=8),
        RetrieveMemory(),
    ),
)

tools_result = tt.get_better_tools(
    tools,
    task_hint="debug a failed database connection",
    steps=(
        CompressToolDescriptions(max_description_chars=160),
        CreateTools(model="gpt-4o-mini"),
    ),
)
```

`model` is the shared tokenizer/default model. `compaction_model` and
`tool_creation_model` let you override model-backed features independently.

## OpenAI Agents SDK

```python
from agents import Agent, Runner

from tokentrim import Tokentrim
from tokentrim.context import CompactConversation, FilterMessages, RetrieveMemory
from tokentrim.integrations import OpenAIAgentsAdapter, OpenAIAgentsOptions

tt = Tokentrim(
    model="gpt-4o-mini",
    compaction_model="gpt-4o-mini",
)

agent = Agent(
    name="Assistant",
    instructions="Answer concisely.",
)

run_config = tt.wrap_integration(
    OpenAIAgentsAdapter(
        options=OpenAIAgentsOptions(
            token_budget=8000,
            steps=(
                FilterMessages(),
                CompactConversation(keep_last=8),
                RetrieveMemory(),
            ),
            user_id="user-123",
            session_id="session-456",
        )
    ),
)

result = Runner.run_sync(
    agent,
    "Summarise the last discussion.",
    run_config=run_config,
)
```

The current adapter trims plain text message inputs only. Rich Responses items
such as tool calls, images, and search results are preserved unchanged.

## Design Notes

- results are frozen dataclasses
- message and tool schemas stay intentionally minimal in v0.1
- RLM is retrieval-only in v0.1
- tool BPE is a deterministic heuristic in v0.1
