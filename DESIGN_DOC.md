# Tokentrim Design Doc

## Purpose

Tokentrim is a local, in-process SDK for managing model context before
execution. It does not run as a hosted service. Its current product shape is:

- compaction for long-running coding and agent workflows
- optional persisted traces for debugging and integrations
- thin integrations that reuse the same core runtime

The system is designed around one rule: context management should happen
through one compose-first pipeline, not through several unrelated APIs.

## User Model

From a user perspective, Tokentrim has two layers:

1. live context
   - the prompt window about to be sent to the model
   - recent messages, working state, compacted history
2. traces
   - execution history
   - useful for debugging and integrations

These are intentionally separate:

- compaction keeps the live prompt small
- tracing records what happened

## Core API

Tokentrim exposes one primary execution pattern:

- `Tokentrim(...).compose(*steps).apply(...)`

Main public objects:

- `Tokentrim`
- `ComposedPipeline`
- `CompactConversation`

Supporting public subsystems:

- `tokentrim.tracing`

Typical execution shape:

1. create a `Tokentrim` client with shared defaults such as `tokenizer`
2. compose transform instances
3. call `.apply(...)`
4. receive one immutable `Result`

There are not separate context and tools runtimes. Both payload tracks flow
through one shared `PipelineState`.

## Architectural Principles

- local-first: no Tokentrim backend
- compose-first: one primary way to use the system
- explicit transforms: no hidden background optimizer
- immutable results: the pipeline returns frozen outputs
- subsystem boundaries: tracing stays separate from compaction

## Package Layout

| Path | Responsibility |
| --- | --- |
| `tokentrim/client.py` | Public facade and composed-pipeline entrypoint |
| `tokentrim/pipeline/` | Request models and unified execution runtime |
| `tokentrim/transforms/base.py` | Shared transform contract |
| `tokentrim/transforms/filter/` | Message filtering transform |
| `tokentrim/transforms/compaction/` | Conversation compaction transform |
| `tokentrim/transforms/rlm/` | Retrieval-memory transform + memory store interface |
| `tokentrim/transforms/compress_tools/` | Deterministic tool description compression |
| `tokentrim/transforms/create_tools/` | Model-backed missing tool creation |
| `tokentrim/core/copy_utils.py` | Clone/freeze helpers for payload safety |
| `tokentrim/core/token_counting.py` | Token counting helpers |
| `tokentrim/core/llm_client.py` | LiteLLM wrapper used by model-backed transforms |
| `tokentrim/tracing/` | Canonical persisted tracing models, store implementations, and pipeline tracing interfaces |
| `tokentrim/types/` | Shared datatypes (`Message`, `Tool`, `PipelineState`, `Trace`, `StepTrace`, `Result`) |
| `tokentrim/integrations/base.py` | Integration adapter contract |
| `tokentrim/integrations/openai_agents/` | OpenAI Agents adapter modules |
| `tokentrim/errors/` | Shared error taxonomy |

## Types

Tokentrim uses minimal payload types:

- `Message`: `{role, content}`
- `Tool`: `{name, description, input_schema}`
- `PipelineState`: `{context: list[Message], tools: list[Tool]}`

Every run returns one `Result`:

- `trace: Trace`
- `context: tuple[Message, ...]`
- `tools: tuple[Tool, ...]`

Both result payloads are always present. Unused sides are empty tuples.

## Transform Contract

All transforms implement `Transform` and provide:

- `name`: stable identifier for tracing
- optional `resolve(tokenizer_model=...)`: binding phase before run
- `run(state, request)`: transform step over one shared `PipelineState`

Transform expectations:

- every transform receives both `context` and `tools` via `PipelineState`
- transforms may read or modify either side of state
- transforms should return a new logical state and must not rely on mutating
  caller-owned input state
- `resolve(...)` is the place to bind tokenizer-dependent or model-dependent
  configuration before execution

## Pipeline Runtime

`UnifiedPipeline.run(request)` executes one shared state flow:

1. clone inbound request payloads into `PipelineState`
2. resolve each step
3. execute step in order
4. compute per-step token/item deltas from state before/after each step
5. enforce final token budget
6. freeze final context/tools payloads
7. return immutable `Result`

Budget enforcement is centralized at pipeline exit and currently uses combined
context + tools token counts.

Tracing is also centralized in the pipeline. Each executed step produces one
`StepTrace` entry with item counts, token counts, and a changed/not-changed
flag.

The default prompt asks for sections such as:

- `Goal`
- `Active State`
- `Critical Artifacts`
- `Open Risks`
- `Next Step`
- `Older Context`

## Working State

`tokentrim/working_state.py` holds the shared representation of active state.

Its role is to capture the most operationally relevant information from the
current run, such as:

- goal
- current step
- active files
- latest command
- active error
- constraints
- next step

This module is shared inside the compaction stack so the system does not rely
on ad hoc parsing of rendered prompt text.

## Tracing Model

Tokentrim currently has two tracing layers:

1. `Result.trace`
   - always produced by the unified pipeline
   - contains per-step `StepTrace` summaries
   - intended for immediate inspection of a single Tokentrim run
2. persisted canonical trace history
   - stored through `TraceStore`
   - returns `TokentrimTraceRecord` values with nested `TokentrimSpanRecord`s
   - intended for integration history and future transform reads

The canonical persisted trace model is source-agnostic:

- trace records carry `source`, `capture_mode`, `source_trace_id`,
  `user_id`, `session_id`, and raw source payload
- span records carry `kind`, `name`, `metrics`, `data`, `parent_id`,
  `source_span_id`, and raw source payload

The default `InMemoryTraceStore` keeps active traces in-process, indexes
completed traces by `(user_id, session_id)`, returns traces newest-first, and
orders stored spans chronologically on read.

## Integration Boundary

Integrations should stay thin and reuse the same core pipeline.

`IntegrationAdapter[ConfigT]` defines the adapter contract.

Adapter responsibilities:

- map host SDK inputs into Tokentrim payloads
- attach Tokentrim behavior to host runtime hooks
- preserve host behavior when Tokentrim is effectively disabled
- optionally connect host tracing to `PipelineTracer`

Current first-class integration:

- `tokentrim/integrations/openai_agents/`

That integration is message-oriented in practice because of the host hook
shapes. It can persist canonical traces and attach Tokentrim transform spans to
the host trace stream.

## Error Model

Tokentrim uses:

- shared root errors in `tokentrim/errors/`
- transform-local or subsystem-local errors where needed

Examples:

- shared: `TokentrimError`, `BudgetExceededError`
- transform-local: configuration/execution/output errors where needed

## Testing Strategy

Tests mirror package structure under `tests/tokentrim/`.

Key testing layers:

- unit tests for transforms and subsystem helpers
- pipeline tests for composition and budget behavior
- integration tests for adapter behavior
- filesystem tests for memory and trace persistence

Important behavior to keep covered:

- compaction model failure behavior
- working-state preservation
- durable-memory retrieval and writing
- trace persistence and round-trip loading
- step ordering and per-step tracing

## Extension Guidance

### Add a New Transform

1. add the transform implementation
2. keep helper logic close to the transform unless it is a true shared concept
3. export it from `tokentrim/transforms/__init__.py` only if it is public
4. add mirrored tests
5. add pipeline-level tests if order or budget behavior matters

### Add a New Subsystem Capability

Use a subsystem instead of a transform when the new feature is primarily:

- storage
- indexing
- retrieval
- tracing
- policy infrastructure

Memory and tracing follow this rule already.

### Add a New Integration

1. keep host SDK details inside `tokentrim/integrations/<name>/`
2. reuse composed pipelines rather than reimplementing runtime logic
3. isolate host-specific tracing behind adapter code and `PipelineTracer`
4. test host mapping and pass-through behavior explicitly

## Current Non-Goals

The current architecture does not treat Tokentrim as:

- a hosted memory service
- a general agent framework
- a background daemon
- a vector database product
- a knowledge graph system

Those may become adjacent concerns later, but they are not part of the current
core design.

For the current OpenAI Agents integration, `RetrieveMemory` runs unchanged on
the core pipeline side. It only needs `trace_store`, `user_id`, and
`session_id` so it can read persisted canonical traces for that session scope.

## Contributor Notes

Preferred local test flow:

- `pip install -e ".[dev,openai-agents]"`
- `PYTHONPATH=.` when running `pytest -q`

For ambiguous empty payloads, prefer:

- `apply(context=[])`
- `apply(tools=[])`

instead of a bare empty positional payload.
