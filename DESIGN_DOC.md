# Tokentrim Design Doc

## Purpose

Tokentrim is a local, in-process SDK for managing model context before
execution. It does not run as a hosted service. Its current product shape is:

- compaction for long-running coding and agent workflows
- durable memory retrieval and writing
- optional persisted traces for debugging, replay, and later memory extraction
- thin integrations that reuse the same core runtime

The system is designed around one rule: context management should happen
through one compose-first pipeline, not through several unrelated APIs.

## User Model

From a user perspective, Tokentrim has three layers:

1. live context
   - the prompt window about to be sent to the model
   - recent messages, working state, compacted history, retrieved memory
2. durable memory
   - reusable facts stored outside the prompt
   - editable on disk
3. traces
   - execution history
   - useful for debugging, integrations, and future memory extraction

These are intentionally separate:

- compaction keeps the live prompt small
- memory preserves durable facts
- tracing records what happened

## Core API

Tokentrim exposes one primary execution pattern:

- `Tokentrim(...).compose(*steps).apply(...)`

Main public objects:

- `Tokentrim`
- `ComposedPipeline`
- `CompactConversation`
- `RetrieveDurableMemory`
- `RememberDurableMemory`

Supporting public subsystems:

- `tokentrim.memory`
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
- subsystem boundaries: memory and tracing are not transforms
- inspectable artifacts: memory and traces can be stored on disk in readable
  forms

## Package Layout

| Path | Responsibility |
| --- | --- |
| `tokentrim/client.py` | Public facade and composed-pipeline entrypoint |
| `tokentrim/pipeline/` | Request models and unified execution runtime |
| `tokentrim/transforms/base.py` | Shared transform contract |
| `tokentrim/transforms/compaction/` | Context editing and conversation compaction |
| `tokentrim/transforms/retrieve_durable_memory.py` | Memory retrieval transform |
| `tokentrim/transforms/remember_durable_memory.py` | Memory write transform |
| `tokentrim/memory/` | Durable-memory storage, retrieval, writing, indexing, markdown serialization, trace extraction |
| `tokentrim/tracing/` | Canonical trace records, trace stores, ATIF export, pipeline tracing interfaces |
| `tokentrim/working_state.py` | Shared working-state extraction, rendering, and parsing |
| `tokentrim/salience.py` | Shared ranking heuristics used by compaction and memory |
| `tokentrim/core/` | Shared low-level helpers |
| `tokentrim/types/` | Shared payload, result, and trace datatypes |
| `tokentrim/integrations/` | External SDK adapters |
| `tokentrim/errors/` | Shared SDK errors |

## Shared Runtime

The unified runtime lives in `tokentrim/pipeline/pipeline.py`.

`UnifiedPipeline.run(request)` performs:

1. request normalization
2. cloning into `PipelineState`
3. transform resolution
4. ordered transform execution
5. per-step trace accounting
6. final budget enforcement
7. freezing outputs into immutable tuples
8. result construction

The runtime owns:

- step ordering
- token accounting
- budget enforcement
- per-step tracing
- final result shape

Transforms should focus on state transformation, not runtime concerns.

## Shared Types

Minimal payload types:

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
- optional `resolve(tokenizer_model=...)`
- `run(state, request)`

Transform rules:

- read from one shared `PipelineState`
- return a new logical state
- do not mutate caller-owned input
- keep transform-specific logic inside the transform or its helper modules
- leave cloning, tracing, and budget enforcement to the pipeline

## Compaction Architecture

Compaction is the main v1 feature.

The compaction path is layered:

1. working-state extraction
2. deterministic context editing
3. microcompaction for bulky terminal-style content
4. model-backed structured history compaction
5. validation and fallback

Current goals of compaction:

- preserve active task state
- preserve commands, file paths, errors, and constraints
- reduce stale context
- keep summary output structured and predictable

Current output shape:

- a working-state system block
- a structured compacted-history system block
- preserved recent live messages

The compacted history currently uses stable sections:

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

This module is shared between compaction and memory so the system does not rely
on ad hoc parsing of rendered prompt text.

## Memory Architecture

Durable memory is a first-class subsystem, not a transform bucket.

Memory responsibilities:

- store reusable facts outside the prompt
- retrieve relevant facts into live context
- write new facts from explicit instructions or stable runtime signals
- extract narrow memory candidates from trace history

Current filesystem layout:

```text
.tokentrim/
  memory/
    users/<user_id>/sessions/<session_id>/
      index.jsonl
      entries/
        <timestamp>_<entry_id>.md
```

Current storage model:

- markdown entries with frontmatter for human readability
- JSONL index for retrieval speed

Current write sources:

- explicit “remember this” style instructions
- working-state checkpoints
- active-error snapshots
- rule-based trace extraction

Current retrieval model:

- retrieve by `user_id + session_id`
- rank with salience and query overlap
- inject one labeled system message after leading system messages

Current concrete scope support is session-oriented, although the type system
already leaves room for `user` and `project` scopes later.

## Trace Architecture

Tokentrim has two trace layers.

### Synchronous Run Trace

`Result.trace` is the lightweight per-run trace returned from every pipeline
execution.

It captures:

- pipeline trace id
- token budget
- aggregate input/output token counts
- per-step `StepTrace` entries

This trace is for immediate inspection of one run.

### Persisted Canonical Traces

Persisted trace history lives behind `TraceStore`.

Main public types:

- `TraceStore`
- `InMemoryTraceStore`
- `FileSystemTraceStore`
- `TokentrimTraceRecord`
- `TokentrimSpanRecord`

Current filesystem trace layout:

```text
.tokentrim/
  traces/
    users/<user_id>/sessions/<session_id>/
      <timestamp>_<trace_id>.atif.json
```

`FileSystemTraceStore` writes ATIF-shaped JSON for interoperability, but the
canonical Tokentrim record is still embedded inside `extra.tokentrim_trace`
for round-trip loading. Today, ATIF is treated as an export-wrapper around the
canonical Tokentrim trace model, not the sole source of truth.

## Salience

`tokentrim/salience.py` contains shared ranking logic used across subsystems.

Current salience inputs include:

- unresolved errors
- explicit constraints
- artifact density
- active file or command overlap
- task overlap
- light recency weighting

This allows memory retrieval and context editing to make consistent decisions
about what matters.

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

- `TokentrimError`
- `BudgetExceededError`

## Testing Strategy

Tests mirror package structure under `tests/tokentrim/`.

Key testing layers:

- unit tests for transforms and subsystem helpers
- pipeline tests for composition and budget behavior
- integration tests for adapter behavior
- filesystem tests for memory and trace persistence

Important behavior to keep covered:

- compaction fallback behavior
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

## Contributor Notes

Preferred local test flow:

- `pip install -e ".[dev,openai-agents]"`
- `PYTHONPATH=. pytest -q`

For ambiguous empty payloads, prefer:

- `apply(context=[])`
- `apply(tools=[])`

instead of a bare empty positional payload.
