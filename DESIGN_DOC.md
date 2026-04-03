# Tokentrim Design Doc

## Scope

This document describes Tokentrim's current architecture, boundaries, and
extension model.

## Core Goals

- local, in-process SDK (no hosted Tokentrim service)
- one primary execution pattern: `compose(...).apply(...)`
- explicit transform composition
- inspectable runs through stable synchronous and persisted trace models
- thin integration adapters that reuse the same core pipeline

## Public API

Tokentrim exposes two primary user-facing objects:

- `Tokentrim` in `tokentrim/client.py`
- `ComposedPipeline` returned by `Tokentrim.compose(...)`

Execution shape:

1. create client with shared defaults (tokenizer, default token budget)
2. compose transform instances
3. call `.apply(...)`
4. receive one immutable `Result`

There are no separate context/tools runner types. Both payload tracks use the
same compose-first execution path and the same internal pipeline state.

For OpenAI Agents users, the primary integration path is:

- `Tokentrim.compose(...).to_openai_agents(...)`

This keeps integration wiring compose-first and avoids direct
adapter/options construction in app code.

Additional public helpers exist on `Tokentrim`:

- `wrap_integration(adapter, config=...)` for generic adapter wiring
- `openai_agents_config(...)` as a convenience wrapper around
  `compose(*steps).to_openai_agents(...)`

Tracing-related public types live under `tokentrim/tracing/`:

- `TraceStore` / `InMemoryTraceStore`
- `TokentrimTraceRecord` / `TokentrimSpanRecord`
- `PipelineTracer` / `PipelineSpan`

## Package Layout

| Path | Responsibility |
| --- | --- |
| `tokentrim/client.py` | Public facade and compose-first API |
| `tokentrim/pipeline/requests.py` | Immutable request models (`PipelineRequest` plus compatibility wrappers) |
| `tokentrim/pipeline/pipeline.py` | Unified execution runtime (`UnifiedPipeline`) |
| `tokentrim/transforms/base.py` | Shared transform contract |
| `tokentrim/transforms/filter/` | Message filtering transform |
| `tokentrim/transforms/compaction/` | Conversation compaction transform |
| `tokentrim/transforms/rlm/` | TraceStore-backed memory synthesis transform using the external `rlm` runtime |
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

Both result payload fields are always present. Unused sides are empty tuples.

Tokentrim also defines a separate persisted tracing model:

- `TokentrimTraceRecord`
- `TokentrimSpanRecord`
- `TraceStore`

This persisted model is intentionally separate from `Result.trace`. `Result.trace`
is the synchronous pipeline trace for one `compose(...).apply(...)` call.
Persisted trace records are integration-facing history records intended for
cross-run lookup by `user_id + session_id`.

## Transform Contract

All transforms implement `Transform` (`tokentrim/transforms/base.py`):

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

`RetrieveMemory` is a context-only transform:

- it reads recent canonical trace history from `TraceStore`
- it builds a prompt from the active task plus current live messages
- it delegates synthesis to the optional external `rlm.RLM` runtime
- it prepends one synthesized `system` memory message when successful
- it leaves tools unchanged

Operationally, `RetrieveMemory` is a no-op when trace scope is missing, no
stored traces exist, or synthesis returns blank output. Invalid runtime setup
raises `RLMConfigurationError`. Unexpected runtime failures, including leaked
RLM scaffold text such as `FINAL(...)` / `FINAL_VAR(...)`, raise
`RLMExecutionError`.

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

The pipeline may also receive a `PipelineTracer`. When present, the runtime
opens one integration-owned span per transform step around `resolved_step.run(...)`.
This allows integrations to attach Tokentrim transform execution to their host
tracing systems without importing host SDKs into the core pipeline.

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

`RetrieveMemory` consumes this persisted history rather than `Result.trace`.
It selects a bounded recent window, serializes stable canonical fields only,
and omits raw source payloads when constructing the synthesis prompt.

## Integration Boundary

`IntegrationAdapter[ConfigT]` defines the integration contract:

- adapters expose `wrap(tokentrim, config=None) -> ConfigT`
- adapters are responsible for attaching Tokentrim behavior to another SDK's
  configuration or runtime hooks
- adapters may map external payloads into Tokentrim payloads internally, but
  that is an implementation detail rather than part of the abstract contract

OpenAI Agents integration is split by concern:

- `options.py`: adapter options
- `sdk.py`: SDK compatibility utilities
- `mappers.py`: payload mapping logic
- `hooks.py`: integration hook wiring
- `pipeline_tracing.py`: OpenAI-backed implementation of `PipelineTracer`
- `translator.py`: OpenAI-to-canonical trace/span translation
- `tracing.py`: global processor installation, metadata routing, and store persistence
- `adapter.py`: orchestrating adapter entrypoint

The current OpenAI Agents integration is still driven by message-oriented hook
payloads. The adapter maps those into Tokentrim message inputs and ignores the
tools side unless the caller explicitly supplies tools through another path.

When `trace_store` is configured for OpenAI Agents:

- the adapter requires both `user_id` and `session_id`
- the adapter installs a process-global OpenAI tracing processor
- reserved Tokentrim routing metadata is injected into `RunConfig.trace_metadata`
- only traces carrying that reserved metadata are persisted
- Tokentrim transform execution is emitted as OpenAI SDK `custom` spans and
  translated back into canonical `kind="transform"` persisted spans

This means an OpenAI-backed stored trace can contain both native SDK spans
(`agent`, `generation`, `response`, `function`, `handoff`, and so on) and
Tokentrim transform spans such as `filter` or `compaction`.

## Error Model

Tokentrim uses shared root errors in `tokentrim/errors/` and per-transform
errors inside each transform package (`error.py`).

Examples:

- shared: `TokentrimError`, `BudgetExceededError`
- transform-local: configuration/execution/output errors where needed
- `tokentrim/transforms/rlm/error.py`: `RLMTransformError`,
  `RLMConfigurationError`, `RLMExecutionError`

## Testing Layout

Tests mirror package structure under `tests/tokentrim/`:

- `tests/tokentrim/core/`
- `tests/tokentrim/pipeline/`
- `tests/tokentrim/transforms/...`
- `tests/tokentrim/integrations/`

Coverage is enforced in CI via pytest coverage thresholds.

## Add a New Transform

1. create `tokentrim/transforms/<name>/`
2. add `transform.py`
3. add `__init__.py` that re-exports the transform class
4. add `error.py` only if the transform needs transform-specific exceptions
5. implement `Transform` with:
   - stable `name` for tracing
   - a `run(state, request)` implementation that accepts and returns
     `PipelineState`
   - optional `resolve(...)` if configuration must be bound at runtime
6. keep the transform focused on state transformation; cloning, freezing,
   token accounting, tracing, and budget enforcement are handled by the
   pipeline
7. export from `tokentrim/transforms/__init__.py` if the transform is public
8. add mirrored tests under `tests/tokentrim/transforms/<name>/`
9. add or update pipeline-level tests if the transform depends on composition
   order, budget behavior, or integration-specific wiring

Conventions used in this repository:

- transform packages are small and self-contained
- public transform names are exported from both the package-local `__init__.py`
  and `tokentrim/transforms/__init__.py`
- transform tests should cover happy path behavior, unchanged-side behavior,
  and any transform-local failure modes

## Add a New Integration

1. create `tokentrim/integrations/<name>/`
2. define any integration-specific options/config helpers
3. implement `IntegrationAdapter[ConfigT]`
4. keep SDK-specific mapping and hook wiring inside the integration package
5. reuse `Tokentrim.compose(...).apply(...)` or composed pipeline helpers
   instead of reimplementing pipeline logic
6. add tests under `tests/tokentrim/integrations/`

Integration design guidance:

- keep the adapter thin and delegate optimization behavior to the core pipeline
- preserve host SDK behavior when Tokentrim is effectively disabled
- document any host-SDK limitations or payload shapes that are intentionally
  passed through unchanged
- prefer implementing host-specific tracing through `PipelineTracer` rather than
  importing host tracing APIs into `tokentrim/pipeline/`

For the current OpenAI Agents integration, `RetrieveMemory` runs unchanged on
the core pipeline side. It only needs `trace_store`, `user_id`, and
`session_id` so it can read persisted canonical traces for that session scope.

## Contributor Notes

For local test runs, install the package in editable mode or set
`PYTHONPATH=.` before running pytest. CI installs with:

- `pip install -e ".[dev,openai-agents]"`
- install `.[rlm]` as well when working on the recursive-memory transform
- `PYTHONPATH=.` when running `pytest -q`

For empty payload runs, prefer explicit `context=[]` or `tools=[]` when calling
`compose(...).apply(...)`. A bare empty `payload=[]` is ambiguous under the
current unified-state API.
