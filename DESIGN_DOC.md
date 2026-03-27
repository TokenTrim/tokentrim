# Tokentrim Design Doc

## Scope

This document describes Tokentrim's current architecture, boundaries, and
extension model.

## Core Goals

- local, in-process SDK (no hosted Tokentrim service)
- one primary execution pattern: `compose(...).apply(...)`
- explicit transform composition
- inspectable runs through a stable trace model
- thin integration adapters that reuse the same core pipeline

## Public API

Tokentrim exposes two primary user-facing objects:

- `Tokentrim` in `tokentrim/client.py`
- `ComposedPipeline` returned by `Tokentrim.compose(...)`

Execution shape:

1. create client with shared defaults (tokenizer, default token budget)
2. compose transform instances
3. call `.apply(payload, ...)`
4. receive one immutable `Result`

There are no separate context/tools runner types. Both payload kinds use the
same compose-first execution path.

For OpenAI Agents users, the primary integration path is:

- `Tokentrim.compose(...).to_openai_agents(...)`

This keeps integration wiring compose-first and avoids direct
adapter/options construction in app code.

Additional public helpers exist on `Tokentrim`:

- `wrap_integration(adapter, config=...)` for generic adapter wiring
- `openai_agents_config(...)` as a convenience wrapper around
  `compose(*steps).to_openai_agents(...)`

## Package Layout

| Path | Responsibility |
| --- | --- |
| `tokentrim/client.py` | Public facade and compose-first API |
| `tokentrim/pipeline/requests.py` | Immutable request models (`ContextRequest`, `ToolsRequest`) |
| `tokentrim/pipeline/pipeline.py` | Unified execution runtime (`UnifiedPipeline`) |
| `tokentrim/transforms/base.py` | Shared transform contract |
| `tokentrim/transforms/filter/` | Message filtering transform |
| `tokentrim/transforms/compaction/` | Conversation compaction transform |
| `tokentrim/transforms/rlm/` | Retrieval-memory transform + memory store interface |
| `tokentrim/transforms/compress_tools/` | Deterministic tool description compression |
| `tokentrim/transforms/create_tools/` | Model-backed missing tool creation |
| `tokentrim/core/copy_utils.py` | Clone/freeze helpers for payload safety |
| `tokentrim/core/token_counting.py` | Token counting helpers |
| `tokentrim/core/llm_client.py` | LiteLLM wrapper used by model-backed transforms |
| `tokentrim/types/` | Shared datatypes (`Message`, `Tool`, `Trace`, `StepTrace`, `Result`) |
| `tokentrim/integrations/base.py` | Integration adapter contract |
| `tokentrim/integrations/openai_agents/` | OpenAI Agents adapter modules |
| `tokentrim/errors/` | Shared error taxonomy |

## Types

Tokentrim uses minimal payload types:

- `Message`: `{role, content}`
- `Tool`: `{name, description, input_schema}`

Every run returns one `Result`:

- `trace: Trace`
- `context: tuple[Message, ...] | None`
- `tools: tuple[Tool, ...] | None`

Exactly one of `context` or `tools` is populated per run.

## Transform Contract

All transforms implement `Transform` (`tokentrim/transforms/base.py`):

- `name`: stable identifier for tracing
- `kind`: payload track (`"context"` or `"tools"`)
- optional `resolve(tokenizer_model=...)`: binding phase before run
- `run(payload, request)`: transform step for one payload kind

`kind` is used internally to prevent invalid composition and wiring mistakes
(for example mixing context and tools steps).

Transform expectations:

- context transforms operate on `list[Message]` and receive `ContextRequest`
- tools transforms operate on `list[Tool]` and receive `ToolsRequest`
- transforms should return a new logical payload and must not rely on mutating
  caller-owned input state
- `resolve(...)` is the place to bind tokenizer-dependent or model-dependent
  configuration before execution

## Pipeline Runtime

`UnifiedPipeline.run(request)` handles both request types with shared internal
execution (`_run_payload`):

1. clone inbound payload
2. resolve each step
3. execute step in order
4. compute per-step token/item deltas
5. enforce final token budget
6. freeze result payload
7. return immutable `Result`

Budget enforcement is centralized at pipeline exit.

Tracing is also centralized in the pipeline. Each executed step produces one
`StepTrace` entry with item counts, token counts, and a changed/not-changed
flag.

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
- `adapter.py`: orchestrating adapter entrypoint

## Error Model

Tokentrim uses shared root errors in `tokentrim/errors/` and per-transform
errors inside each transform package (`error.py`).

Examples:

- shared: `TokentrimError`, `BudgetExceededError`
- transform-local: configuration/execution/output errors where needed

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
   - correct `kind` (`"context"` or `"tools"`)
   - a `run(...)` signature that matches that kind's request type
   - optional `resolve(...)` if configuration must be bound at runtime
6. keep the transform focused on payload transformation; cloning, freezing,
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
- transform tests should cover happy path behavior and any transform-local
  failure modes

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

## Contributor Notes

For local test runs, install the package in editable mode or set
`PYTHONPATH=.` before running pytest. CI installs with:

- `pip install -e ".[dev,openai-agents]"`
- `PYTHONPATH=.` when running `pytest -q`
