# Tokentrim Design Doc

## Scope

This document describes Tokentrim's current architecture, boundaries, and
extension model.

## Core Goals

- local, in-process SDK (no hosted Tokentrim service)
- one public execution pattern: `compose(...).apply(...)`
- explicit transform composition
- inspectable runs through a stable trace model
- thin integration adapters that reuse the same core pipeline

## Public API

Tokentrim exposes two user-facing objects:

- `Tokentrim` in `tokentrim/client.py`
- `ComposedPipeline` returned by `Tokentrim.compose(...)`

Execution shape:

1. create client with shared defaults (tokenizer, default token budget)
2. compose transform instances
3. call `.apply(payload, ...)`
4. receive one immutable `Result`

There are no separate context/tools public runner methods.

For OpenAI Agents users, the primary integration path is:

- `Tokentrim.compose(...).to_openai_agents(...)`

This keeps integration wiring compose-first and avoids direct
adapter/options construction in app code.

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
- `run(payload, request)`: pure transform step

`kind` is used internally to prevent invalid composition and wiring mistakes
(for example mixing context and tools steps).

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

## Integration Boundary

`IntegrationAdapter[ConfigT]` defines the integration contract:

- adapters map SDK-native payloads into Tokentrim payloads
- adapters call `Tokentrim.compose(...).apply(...)`
- adapters map optimized payloads back into SDK-native shape

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
2. add `transform.py` and optional `error.py`
3. implement `Transform` (`name`, `kind`, `run`, optional `resolve`)
4. export from `tokentrim/transforms/__init__.py` if public
5. add mirrored tests under `tests/tokentrim/transforms/<name>/`
