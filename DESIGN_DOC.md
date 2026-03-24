# Tokentrim Design Doc

## Scope

This document explains the package architecture of Tokentrim, why the core
abstractions are shaped the way they are, how the package is intended to be
used, and how to extend it by adding new context and tool steps.

It is intentionally focused on architecture. It does not explain the behavior
of individual built-in step implementations in detail.

## Goals

- Keep Tokentrim as a local library, not a service.
- Make context and tool optimization explicit and composable.
- Keep the core independent from any single agent SDK.
- Provide a small set of stable extension points.
- Make behavior inspectable through structured results and traces.

## Non-Goals

- Tokentrim is not an agent runtime.
- Tokentrim is not a workflow engine.
- Tokentrim does not define a universal schema for every rich multimodal SDK
  payload.
- Tokentrim does not automatically optimize everything by default. The caller
  opts into concrete step objects.

## High-Level Architecture

Tokentrim has five layers:

1. `Tokentrim` is the public facade.
2. Request objects capture a single optimization operation.
3. Pipelines execute ordered step objects.
4. Result objects report the transformed payload and per-step traces.
5. Integration adapters attach Tokentrim to external SDK hook systems.

There are two parallel optimization tracks:

- Context optimization, which operates on `Message` values.
- Tool optimization, which operates on `Tool` values.

Those tracks share the same architectural pattern but stay separate so that
their APIs, defaults, validation, and future evolution do not get coupled.

## Package Map

| Path | Responsibility |
| --- | --- |
| `tokentrim/client.py` | Public facade and default dependency wiring |
| `tokentrim/context/base.py` | `ContextStep` abstraction |
| `tokentrim/context/request.py` | Immutable input for one context run |
| `tokentrim/context/pipeline.py` | Ordered execution of context steps |
| `tokentrim/context/store.py` | Retrieval boundary used by context steps |
| `tokentrim/tools/base.py` | `ToolStep` abstraction |
| `tokentrim/tools/request.py` | Immutable input for one tool run |
| `tokentrim/tools/pipeline.py` | Ordered execution of tool steps |
| `tokentrim/integrations/base.py` | Abstract adapter contract |
| `tokentrim/integrations/openai_agents.py` | Example SDK adapter |
| `tokentrim/types/*` | Minimal payload, result, and trace types |
| `tokentrim/_copy.py` | Defensive cloning and freezing helpers |
| `tokentrim/_tokens.py` | Token estimation and counting |

## Core Data Model

Tokentrim uses intentionally small core payload types:

- `Message` has `role` and `content`.
- `Tool` has `name`, `description`, and `input_schema`.

### Rationale

- The core should stay generic and easy to reason about.
- Most optimization logic only needs a minimal textual representation.
- Vendor-specific payload shapes belong at the integration boundary, not in the
  heart of the package.

### Tradeoff

This design keeps the core simple, but it means rich SDK payloads must be
adapted at the edges. Some integrations may choose to preserve unsupported
payloads unchanged instead of forcing them into the core schema.

## Public Facade: `Tokentrim`

`Tokentrim` is the main entrypoint for application code. It does three things:

- owns default dependencies such as tokenizer model names and stores,
- creates immutable request objects from caller input,
- delegates execution to the correct pipeline.

Its public methods are:

- `get_better_context(...)`
- `get_better_tools(...)`
- `wrap_integration(...)`

### Rationale

- Application code should not need to instantiate pipelines directly.
- Shared defaults should be configured once and reused across calls.
- A single facade gives the package one obvious place to start without forcing
  every feature into one giant class implementation.

## Requests

Each pipeline run is represented by a frozen request object:

- `ContextRequest`
- `ToolsRequest`

These requests carry:

- the input payload,
- run-specific metadata such as `user_id`, `session_id`, or `task_hint`,
- the budget for that run,
- the ordered tuple of step objects to execute.

### Rationale

- Freezing request objects makes the execution contract explicit.
- Pipelines and steps can rely on a stable input model.
- Request objects separate caller intent from internal execution state.

## Steps

The primary extension point in Tokentrim is the step object.

- `ContextStep` transforms a list of messages.
- `ToolStep` transforms a list of tools.

Each step has:

- a stable `name`,
- an optional `resolve(...)` phase,
- a `run(...)` method that returns a transformed list.

### Why Steps Are Objects

Tokentrim originally could have expressed step selection with booleans or string
identifiers. The object-based design is more extensible for several reasons:

- Step configuration can live on the step itself instead of being spread across
  facade methods, request types, and pipelines.
- Different instances of the same step class can carry different parameters.
- New steps can be added without expanding a central list of feature flags.
- The caller can control order explicitly by the order of the objects passed in.

This is why the intended usage is:

```python
steps = (
    SomeContextStep(...),
    AnotherContextStep(...),
)
```

instead of:

```python
steps = ("step_a", "step_b")
```

or:

```python
enable_step_a=True
enable_step_b=True
```

## Step Resolution

Both abstract step types expose a `resolve(...)` method before execution.

`resolve(...)` exists so a step can be configured partly by the caller and
partly by Tokentrim defaults. For example, a step may omit a model name at
construction time and bind it later from the `Tokentrim` instance that owns the
pipeline.

### Rationale

- The public step object stays focused on behavior-specific parameters.
- Shared defaults stay centralized in `Tokentrim`.
- The same step class can work both with explicit per-step configuration and
  with client-level defaults.

### Design Rule

`resolve(...)` should return an executable step object for the current run.
Usually that means returning `self` if nothing needs to change, or returning a
new instance with defaults filled in.

## Pipelines

Tokentrim has two pipeline classes:

- `ContextPipeline`
- `ToolsPipeline`

The pipeline algorithm is deliberately simple:

1. Clone the input payload.
2. Iterate over the requested step objects in order.
3. Resolve each step against client defaults.
4. Run the step.
5. Record a `StepTrace`.
6. Count final tokens.
7. Enforce the final budget invariant.
8. Freeze the output into a result object.

### Rationale

- Ordered execution keeps behavior explicit.
- Cloning prevents hidden mutation of caller-owned objects.
- Per-step traces make the pipeline observable without coupling results to any
  one concrete step type.
- Final budget enforcement gives the caller one simple guarantee: if a result is
  returned, it is within the requested budget.

## Copy and Freeze Boundaries

Tokentrim clones incoming messages and tools, and freezes outgoing results into
tuples.

### Rationale

- Steps should not accidentally mutate caller input.
- Results should be safe to cache, log, and compare.
- The library should behave like a pure transformation pipeline from the
  caller's perspective even if individual steps do complex internal work.

## Results and Traces

Each run returns a frozen result object:

- `ContextResult`
- `ToolsResult`

Each result includes:

- the final payload,
- `token_count`,
- `trace_id`,
- `step_traces`.

`StepTrace` is intentionally generic. It reports:

- `step_name`
- `input_count`
- `output_count`
- `changed`

### Rationale

- Results should be stable and easy to inspect.
- A generic trace shape avoids baking concrete step semantics into the public
  API.
- The trace model gives users enough visibility to debug pipeline behavior
  without forcing them to understand internal implementation details.

## Budget Enforcement

Budget checks happen after the pipeline finishes.

### Rationale

- Each step should focus on its own transformation logic.
- A single final check is simpler than requiring every step to duplicate budget
  logic.
- The final result is the only state that matters for the public invariant.

### Consequence

Steps may choose to consult the requested budget while running, but the pipeline
is the final authority that either accepts or rejects the result.

## Integration Adapters

The integration layer exists so Tokentrim can attach to external SDKs without
making the core depend on any specific one.

`IntegrationAdapter[ConfigT]` is the adapter contract. An adapter takes a
`Tokentrim` instance plus optional integration-specific configuration and returns
the wrapped SDK config or object.

### Rationale

- The core package should not import every supported agent SDK.
- Different SDKs expose different hook models, callback signatures, and payload
  shapes.
- A dedicated adapter layer isolates those differences from the pipeline core.

### Architectural Rule

Adapters should be thin. They are responsible for:

- translating SDK payloads into Tokentrim core payloads when possible,
- calling the appropriate Tokentrim entrypoint,
- translating the result back into SDK-native structures,
- preserving unsupported SDK-native payloads conservatively when lossless
  conversion is not possible.

The core is not supposed to know how a specific SDK represents a handoff, a
tool call, or a multimodal block. That logic belongs in the adapter.

## End-to-End Flow

### Direct Context Usage

```text
Application code
  -> Tokentrim.get_better_context(...)
  -> ContextRequest
  -> ContextPipeline
  -> resolve step 1
  -> run step 1
  -> resolve step 2
  -> run step 2
  -> final token count
  -> ContextResult
```

### Direct Tool Usage

```text
Application code
  -> Tokentrim.get_better_tools(...)
  -> ToolsRequest
  -> ToolsPipeline
  -> resolve step 1
  -> run step 1
  -> resolve step 2
  -> run step 2
  -> final token count
  -> ToolsResult
```

### SDK Adapter Usage

```text
Application code
  -> Tokentrim.wrap_integration(adapter, config=...)
  -> adapter installs SDK hooks
  -> SDK invokes hook
  -> adapter maps SDK payload to Tokentrim payload
  -> Tokentrim pipeline runs
  -> adapter maps result back to SDK payload
  -> SDK continues normally
```

## Usage Patterns

### Direct Context Optimization

```python
from tokentrim import Tokentrim
from tokentrim.context import ContextStep


tt = Tokentrim(
    model="gpt-4o-mini",
    token_budget=8000,
)

result = tt.get_better_context(
    messages,
    user_id="user-123",
    session_id="session-456",
    steps=(
        MyContextStep(...),
    ),
)

optimized_messages = result.messages
```

### Direct Tool Optimization

```python
from tokentrim import Tokentrim
from tokentrim.tools import ToolStep


tt = Tokentrim(
    model="gpt-4o-mini",
    token_budget=8000,
)

result = tt.get_better_tools(
    tools,
    task_hint="handle the current user task",
    steps=(
        MyToolStep(...),
    ),
)

optimized_tools = result.tools
```

### Integration Usage

```python
from tokentrim import Tokentrim
from tokentrim.integrations import OpenAIAgentsAdapter, OpenAIAgentsOptions


tt = Tokentrim(
    model="gpt-4o-mini",
    token_budget=8000,
)

run_config = tt.wrap_integration(
    OpenAIAgentsAdapter(
        options=OpenAIAgentsOptions(
            token_budget=8000,
            steps=(
                MyContextStep(...),
            ),
        )
    )
)
```

The important point is architectural, not SDK-specific: integrations are just
another way to invoke the same core pipeline.

## Extending Tokentrim With a New `ContextStep`

Add a new class that subclasses `ContextStep` and import it from
`tokentrim.context` if it should be public.

### Step Contract

A context step should:

- expose a stable `name`,
- accept its own configuration through the constructor,
- optionally implement `resolve(...)` if it needs client defaults,
- implement `run(messages, request) -> list[Message]`.

### Skeleton

```python
from __future__ import annotations

from dataclasses import dataclass, replace

from tokentrim.context import ContextStep
from tokentrim.context.request import ContextRequest
from tokentrim.types.message import Message


@dataclass(frozen=True, slots=True)
class MyContextStep(ContextStep):
    threshold: int = 10
    model: str | None = None

    @property
    def name(self) -> str:
        return "my_context_step"

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
        compaction_model: str | None = None,
        memory_store=None,
    ) -> ContextStep:
        del tokenizer_model
        del memory_store
        return replace(
            self,
            model=self.model if self.model is not None else compaction_model,
        )

    def run(self, messages: list[Message], request: ContextRequest) -> list[Message]:
        del request
        return list(messages)
```

### Guidance

- Prefer frozen dataclasses with `slots=True`. That matches the rest of the
  package and makes step objects lightweight and predictable.
- Treat the input list as read-only and return a new list.
- Use `request` for run-scoped information such as budgets or identifiers.
- Use `resolve(...)` for client-level defaults, not for run-specific logic.
- Raise `TokentrimError` for invalid configuration or unrecoverable step
  failures.
- Keep the `name` stable. It is part of trace output and may appear in user
  logs.

### When to Use Constructor Parameters vs `request`

Use constructor parameters for behavior that is part of the step's identity:

- thresholds,
- model overrides,
- mode flags,
- policy choices.

Use `request` for behavior that belongs to a single pipeline run:

- the actual messages,
- the active token budget,
- user or session metadata.

### Ordering Considerations

Context steps execute in the exact order provided by the caller. New steps
should therefore be designed to compose cleanly with other steps before and
after them.

Questions to ask when adding a step:

- Does it expect raw input or already-transformed input?
- Does it remove information or only annotate/reshape it?
- Is it safe to run more than once?
- Does it depend on a budget being present?

## Extending Tokentrim With a New `ToolStep`

Add a new class that subclasses `ToolStep` and import it from `tokentrim.tools`
if it should be public.

### Step Contract

A tool step should:

- expose a stable `name`,
- accept its own configuration through the constructor,
- optionally implement `resolve(...)` if it needs client defaults,
- implement `run(tools, request) -> list[Tool]`.

### Skeleton

```python
from __future__ import annotations

from dataclasses import dataclass, replace

from tokentrim.tools import ToolStep
from tokentrim.tools.request import ToolsRequest
from tokentrim.types.tool import Tool


@dataclass(frozen=True, slots=True)
class MyToolStep(ToolStep):
    model: str | None = None

    @property
    def name(self) -> str:
        return "my_tool_step"

    def resolve(
        self,
        *,
        tokenizer_model: str | None = None,
        tool_creation_model: str | None = None,
    ) -> ToolStep:
        del tokenizer_model
        return replace(
            self,
            model=self.model if self.model is not None else tool_creation_model,
        )

    def run(self, tools: list[Tool], request: ToolsRequest) -> list[Tool]:
        del request
        return list(tools)
```

### Guidance

- Return the full transformed tool list, not only a delta.
- Preserve valid tool structure in every returned item.
- Use `request.task_hint` only if the step actually needs task-scoped context.
- Raise `TokentrimError` when a step cannot complete safely.
- Keep output deterministic where practical. Tool optimization affects model
  behavior directly and benefits from predictability.

## Testing New Steps

Every new step should have unit tests for:

- the happy path,
- invalid configuration,
- no-op behavior when its preconditions are not met,
- interaction with request metadata it relies on,
- any model/store default binding done in `resolve(...)`.

In addition to the step's own tests, add a pipeline-level test if the step has
important ordering or budget interactions.

## Adding a New Integration Adapter

If Tokentrim is integrated with another SDK, add a new class that implements
`IntegrationAdapter[ConfigT]`.

### Adapter Responsibilities

- accept integration-specific options,
- attach to the SDK's hook or middleware system,
- convert compatible SDK payloads to Tokentrim payloads,
- call the correct Tokentrim facade method,
- convert results back to SDK-native payloads,
- preserve unsupported payloads conservatively.

### Design Guidance

- Keep adapters thin. Optimization logic belongs in steps and pipelines.
- Do not leak SDK-specific types into the core context or tool abstractions.
- Prefer fail-safe behavior over lossy conversion when an SDK payload cannot be
  represented by Tokentrim's minimal core schema.

## Why Context and Tools Stay Separate

Tokentrim could have modeled everything as one generalized "optimization
pipeline," but that would make the core more abstract without improving the
caller experience.

The context and tool tracks differ in:

- input schema,
- metadata needs,
- token counting strategy,
- typical transformation patterns,
- likely future integrations.

Keeping them separate provides cleaner APIs while still sharing the same
architectural shape.

## Current Architectural Constraints

These are deliberate boundaries of the current design:

- The public API is synchronous.
- The core payload model is intentionally minimal.
- Budget enforcement is final-result based.
- Rich SDK-native payload support depends on adapter mapping quality.

These constraints keep the package small and understandable. They are also the
main areas where future evolution could happen if the package grows.

## Summary

Tokentrim is built around a simple idea:

- the public facade wires defaults,
- frozen requests describe one run,
- object-based steps express behavior,
- pipelines execute steps in order and enforce invariants,
- result objects expose outputs and traces,
- adapters connect external SDKs without coupling the core to any one of them.

That structure is what makes the package usable today and extensible for future
context steps, tool steps, and SDK integrations.
