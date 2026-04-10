# Core Concepts

## What Kortex Does

Kortex sits between your agent framework and your LLM providers. It doesn't replace your framework — it adds routing, checkpointing, and anomaly detection to how tasks flow through your pipeline.

Without Kortex, your agents call models directly. The framework picks a model at build time and every task uses it, regardless of whether the task is simple or complex, cheap or expensive.

With Kortex, each sub-task is evaluated independently. The router picks the best model for that specific task based on cost, latency, and capabilities. State flows between agents through managed checkpoints. Every decision is logged as a structured event.

## Task Routing

### How the Router Picks Models

The built-in `HeuristicRoutingStrategy` uses rule-based selection — no ML, no learned weights. It evaluates a `TaskSpec` against all registered `ProviderModel` instances:

1. **Filter** — remove models that violate constraints:
    - Cost ceiling: `estimated_cost > task.cost_ceiling_usd`
    - Latency SLA: `avg_latency_ms > task.latency_sla_ms`
    - Capabilities: model must have all `task.required_capabilities`

2. **Select by complexity tier** — match the task's `complexity_hint` to model tiers:
    - `simple` → cheapest fast-tier model
    - `moderate` → cheapest balanced-tier model
    - `complex` → most capable powerful-tier model

3. **Fallback** — the next-best model (by cost) is recorded as a fallback in case the primary fails.

### ProviderModel

Each model is described by a `ProviderModel` with pricing, latency, capabilities, and a tier:

```python
ProviderModel(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    cost_per_1k_input_tokens=0.003,
    cost_per_1k_output_tokens=0.015,
    avg_latency_ms=800,
    capabilities=["reasoning", "code_generation", "content_generation"],
    tier="balanced",  # "fast", "balanced", or "powerful"
)
```

Models are identified by composite keys (`provider::model_name`) to prevent collisions when different providers offer models with the same name.

### Capabilities

Capabilities use a canonical vocabulary enforced at registration. Valid values:

`reasoning`, `analysis`, `code_generation`, `content_generation`, `vision`, `audio`, `quality_assurance`, `data_processing`, `planning`, `research`, `testing`

Common aliases are resolved automatically: `"writing"` → `"content_generation"`, `"coding"` → `"code_generation"`, `"review"` → `"quality_assurance"`, etc.

Free-form strings are rejected with a `ValueError` and a "did you mean?" suggestion.

### Custom Routing Strategies

You can replace the built-in heuristic strategy by implementing the `RoutingStrategy` protocol:

```python
class MyStrategy:
    async def select(self, task: TaskSpec, candidates: list[ProviderModel]) -> RoutingDecision:
        # Your logic here
        ...

router = Router(strategy=MyStrategy())
```

## Stateful Handoffs

### How Context Flows Between Agents

When Agent A finishes and Agent B starts, Kortex creates a `HandoffContext`:

- **State snapshot** — the boundary-crossing payload (task data, model output, metadata)
- **Compressed summary** — a token-efficient summary for large payloads
- **Checkpoint ID** — unique identifier for this point in time
- **Parent checkpoint ID** — links to the previous checkpoint, forming a chain

This chain is persisted in the state store (in-memory, SQLite, or Redis).

### Storage Backends

```python
from kortex.core.state import StateManager

# In-memory (default, for testing)
state = StateManager.create("memory")

# SQLite (local development, durable)
state = StateManager.create("sqlite", db_path="kortex.db")

# Redis (production, hot state)
state = StateManager.create("redis", redis_url="redis://localhost:6379")
```

## Checkpoint Chains and Rollback

Checkpoints form a linked list. Each checkpoint points to its parent:

```
[input] -> [researcher] -> [writer] -> [reviewer]
   c1           c2             c3           c4
```

To roll back to any point:

```python
context = await runtime.rollback_to(checkpoint_id)
# Returns the HandoffContext at that checkpoint
```

To get the full chain from root to a specific point:

```python
chain = await state_manager.get_history(checkpoint_id)
# Returns [c1, c2, c3] ordered root-first
```

## Anomaly Detection and Recovery

### Detection

The `FailureDetector` monitors the pipeline using configurable thresholds (`DetectionPolicy`):

| Check | What It Detects |
|-------|-----------------|
| `check_routing` | Cost overruns (actual vs estimated) |
| `check_execution` | Latency spikes, output quality drops (content too short) |
| `check_handoff` | Context degradation (snapshot shrunk >80%), checkpoint chain too deep |
| `check_coordination` | Aggregate anomalies across the full pipeline |

Each anomaly is reported as an `AnomalyReport` with a severity (`low`, `medium`, `high`, `critical`) and a recommended action.

### Recovery

The `RecoveryExecutor` translates anomaly recommendations into actions:

| Action | What Happens |
|--------|-------------|
| `continue` | Log the anomaly and proceed |
| `retry` | Re-route and optionally re-execute the current step (budget: 1 per step, 3 total) |
| `fallback` | Use the fallback model from the routing decision |
| `rollback` | Restore checkpoint from the parent checkpoint |
| `escalate` | Stop the pipeline |

Recovery is configurable via `RecoveryPolicy` and runs automatically during `coordinate()` when a detector is attached.

## Execution Events

Every action in the pipeline emits an `ExecutionEvent`:

| Event Type | When |
|------------|------|
| `route` | Router selects a model for a task |
| `handoff` | Context is passed between agents |
| `failure` | A routing or execution error occurs |
| `recovery` | System recovers from a failure |
| `completion` | Pipeline finishes successfully |

Events are attached to the `CoordinationResult`:

```python
for event in result.events:
    print(f"[{event.event_type}] agent={event.agent_id} payload={event.payload}")
```

These events are structured data. They are not sent anywhere automatically — they live on the `CoordinationResult` object for you to log, store, or analyze however you choose.

## Dry-Run vs Live Execution

- **`execute=False`** (default) — routes tasks and creates checkpoints, but makes no LLM API calls. Use for cost estimation and pipeline validation.
- **`execute=True`** — actually calls the LLM providers. Requires a `ProviderRegistry` on the runtime. Populates `result.responses` and `result.actual_cost_usd`.
