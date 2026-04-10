# Quick Start

Get Kortex running in under 5 minutes.

## Install

```bash
pip install kortex-ai
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add kortex-ai
```

## Register a Provider

Kortex needs to know what models are available. You can register models manually or use auto-discovery.

### Option A: Local Model (Ollama)

No API key needed. Start Ollama, then:

```python
from kortex.core.router import ProviderModel, Router
from kortex.providers.base import GenericOpenAIConnector
from kortex.providers.registry import ProviderRegistry

registry = ProviderRegistry()
registry.register_openai_compatible(
    name="ollama",
    base_url="http://localhost:11434/v1",
    models=[ProviderModel(
        provider="ollama", model="llama3",
        cost_per_1k_input_tokens=0, cost_per_1k_output_tokens=0,
        avg_latency_ms=100, capabilities=["reasoning"], tier="fast",
    )],
)
```

### Option B: Cloud API (Auto-discovery)

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

Then let Kortex discover it:

```python
registry = ProviderRegistry()
registry.auto_discover()  # detects OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
```

## Create a Router and Register Models

```python
router = Router()
for model in registry.get_all_models():
    router.register_model(model)
```

If you registered models manually (without a registry), add them directly:

```python
router = Router()
router.register_model(ProviderModel(
    provider="anthropic", model="claude-sonnet-4-20250514",
    cost_per_1k_input_tokens=0.003, cost_per_1k_output_tokens=0.015,
    avg_latency_ms=800, capabilities=["reasoning", "content_generation"], tier="balanced",
))
router.register_model(ProviderModel(
    provider="openai", model="gpt-4o-mini",
    cost_per_1k_input_tokens=0.00015, cost_per_1k_output_tokens=0.0006,
    avg_latency_ms=200, capabilities=["reasoning"], tier="fast",
))
```

## Create Agents

Agents are descriptors — they tell the runtime what each step in your pipeline does:

```python
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager

runtime = KortexRuntime(router=router, state_manager=StateManager())

runtime.register_agent(AgentDescriptor(
    agent_id="researcher",
    name="Researcher",
    description="Gathers and analyzes information",
    capabilities=["reasoning", "research"],
))
runtime.register_agent(AgentDescriptor(
    agent_id="writer",
    name="Writer",
    description="Drafts content and documentation",
    capabilities=["content_generation"],
))
runtime.register_agent(AgentDescriptor(
    agent_id="reviewer",
    name="Reviewer",
    description="Reviews and provides feedback",
    capabilities=["reasoning", "analysis"],
))
```

Agent capabilities influence routing: when a task has no `required_capabilities`, the current agent's capabilities are used to filter models. This means a "code_generation" agent will be routed to models that support code generation.

### Capability Vocabulary

All capabilities must come from the canonical set:

`reasoning`, `analysis`, `code_generation`, `content_generation`, `vision`, `audio`, `quality_assurance`, `data_processing`, `planning`, `research`, `testing`

Common aliases like `"writing"`, `"coding"`, `"review"` are resolved automatically. Invalid values are rejected at registration with a helpful error message.

## Run Your First Coordination

```python
import asyncio
from kortex.core.types import TaskSpec

task = TaskSpec(
    content="Write an article about multi-agent AI systems",
    complexity_hint="complex",
    required_capabilities=["reasoning"],
)

result = asyncio.run(runtime.coordinate(
    task,
    agent_pipeline=["researcher", "writer", "reviewer"],
))
```

## Understand the Output

The `CoordinationResult` contains everything that happened:

```python
# What models were selected for each step
for d in result.routing_decisions:
    print(f"{d.chosen_provider}/{d.chosen_model} — {d.reasoning}")

# The handoff chain (checkpoint IDs for rollback)
for h in result.handoffs:
    print(f"{h.source_agent} -> {h.target_agent} [{h.checkpoint_id[:8]}]")

# Cost estimate
print(f"Estimated cost: ${result.total_estimated_cost_usd:.4f}")

# Human-readable summary
print(runtime.get_coordination_summary(result))
```

### Dry-run vs Live Execution

By default, `coordinate()` runs in **dry-run mode** (`execute=False`). It routes tasks and creates checkpoints but doesn't call any LLM APIs. This is useful for cost estimation and pipeline validation.

To actually call providers:

```python
runtime_with_registry = KortexRuntime(
    router=router, state_manager=StateManager(), registry=registry,
)
result = await runtime_with_registry.coordinate(task, pipeline, execute=True)

# Now result.responses contains actual LLM outputs
# and result.actual_cost_usd has the real cost
```

## Next Steps

- [Core Concepts](concepts.md) — understand routing, handoffs, and events in depth
- [Provider Setup](providers.md) — connect more providers and mix local + cloud
- [Framework Adapters](adapters.md) — integrate with LangGraph or CrewAI
