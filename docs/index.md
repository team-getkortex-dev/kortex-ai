# Kortex

**Routing and replay for multi-model agent workflows.**

Route tasks across models, persist handoff checkpoints, and inspect failures — without replacing your existing agent framework.

## The Problem

Multi-agent AI systems are powerful in demos and fragile in production. Three things consistently break:

1. **Context loss** — when Agent A hands off to Agent B, critical context gets dropped. There's no checkpoint system, no rollback, no way to recover.
2. **No cost control** — every task hits your most expensive model because there's no routing logic. A simple summarization task shouldn't cost the same as deep analysis.
3. **Silent failures** — agents fail mid-pipeline and nobody knows until the output is garbage. No structured event stream, no anomaly detection, no recovery.

## What Kortex Does

Kortex is middleware that sits between your agent framework (LangGraph, CrewAI, or custom) and your LLM providers. It handles three things:

### Heuristic Task Routing

The router evaluates each sub-task against available models using rules: cost ceilings, latency SLAs, required capabilities, and complexity tiers. Simple tasks go to fast/cheap models. Complex tasks go to powerful models. You set constraints and the router picks the best match.

This is rule-based selection, not ML. There are no learned weights or training loops. The `RoutingStrategy` protocol is pluggable — you can replace the built-in `HeuristicRoutingStrategy` with your own logic.

### Stateful Handoffs

Every agent-to-agent handoff creates a checkpoint with the state snapshot. Checkpoints form chains — you can roll back to any point in the execution history. Backends: in-memory, SQLite, Redis.

### Anomaly Detection and Recovery

A threshold-based detector monitors routing decisions, execution responses, and handoffs for cost overruns, latency spikes, output quality drops, and context degradation. When an anomaly is detected, a recovery executor can retry, fall back to another model, roll back to a checkpoint, or escalate.

## Current Scope

- **Heuristic routing** — cost, latency, capability, and complexity-tier policies. No ML.
- **Checkpoint persistence** — rollback and history across in-memory, SQLite, and Redis backends.
- **Threshold-based anomaly detection** — with real recovery actions (retry, fallback, rollback, escalate).
- **Framework adapters** — drop-in wrappers for LangGraph and CrewAI.
- **Provider support** — OpenAI, Anthropic, OpenRouter, and any OpenAI-compatible endpoint (Ollama, vLLM, etc.).
- **CLI** — terminal commands for inspecting providers, models, routing decisions, and checkpoint history.

## Non-Goals (Today)

- **Not a learned/adaptive router.** Every routing decision is logged as a structured event — this data could train a learned router in the future, but that doesn't exist yet.
- **Not an observability platform.** Events are attached to coordination results. There is no built-in dashboard, alerting, or metrics pipeline.
- **Not semantic state compression.** Snapshots are stored as-is.

## Quick Start

```python
import asyncio
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec

router = Router()
router.register_model(ProviderModel(
    provider="anthropic", model="claude-sonnet-4-20250514",
    cost_per_1k_input_tokens=0.003, cost_per_1k_output_tokens=0.015,
    avg_latency_ms=800, capabilities=["reasoning", "content_generation"], tier="balanced",
))

runtime = KortexRuntime(router=router, state_manager=StateManager())
runtime.register_agent(AgentDescriptor("researcher", "Researcher", "Gathers info"))
runtime.register_agent(AgentDescriptor("writer", "Writer", "Drafts content"))

task = TaskSpec(content="Write about AI coordination", complexity_hint="moderate")
result = asyncio.run(runtime.coordinate(task, ["researcher", "writer"]))
print(runtime.get_coordination_summary(result))
```

## Design Principles

- **Framework-agnostic** — Kortex wraps existing frameworks, never replaces them.
- **Fail-open** — if Kortex is unavailable, agents fall back to direct execution.
- **Any provider** — OpenAI, Anthropic, Ollama, vLLM, or any OpenAI-compatible API.
- **Minimal integration** — adding Kortex to an existing project requires <20 lines of code.

## Next Steps

- [Quick Start Guide](quickstart.md) — step-by-step setup
- [Core Concepts](concepts.md) — understand routing, handoffs, and events
- [Provider Setup](providers.md) — connect your LLM providers
- [Framework Adapters](adapters.md) — integrate with LangGraph or CrewAI
- [CLI Reference](cli.md) — inspect your system from the terminal
