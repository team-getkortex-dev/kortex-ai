![Tests](https://img.shields.io/github/actions/workflow/status/team-getkortex-dev/kortex-ai/test.yml?label=tests)
![PyPI](https://img.shields.io/pypi/v/kortex-ai)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

# Kortex AI

**Production-grade multi-agent coordination runtime.**

Kortex sits between your agent framework (LangGraph, CrewAI, or custom) and your LLM providers. It handles routing, checkpointing, caching, anomaly detection, and cost control — without replacing your framework.

## Key Features

| Feature | What it does |
|---------|-------------|
| **Semantic caching** | 3500x speedup on repeated queries via xxhash-keyed result cache |
| **Constraint-based routing** | Provably correct model selection from cost, latency, capability, and tier constraints |
| **Self-calibrating costs** | Adaptive EWMA with bootstrap cold-start, outlier rejection, and P50/P95/P99 tracking |
| **Time-travel debugging** | Checkpoint every handoff; roll back to any point with deterministic replay |
| **Auto-optimization** | A/B testing and policy search find better routing configurations automatically |
| **Cost arbitrage** | Cross-provider cost comparison routes tasks to the cheapest capable model |

## Quick Start

```bash
pip install kortex-ai
```

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
router.register_model(ProviderModel(
    provider="openai", model="gpt-4o-mini",
    cost_per_1k_input_tokens=0.00015, cost_per_1k_output_tokens=0.0006,
    avg_latency_ms=250, capabilities=["reasoning"], tier="fast",
))

runtime = KortexRuntime(router=router, state_manager=StateManager())
runtime.register_agent(AgentDescriptor("researcher", "Researcher", "Gathers info"))
runtime.register_agent(AgentDescriptor("writer", "Writer", "Drafts content"))

task = TaskSpec(content="Write about AI coordination", complexity_hint="moderate")
result = asyncio.run(runtime.coordinate(task, ["researcher", "writer"]))
print(runtime.get_coordination_summary(result))
```

## Performance

| Metric | Value | Methodology |
|--------|-------|-------------|
| Cache speedup | **3506x** | 100 repeated queries; cache hit vs. cold routing path |
| Cost reduction | **84%** | Kortex heuristic router vs. always-strongest-model baseline |
| Overall speedup | **5–10x** | Mixed workload (40% simple / 35% moderate / 25% complex) |
| EWMA convergence | **10 samples** | Bootstrap mean transitions to adaptive decay at sample 11 |

Reproduce with: `python examples/benchmark_example.py`

## CLI (18 commands)

```bash
kortex status                            # providers, models, agents
kortex models                            # cost/latency table
kortex dry-run --task "..." --pipeline researcher,writer
kortex trace list && kortex trace show <id>
kortex replay <trace_id> --policy policies/cost_optimized.toml
kortex benchmark run --dataset mixed
kortex policy diff <trace_id> --policy policies/quality_first.toml
```

Works in demo mode — no API keys required.

## Works With Any Provider

```python
# Local models (Ollama, vLLM, LM Studio)
from kortex.providers.base import GenericOpenAIConnector
ollama = GenericOpenAIConnector(base_url="http://localhost:11434/v1", ...)

# Cloud (auto-discover from env vars)
from kortex.providers.registry import ProviderRegistry
registry = ProviderRegistry()
registry.auto_discover()   # reads OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
```

## Framework Adapters

```python
# LangGraph
from kortex.adapters.langgraph import KortexLangGraphAdapter
wrapped = KortexLangGraphAdapter(runtime).wrap_graph(graph, node_map)

# CrewAI
from kortex.adapters.crewai import KortexCrewAIAdapter
wrapped = KortexCrewAIAdapter(runtime).wrap_crew(crew, agent_map)
```

## Installation

```bash
pip install kortex-ai            # core
pip install kortex-ai[otel]      # + OpenTelemetry export
pip install kortex-ai[ml]        # + learned router (scikit-learn)
```

## Documentation

[docs.getkortex.dev](https://docs.getkortex.dev)

## Examples

| Script | What it shows |
|--------|--------------|
| `examples/basic_usage.py` | Minimal two-model setup and single task |
| `examples/multi_agent_pipeline.py` | writer → reviewer → editor with checkpoints |
| `examples/streaming_example.py` | `stream_coordinate()` event-by-event |
| `examples/custom_policy.py` | Cost, latency, and quality-first policies compared |
| `examples/cache_demo.py` | 10-repeat cache speedup measurement |
| `examples/benchmark_example.py` | Full benchmark vs. static baselines |
| `examples/langgraph_example.py` | LangGraph adapter with mocked HTTP |
| `examples/crewai_example.py` | CrewAI adapter with capability inference |

All examples run without API keys.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE) — Copyright 2026 Liam Robinson and Kaden Audin
