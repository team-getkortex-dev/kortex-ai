# Kortex Examples

All examples run without API keys — HTTP calls are mocked or skipped entirely.

## Examples

### Basic Usage

```bash
python examples/basic_usage.py
```

Minimal runtime setup: two models, one agent, one task. Shows routing decision output and cost estimate. Good starting point for new integrations.

### Multi-Agent Pipeline

```bash
python examples/multi_agent_pipeline.py
```

Three-agent pipeline: `writer → reviewer → editor`. Demonstrates checkpoint persistence, the handoff chain, and how to retrieve checkpoint IDs for rollback.

### Streaming

```bash
python examples/streaming_example.py
```

Uses `stream_coordinate()` to consume an async event stream. Events are yielded as they occur — routing decisions, handoff checkpoints, tokens (when supported by the provider), and a completion summary.

### Custom Policy

```bash
python examples/custom_policy.py
```

Compares three routing policies on the same task:

| Policy | Constraint | Objective |
|--------|-----------|-----------|
| `cost_optimised` | `max_cost_usd=0.005` | `minimize="cost"` |
| `latency_optimised` | `max_latency_ms=400` | `minimize="latency"` |
| `quality_first` | `required_capabilities=["deep_analysis"]` | `prefer_tier="powerful"` |

### Cache Demo

```bash
python examples/cache_demo.py
```

Issues the same task 10 times against a `SemanticCache` backed by `MemoryCache`. Prints per-call latency, hit/miss label, and final speedup ratio.

### LangGraph Integration

```bash
python examples/langgraph_example.py
```

Three-agent pipeline (researcher → writer → reviewer) using LangGraph-style graph wrappers. Runs in both dry-run (`execute=False`) and live execution (`execute=True`, HTTP mocked) modes.

### CrewAI Integration

```bash
python examples/crewai_example.py
```

Wraps a CrewAI crew with `KortexCrewAIAdapter`. Shows auto-generated `AgentDescriptor`s and capability inference from agent role keywords.

### Custom Provider

```bash
python examples/custom_provider_example.py
```

Mixes local Ollama models, a corporate API endpoint, and cloud providers (Anthropic) in one runtime. Demonstrates `GenericOpenAIConnector` and per-provider auth headers.

### Benchmark

```bash
python examples/benchmark_example.py
```

Full benchmark suite: registers 6 models across 3 tiers, runs Kortex routing against `cheapest` / `strongest` / `random` baselines on three workload datasets, and prints a markdown report.

---

## CLI Commands

```bash
kortex status
kortex models
kortex config
kortex dry-run --task "Summarise this" --complexity moderate
kortex history --last 10
kortex trace list
kortex trace show <trace_id>
kortex replay <trace_id>
kortex policy show
```

All commands work in demo mode (no API keys needed).

---

## Policy Files

TOML policy files in `examples/policies/` can be passed to `--policy` flags:

- **`cost_optimized.toml`** — Minimize cost, prefer fast-tier models, cap at $0.05/request.
- **`quality_first.toml`** — Prefer powerful-tier models, budget ceiling $1.00, latency cap 5s.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KORTEX_TRACE_STORE` | Trace backend: `memory`, `sqlite` | `memory` |
| `KORTEX_TRACE_DB` | SQLite database path | `kortex_traces.db` |
| `KORTEX_STATE_BACKEND` | State backend: `memory`, `sqlite`, `redis` | `memory` |
| `ANTHROPIC_API_KEY` | Anthropic API key (live execution only) | — |
| `OPENAI_API_KEY` | OpenAI API key (live execution only) | — |
