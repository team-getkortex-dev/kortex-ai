# Benchmarking

Kortex ships a benchmark harness that measures the cost and routing-quality advantage of Kortex's heuristic router over static model assignment strategies. Run it before deploying to production to quantify the routing benefit for your specific workload.

---

## Quick Start

```python
import asyncio
from kortex.benchmark.harness import BenchmarkHarness
from kortex.core.router import ProviderModel

models = [
    ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015, cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=250, capabilities=["reasoning", "analysis"], tier="fast",
    ),
    ProviderModel(
        provider="anthropic", model="claude-sonnet-4-20250514",
        cost_per_1k_input_tokens=0.003, cost_per_1k_output_tokens=0.015,
        avg_latency_ms=800,
        capabilities=["reasoning", "analysis", "code_generation", "content_generation"],
        tier="balanced",
    ),
    ProviderModel(
        provider="anthropic", model="claude-opus-4-20250514",
        cost_per_1k_input_tokens=0.015, cost_per_1k_output_tokens=0.075,
        avg_latency_ms=2000,
        capabilities=["reasoning", "analysis", "code_generation", "content_generation"],
        tier="powerful",
    ),
]

async def main():
    harness = BenchmarkHarness(models)
    report = await harness.full_benchmark()
    print(report.to_markdown())
    print(report.summary)

asyncio.run(main())
```

No API keys or network access required — the harness works entirely from cost and latency metadata.

---

## Task Datasets

Three pre-built datasets cover different real-world workload profiles:

| Dataset | Tasks | Profile |
|---------|-------|---------|
| `mixed` | 100 | 40% simple / 35% moderate / 25% complex |
| `cost_sensitive` | 100 | All have cost ceiling constraints |
| `latency_sensitive` | 100 | All have tight latency SLA requirements |

```python
from kortex.benchmark.harness import TaskDataset

mixed     = TaskDataset.mixed_workload(n=100)
cost      = TaskDataset.cost_sensitive(n=100)
latency   = TaskDataset.latency_sensitive(n=100)
```

---

## Baseline Strategies

The harness compares Kortex routing against three static baselines:

| Strategy | Behaviour |
|----------|-----------|
| `cheapest` | Always routes to the cheapest registered model |
| `strongest` | Always routes to the most powerful registered model |
| `random` | Random model selection (lower bound) |

---

## CLI Commands

### Run the full benchmark suite

```bash
kortex benchmark run
kortex benchmark run --dataset cost_sensitive
kortex benchmark run --dataset latency_sensitive
kortex benchmark run --output results.json
```

### Compare routing vs. a baseline under a specific policy

```bash
kortex benchmark compare --policy examples/policies/cost_optimized.toml --baseline cheapest
kortex benchmark compare --policy examples/policies/quality_first.toml --baseline strongest
```

---

## Report Format

`BenchmarkReport.to_markdown()` produces a table like:

```markdown
| Dataset          | Kortex Cost | Baseline Cost | Savings | Kortex P95 | Baseline P95 |
|------------------|-------------|---------------|---------|------------|--------------|
| mixed            | $0.0412     | $0.1850       | 77.7%   | 800ms      | 2000ms       |
| cost_sensitive   | $0.0087     | $0.0450       | 80.7%   | 250ms      | 800ms        |
| latency_sensitive| $0.0031     | $0.1850       | 98.3%   | 250ms      | 2000ms       |
```

`BenchmarkReport.summary` is a single human-readable sentence suitable for logging.

---

## Programmatic API

```python
from kortex.benchmark.harness import BenchmarkHarness, BaselineStrategy

harness = BenchmarkHarness(models)

# Run individual phases
kortex_run  = await harness.run_kortex(dataset)
baseline_run = await harness.run_baseline(dataset, BaselineStrategy.CHEAPEST)

# Compare
comparison = harness.compare(kortex_run, baseline_run)
print(f"Cost savings: {comparison.cost_savings_pct:.1f}%")
print(f"Routing failures: {comparison.kortex_routing_failures}")
```

---

## Running the Example

```bash
python examples/benchmark_example.py
```

This registers 6 models across 3 tiers, runs the full suite, and prints a markdown report.
