"""Benchmark example — proves Kortex routing beats static model assignment.

Registers 6 models across 3 tiers, runs the full benchmark suite,
and prints a markdown report. No API keys or network access required.

Usage:
    python examples/benchmark_example.py
"""

from __future__ import annotations

import asyncio

from kortex.benchmark.harness import BenchmarkHarness, TaskDataset
from kortex.core.policy import RoutingPolicy
from kortex.core.router import ProviderModel


# ---------------------------------------------------------------------------
# Model registry — same models used in stress tests
# ---------------------------------------------------------------------------

BENCHMARK_MODELS: list[ProviderModel] = [
    # Fast tier — cheap and quick, limited capabilities
    ProviderModel(
        provider="local",
        model="speed-7b",
        cost_per_1k_input_tokens=0.0001,
        cost_per_1k_output_tokens=0.0002,
        avg_latency_ms=50,
        capabilities=["analysis"],
        tier="fast",
    ),
    ProviderModel(
        provider="cloud",
        model="turbo-mini",
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0003,
        avg_latency_ms=80,
        capabilities=["analysis", "code_generation"],
        tier="fast",
    ),
    # Balanced tier — moderate cost, broad capabilities
    ProviderModel(
        provider="cloud",
        model="standard-70b",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.006,
        avg_latency_ms=400,
        capabilities=[
            "reasoning", "code_generation", "analysis",
            "content_generation", "research",
        ],
        tier="balanced",
    ),
    ProviderModel(
        provider="cloud",
        model="mid-range-v2",
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.010,
        avg_latency_ms=600,
        capabilities=[
            "reasoning", "code_generation", "analysis", "vision",
            "content_generation", "quality_assurance",
        ],
        tier="balanced",
    ),
    # Powerful tier — expensive, full capabilities
    ProviderModel(
        provider="cloud",
        model="apex-405b",
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.030,
        avg_latency_ms=1500,
        capabilities=[
            "reasoning", "code_generation", "analysis", "vision",
            "research", "content_generation", "quality_assurance",
        ],
        tier="powerful",
    ),
    ProviderModel(
        provider="cloud",
        model="titan-ultra",
        cost_per_1k_input_tokens=0.020,
        cost_per_1k_output_tokens=0.040,
        avg_latency_ms=2000,
        capabilities=[
            "reasoning", "code_generation", "analysis", "vision", "audio",
            "research", "content_generation", "quality_assurance",
        ],
        tier="powerful",
    ),
]


async def main() -> None:
    """Run the full benchmark and print markdown report."""
    harness = BenchmarkHarness(BENCHMARK_MODELS)

    print("Running full benchmark suite...")
    print(f"Models: {len(BENCHMARK_MODELS)} across 3 tiers")
    print()

    report = await harness.full_benchmark()

    print(report.to_markdown())
    print()
    print("---")
    print(report.summary)


if __name__ == "__main__":
    asyncio.run(main())
