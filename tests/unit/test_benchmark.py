"""Unit tests for the benchmark harness."""

from __future__ import annotations

import json
import tempfile

import pytest

from kortex.benchmark.harness import (
    BaselineStrategy,
    BenchmarkHarness,
    TaskDataset,
)
from kortex.core.policy import RoutingPolicy
from kortex.core.router import ProviderModel
from kortex.dashboard.cli import KortexCLI, main
from kortex.core.router import Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _models() -> list[ProviderModel]:
    """6 models across 3 tiers, matching stress test fixtures."""
    return [
        ProviderModel(
            provider="local", model="speed-7b",
            cost_per_1k_input_tokens=0.0001, cost_per_1k_output_tokens=0.0002,
            avg_latency_ms=50, capabilities=["analysis"], tier="fast",
        ),
        ProviderModel(
            provider="cloud", model="turbo-mini",
            cost_per_1k_input_tokens=0.00015, cost_per_1k_output_tokens=0.0003,
            avg_latency_ms=80, capabilities=["analysis", "code_generation"],
            tier="fast",
        ),
        ProviderModel(
            provider="cloud", model="standard-70b",
            cost_per_1k_input_tokens=0.003, cost_per_1k_output_tokens=0.006,
            avg_latency_ms=400,
            capabilities=["reasoning", "code_generation", "analysis",
                          "content_generation", "research"],
            tier="balanced",
        ),
        ProviderModel(
            provider="cloud", model="mid-range-v2",
            cost_per_1k_input_tokens=0.005, cost_per_1k_output_tokens=0.010,
            avg_latency_ms=600,
            capabilities=["reasoning", "code_generation", "analysis", "vision",
                          "content_generation", "quality_assurance"],
            tier="balanced",
        ),
        ProviderModel(
            provider="cloud", model="apex-405b",
            cost_per_1k_input_tokens=0.015, cost_per_1k_output_tokens=0.030,
            avg_latency_ms=1500,
            capabilities=["reasoning", "code_generation", "analysis", "vision",
                          "research", "content_generation", "quality_assurance"],
            tier="powerful",
        ),
        ProviderModel(
            provider="cloud", model="titan-ultra",
            cost_per_1k_input_tokens=0.020, cost_per_1k_output_tokens=0.040,
            avg_latency_ms=2000,
            capabilities=["reasoning", "code_generation", "analysis", "vision",
                          "audio", "research", "content_generation",
                          "quality_assurance"],
            tier="powerful",
        ),
    ]


def _harness() -> BenchmarkHarness:
    return BenchmarkHarness(_models())


# ---------------------------------------------------------------------------
# 1. TaskDataset.mixed_workload generates correct distribution
# ---------------------------------------------------------------------------


class TestMixedWorkloadDistribution:
    def test_distribution(self) -> None:
        ds = TaskDataset.mixed_workload(100)
        simple = sum(1 for t in ds.tasks if t.complexity_hint == "simple")
        moderate = sum(1 for t in ds.tasks if t.complexity_hint == "moderate")
        complex_ = sum(1 for t in ds.tasks if t.complexity_hint == "complex")

        assert simple == 40
        assert moderate == 35
        assert complex_ == 25
        assert len(ds.tasks) == 100

    def test_small_dataset(self) -> None:
        ds = TaskDataset.mixed_workload(10)
        assert len(ds.tasks) == 10


# ---------------------------------------------------------------------------
# 2. Baseline always_cheapest picks the cheapest model every time
# ---------------------------------------------------------------------------


class TestBaselineAlwaysCheapest:
    @pytest.mark.asyncio
    async def test_picks_cheapest(self) -> None:
        harness = _harness()
        ds = TaskDataset.mixed_workload(20)

        run = await harness.run_baseline(ds, BaselineStrategy.ALWAYS_CHEAPEST)

        # speed-7b is the cheapest model
        for r in run.results:
            assert r.chosen_model == "speed-7b"
            assert r.chosen_provider == "local"


# ---------------------------------------------------------------------------
# 3. Baseline always_strongest picks the most powerful model every time
# ---------------------------------------------------------------------------


class TestBaselineAlwaysStrongest:
    @pytest.mark.asyncio
    async def test_picks_strongest(self) -> None:
        harness = _harness()
        ds = TaskDataset.mixed_workload(20)

        run = await harness.run_baseline(ds, BaselineStrategy.ALWAYS_STRONGEST)

        # titan-ultra is the most powerful (most expensive in powerful tier)
        for r in run.results:
            assert r.chosen_model == "titan-ultra"


# ---------------------------------------------------------------------------
# 4. Baseline random picks different models (not all the same)
# ---------------------------------------------------------------------------


class TestBaselineRandom:
    @pytest.mark.asyncio
    async def test_picks_varying_models(self) -> None:
        harness = _harness()
        ds = TaskDataset.mixed_workload(50)

        run = await harness.run_baseline(ds, BaselineStrategy.RANDOM)

        unique_models = {r.chosen_model for r in run.results}
        # With 50 tasks and 6 models, random should pick at least 2 different ones
        assert len(unique_models) >= 2


# ---------------------------------------------------------------------------
# 5. Kortex cost_optimized policy beats always_strongest on cost
# ---------------------------------------------------------------------------


class TestKortexBeatsCheapest:
    @pytest.mark.asyncio
    async def test_cost_optimized_cheaper_than_strongest(self) -> None:
        harness = _harness()
        ds = TaskDataset.mixed_workload(50)

        baseline = await harness.run_baseline(
            ds, BaselineStrategy.ALWAYS_STRONGEST,
        )
        kortex = await harness.run_kortex(
            ds, RoutingPolicy.cost_optimized(),
        )

        assert kortex.total_estimated_cost_usd < baseline.total_estimated_cost_usd


# ---------------------------------------------------------------------------
# 6. Kortex quality_optimized matches always_strongest on capability match
# ---------------------------------------------------------------------------


class TestKortexQualityCapabilities:
    @pytest.mark.asyncio
    async def test_quality_no_worse_than_strongest(self) -> None:
        harness = _harness()
        ds = TaskDataset.mixed_workload(50)

        baseline = await harness.run_baseline(
            ds, BaselineStrategy.ALWAYS_STRONGEST,
        )
        kortex = await harness.run_kortex(
            ds, RoutingPolicy.quality_optimized(),
        )

        # Quality policy should have same or fewer capability mismatches
        assert kortex.capability_mismatches <= baseline.capability_mismatches


# ---------------------------------------------------------------------------
# 7. Kortex cost_optimized has fewer capability mismatches than random
# ---------------------------------------------------------------------------


class TestKortexBetterThanRandom:
    @pytest.mark.asyncio
    async def test_fewer_mismatches_than_random(self) -> None:
        harness = _harness()
        ds = TaskDataset.mixed_workload(100)

        baseline = await harness.run_baseline(
            ds, BaselineStrategy.RANDOM,
        )
        kortex = await harness.run_kortex(
            ds, RoutingPolicy.cost_optimized(),
        )

        # Kortex should have same or fewer mismatches than random
        assert kortex.capability_mismatches <= baseline.capability_mismatches


# ---------------------------------------------------------------------------
# 8. BenchmarkComparison cost_delta_pct is negative when Kortex saves money
# ---------------------------------------------------------------------------


class TestComparisonCostDelta:
    @pytest.mark.asyncio
    async def test_negative_delta_means_savings(self) -> None:
        harness = _harness()
        ds = TaskDataset.mixed_workload(50)

        comparison = await harness.compare(
            ds,
            RoutingPolicy.cost_optimized(),
            BaselineStrategy.ALWAYS_STRONGEST,
        )

        # Cost-optimized should be cheaper than always-strongest
        assert comparison.cost_delta_pct < 0


# ---------------------------------------------------------------------------
# 9. BenchmarkComparison summary is human-readable with key numbers
# ---------------------------------------------------------------------------


class TestComparisonSummary:
    @pytest.mark.asyncio
    async def test_summary_contains_key_info(self) -> None:
        harness = _harness()
        ds = TaskDataset.mixed_workload(50)

        comparison = await harness.compare(
            ds,
            RoutingPolicy.cost_optimized(),
            BaselineStrategy.ALWAYS_STRONGEST,
        )

        assert "kortex" in comparison.summary.lower() or "Kortex" in comparison.summary
        assert "%" in comparison.summary
        assert "cost" in comparison.summary.lower()
        assert "50" in comparison.summary  # number of tasks


# ---------------------------------------------------------------------------
# 10. BenchmarkReport.to_markdown() produces a valid markdown table
# ---------------------------------------------------------------------------


class TestReportMarkdown:
    @pytest.mark.asyncio
    async def test_markdown_has_table(self) -> None:
        harness = _harness()
        ds = TaskDataset.mixed_workload(20)

        report = await harness.full_benchmark(
            datasets=[ds],
            policies=[RoutingPolicy.cost_optimized()],
        )

        md = report.to_markdown()

        assert "## Benchmark Results" in md
        assert "| Dataset" in md
        assert "|---" in md
        assert "Models in registry: 6" in md


# ---------------------------------------------------------------------------
# 11. Full benchmark runs all combinations without error
# ---------------------------------------------------------------------------


class TestFullBenchmark:
    @pytest.mark.asyncio
    async def test_runs_all_combinations(self) -> None:
        harness = _harness()

        report = await harness.full_benchmark()

        # 3 datasets x 3 policies x 3 baselines = 27 comparisons
        assert len(report.comparisons) == 27
        assert report.model_registry_size == 6
        assert report.generated_at  # non-empty
        assert report.summary  # non-empty

        # All comparisons should have results
        for c in report.comparisons:
            assert len(c.baseline.results) > 0
            assert len(c.kortex.results) > 0


# ---------------------------------------------------------------------------
# 12. CLI benchmark run produces formatted output
# ---------------------------------------------------------------------------


class TestCLIBenchmarkRun:
    @pytest.mark.asyncio
    async def test_benchmark_run_output(self) -> None:
        registry = ProviderRegistry()
        mock_provider = MagicMock()
        mock_provider.provider_name = "cloud"
        mock_provider.health_check = AsyncMock(return_value=True)
        mock_provider.get_available_models.return_value = _models()
        registry.register_provider(mock_provider)

        router = Router()
        for m in _models():
            router.register_model(m)

        runtime = KortexRuntime(
            router=router,
            state_manager=StateManager(InMemoryStateStore()),
            registry=registry,
        )

        cli = KortexCLI(runtime, registry)

        output = await cli.cmd_benchmark_run(dataset_name="mixed")

        assert "Benchmark Results" in output
        assert "Dataset" in output
        assert "|---" in output
