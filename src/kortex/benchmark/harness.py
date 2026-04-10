"""Benchmark harness for comparing Kortex routing vs static baselines.

Produces measurable proof that policy-based routing beats naive model
assignment on cost, latency, and capability coverage — without making
any real API calls.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from kortex.core.capabilities import Capability
from kortex.core.policy import PolicyRouter, RoutingPolicy
from kortex.core.router import ProviderModel
from kortex.core.types import TaskSpec

logger = structlog.get_logger(component="benchmark")

# ---------------------------------------------------------------------------
# Task content templates for realistic workloads
# ---------------------------------------------------------------------------

_SIMPLE_TASKS = [
    "Summarize the following document in three bullet points.",
    "Translate this paragraph from English to French.",
    "Extract the key dates from this meeting transcript.",
    "Fix the grammar and spelling in this email draft.",
    "Generate a one-sentence tagline for this product.",
]

_MODERATE_TASKS = [
    "Analyze the sentiment trends across these 50 customer reviews.",
    "Write a technical blog post about async programming patterns.",
    "Review this pull request for security vulnerabilities.",
    "Design a database schema for a multi-tenant SaaS application.",
    "Compare these three API designs and recommend the best one.",
]

_COMPLEX_TASKS = [
    "Architect a distributed event sourcing system for financial transactions.",
    "Write a comprehensive test suite for this authentication module.",
    "Analyze this codebase and propose a migration plan from monolith to microservices.",
    "Build a real-time data pipeline design for processing 10M events/day.",
    "Design and document a zero-downtime deployment strategy for Kubernetes.",
]

# Capabilities that make sense for each complexity level
_SIMPLE_CAPS: list[list[str]] = [
    ["content_generation"],
    ["content_generation"],
    ["analysis"],
    ["content_generation"],
    ["content_generation"],
]

_MODERATE_CAPS: list[list[str]] = [
    ["analysis", "reasoning"],
    ["content_generation", "reasoning"],
    ["code_generation", "quality_assurance"],
    ["reasoning", "analysis"],
    ["analysis", "reasoning"],
]

_COMPLEX_CAPS: list[list[str]] = [
    ["reasoning", "analysis", "content_generation"],
    ["code_generation", "quality_assurance", "reasoning"],
    ["analysis", "code_generation", "reasoning"],
    ["reasoning", "analysis", "research"],
    ["reasoning", "code_generation", "analysis"],
]


# ---------------------------------------------------------------------------
# TaskDataset
# ---------------------------------------------------------------------------


@dataclass
class TaskDataset:
    """A named collection of tasks for benchmarking.

    Args:
        name: Dataset identifier.
        tasks: The task specifications to route.
        description: What this dataset tests.
    """

    name: str
    tasks: list[TaskSpec]
    description: str

    @classmethod
    def mixed_workload(cls, n: int = 100) -> TaskDataset:
        """Generate a mixed-complexity workload.

        Distribution: 40% simple, 35% moderate, 25% complex.
        Each task has realistic content and required capabilities.

        Args:
            n: Number of tasks to generate.

        Returns:
            A TaskDataset with mixed complexity tasks.
        """
        tasks: list[TaskSpec] = []
        rng = random.Random(42)  # deterministic for reproducibility

        n_simple = int(n * 0.40)
        n_moderate = int(n * 0.35)
        n_complex = n - n_simple - n_moderate

        for _ in range(n_simple):
            idx = rng.randrange(len(_SIMPLE_TASKS))
            tasks.append(TaskSpec(
                content=_SIMPLE_TASKS[idx],
                complexity_hint="simple",
                required_capabilities=_SIMPLE_CAPS[idx],
            ))

        for _ in range(n_moderate):
            idx = rng.randrange(len(_MODERATE_TASKS))
            tasks.append(TaskSpec(
                content=_MODERATE_TASKS[idx],
                complexity_hint="moderate",
                required_capabilities=_MODERATE_CAPS[idx],
            ))

        for _ in range(n_complex):
            idx = rng.randrange(len(_COMPLEX_TASKS))
            tasks.append(TaskSpec(
                content=_COMPLEX_TASKS[idx],
                complexity_hint="complex",
                required_capabilities=_COMPLEX_CAPS[idx],
            ))

        rng.shuffle(tasks)

        return cls(
            name="mixed_workload",
            tasks=tasks,
            description=f"{n} tasks: 40% simple, 35% moderate, 25% complex",
        )

    @classmethod
    def cost_sensitive(cls, n: int = 100) -> TaskDataset:
        """Generate tasks with tight cost ceilings.

        Forces routing to pick cheap models. All tasks have a
        cost_ceiling_usd that eliminates expensive models.

        Args:
            n: Number of tasks to generate.

        Returns:
            A TaskDataset with cost-constrained tasks.
        """
        tasks: list[TaskSpec] = []
        rng = random.Random(43)

        ceilings = [0.001, 0.002, 0.005]

        for i in range(n):
            idx = rng.randrange(len(_SIMPLE_TASKS))
            tasks.append(TaskSpec(
                content=_SIMPLE_TASKS[idx],
                complexity_hint="simple" if i % 2 == 0 else "moderate",
                required_capabilities=_SIMPLE_CAPS[idx],
                cost_ceiling_usd=rng.choice(ceilings),
            ))

        return cls(
            name="cost_sensitive",
            tasks=tasks,
            description=f"{n} tasks with tight cost ceilings ($0.001-$0.005)",
        )

    @classmethod
    def latency_sensitive(cls, n: int = 100) -> TaskDataset:
        """Generate tasks with tight latency SLAs.

        Forces routing to pick fast models. All tasks have a
        latency_sla_ms that eliminates slow models.

        Args:
            n: Number of tasks to generate.

        Returns:
            A TaskDataset with latency-constrained tasks.
        """
        tasks: list[TaskSpec] = []
        rng = random.Random(44)

        slas = [100.0, 200.0, 300.0]

        # Use simple capabilities so fast-tier models can serve them
        for i in range(n):
            idx = rng.randrange(len(_SIMPLE_TASKS))
            tasks.append(TaskSpec(
                content=_SIMPLE_TASKS[idx],
                complexity_hint="simple" if i % 3 == 0 else "moderate",
                required_capabilities=_SIMPLE_CAPS[idx],
                latency_sla_ms=rng.choice(slas),
            ))

        return cls(
            name="latency_sensitive",
            tasks=tasks,
            description=f"{n} tasks with tight latency SLAs (100-300ms)",
        )


# ---------------------------------------------------------------------------
# BaselineStrategy
# ---------------------------------------------------------------------------


class BaselineStrategy(Enum):
    """Static model selection strategies for baseline comparison."""

    ALWAYS_CHEAPEST = "always_cheapest"
    ALWAYS_STRONGEST = "always_strongest"
    RANDOM = "random"
    SINGLE_MODEL = "single_model"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkStepResult:
    """Result of routing a single task.

    Args:
        task_index: Position in the dataset.
        task_complexity: The task's complexity hint.
        chosen_provider: Provider selected.
        chosen_model: Model selected.
        estimated_cost_usd: Estimated cost for this task.
        estimated_latency_ms: Estimated latency for this task.
        capabilities_met: Whether the model had all required capabilities.
        explanation: Human-readable routing explanation.
    """

    task_index: int
    task_complexity: str
    chosen_provider: str
    chosen_model: str
    estimated_cost_usd: float
    estimated_latency_ms: float
    capabilities_met: bool
    explanation: str


@dataclass
class BenchmarkRun:
    """Aggregated results of running one strategy across a dataset.

    Args:
        strategy_name: Identifier for the strategy used.
        dataset_name: Name of the dataset.
        results: Per-task results.
        total_estimated_cost_usd: Sum of all task costs.
        total_estimated_latency_ms: Sum of all task latencies.
        avg_cost_per_task: Mean cost per task.
        avg_latency_per_task: Mean latency per task.
        routing_failures: Tasks that could not be routed.
        capability_mismatches: Tasks routed to models missing capabilities.
        wall_time_ms: Time to run the benchmark itself.
    """

    strategy_name: str
    dataset_name: str
    results: list[BenchmarkStepResult]
    total_estimated_cost_usd: float
    total_estimated_latency_ms: float
    avg_cost_per_task: float
    avg_latency_per_task: float
    routing_failures: int
    capability_mismatches: int
    wall_time_ms: float


@dataclass
class BenchmarkComparison:
    """Side-by-side comparison of a baseline vs Kortex routing.

    Args:
        baseline: Results from the baseline strategy.
        kortex: Results from Kortex policy routing.
        cost_delta_pct: Percentage change (negative = Kortex saved money).
        latency_delta_pct: Percentage change (negative = Kortex was faster).
        capability_match_improvement: Extra tasks where Kortex met capabilities.
        summary: Human-readable comparison summary.
    """

    baseline: BenchmarkRun
    kortex: BenchmarkRun
    cost_delta_pct: float
    latency_delta_pct: float
    capability_match_improvement: int
    summary: str


@dataclass
class BenchmarkReport:
    """Full benchmark report across datasets and policies.

    Args:
        comparisons: All comparisons run.
        generated_at: ISO timestamp of report generation.
        model_registry_size: Number of models available.
        summary: Top-level summary.
    """

    comparisons: list[BenchmarkComparison]
    generated_at: str
    model_registry_size: int
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "generated_at": self.generated_at,
            "model_registry_size": self.model_registry_size,
            "summary": self.summary,
            "comparisons": [
                {
                    "baseline": {
                        "strategy": c.baseline.strategy_name,
                        "dataset": c.baseline.dataset_name,
                        "total_cost": c.baseline.total_estimated_cost_usd,
                        "avg_cost": c.baseline.avg_cost_per_task,
                        "total_latency": c.baseline.total_estimated_latency_ms,
                        "avg_latency": c.baseline.avg_latency_per_task,
                        "capability_mismatches": c.baseline.capability_mismatches,
                        "routing_failures": c.baseline.routing_failures,
                    },
                    "kortex": {
                        "strategy": c.kortex.strategy_name,
                        "dataset": c.kortex.dataset_name,
                        "total_cost": c.kortex.total_estimated_cost_usd,
                        "avg_cost": c.kortex.avg_cost_per_task,
                        "total_latency": c.kortex.total_estimated_latency_ms,
                        "avg_latency": c.kortex.avg_latency_per_task,
                        "capability_mismatches": c.kortex.capability_mismatches,
                        "routing_failures": c.kortex.routing_failures,
                    },
                    "cost_delta_pct": c.cost_delta_pct,
                    "latency_delta_pct": c.latency_delta_pct,
                    "capability_match_improvement": c.capability_match_improvement,
                    "summary": c.summary,
                }
                for c in self.comparisons
            ],
        }

    def to_markdown(self) -> str:
        """Render as a markdown table suitable for README or docs."""
        lines: list[str] = []
        lines.append("## Benchmark Results")
        lines.append("")
        lines.append(f"Generated: {self.generated_at}")
        lines.append(f"Models in registry: {self.model_registry_size}")
        lines.append("")

        # Summary table
        lines.append(
            "| Dataset | Policy | Baseline | Cost Delta | Latency Delta "
            "| Cap. Mismatches (B/K) |"
        )
        lines.append(
            "|---------|--------|----------|------------|---------------"
            "|-----------------------|"
        )

        for c in self.comparisons:
            cost_sign = "+" if c.cost_delta_pct >= 0 else ""
            lat_sign = "+" if c.latency_delta_pct >= 0 else ""
            lines.append(
                f"| {c.kortex.dataset_name} "
                f"| {c.kortex.strategy_name} "
                f"| {c.baseline.strategy_name} "
                f"| {cost_sign}{c.cost_delta_pct:.1f}% "
                f"| {lat_sign}{c.latency_delta_pct:.1f}% "
                f"| {c.baseline.capability_mismatches}/{c.kortex.capability_mismatches} |"
            )

        lines.append("")
        lines.append(self.summary)
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BenchmarkHarness
# ---------------------------------------------------------------------------


class BenchmarkHarness:
    """Runs routing benchmarks comparing Kortex policies vs static baselines.

    Args:
        models: The available model registry to route against.
    """

    def __init__(self, models: list[ProviderModel]) -> None:
        self._models = list(models)
        if not self._models:
            raise ValueError("At least one model is required for benchmarking.")

        # Pre-sort for baseline strategies
        self._by_cost = sorted(self._models, key=lambda m: m.estimated_cost())
        self._by_power = sorted(
            self._models,
            key=lambda m: (
                {"powerful": 0, "balanced": 1, "fast": 2}.get(m.tier, 3),
                -m.estimated_cost(),
            ),
        )
        self._rng = random.Random(99)

    async def run_baseline(
        self,
        dataset: TaskDataset,
        strategy: BaselineStrategy,
        model_identity: str | None = None,
    ) -> BenchmarkRun:
        """Run a baseline strategy across a dataset.

        Selects models using a static strategy (no intelligence).

        Args:
            dataset: Tasks to route.
            strategy: Which baseline strategy to use.
            model_identity: Required when strategy is SINGLE_MODEL.
                Format: "provider::model_name".

        Returns:
            Aggregated benchmark results.
        """
        start = time.perf_counter()
        results: list[BenchmarkStepResult] = []
        failures = 0

        target_model: ProviderModel | None = None
        if strategy == BaselineStrategy.SINGLE_MODEL:
            if model_identity is None:
                raise ValueError(
                    "model_identity is required for SINGLE_MODEL strategy"
                )
            for m in self._models:
                if f"{m.provider}::{m.model}" == model_identity:
                    target_model = m
                    break
            if target_model is None:
                raise ValueError(
                    f"Model '{model_identity}' not found in registry"
                )

        for i, task in enumerate(dataset.tasks):
            model = self._pick_baseline_model(strategy, target_model)
            if model is None:
                failures += 1
                continue

            caps_met = all(
                cap in model.capabilities
                for cap in task.required_capabilities
            )

            results.append(BenchmarkStepResult(
                task_index=i,
                task_complexity=task.complexity_hint,
                chosen_provider=model.provider,
                chosen_model=model.model,
                estimated_cost_usd=model.estimated_cost(),
                estimated_latency_ms=model.avg_latency_ms,
                capabilities_met=caps_met,
                explanation=f"Baseline {strategy.value}: {model.model}",
            ))

        wall_time = (time.perf_counter() - start) * 1000
        return self._aggregate(
            f"baseline_{strategy.value}", dataset.name, results, failures,
            wall_time,
        )

    async def run_kortex(
        self, dataset: TaskDataset, policy: RoutingPolicy
    ) -> BenchmarkRun:
        """Run Kortex policy-based routing across a dataset.

        Args:
            dataset: Tasks to route.
            policy: The routing policy to evaluate.

        Returns:
            Aggregated benchmark results.
        """
        start = time.perf_counter()
        results: list[BenchmarkStepResult] = []
        failures = 0

        router = PolicyRouter(policy, self._models)

        for i, task in enumerate(dataset.tasks):
            try:
                evaluation = await router.evaluate(task)
            except Exception:
                failures += 1
                continue

            chosen = evaluation.chosen
            caps_met = all(
                cap in chosen.capabilities
                for cap in task.required_capabilities
            )

            results.append(BenchmarkStepResult(
                task_index=i,
                task_complexity=task.complexity_hint,
                chosen_provider=chosen.provider,
                chosen_model=chosen.model,
                estimated_cost_usd=chosen.estimated_cost(),
                estimated_latency_ms=chosen.avg_latency_ms,
                capabilities_met=caps_met,
                explanation=evaluation.explanation,
            ))

        wall_time = (time.perf_counter() - start) * 1000
        return self._aggregate(
            f"kortex_{policy.name}", dataset.name, results, failures,
            wall_time,
        )

    async def compare(
        self,
        dataset: TaskDataset,
        policy: RoutingPolicy,
        baseline_strategy: BaselineStrategy,
        baseline_model: str | None = None,
    ) -> BenchmarkComparison:
        """Run both baseline and Kortex, then compare.

        Args:
            dataset: Tasks to route.
            policy: Kortex routing policy.
            baseline_strategy: Which baseline to compare against.
            baseline_model: Model identity for SINGLE_MODEL baseline.

        Returns:
            A comparison with deltas and summary.
        """
        baseline_run = await self.run_baseline(
            dataset, baseline_strategy, model_identity=baseline_model,
        )
        kortex_run = await self.run_kortex(dataset, policy)

        return self._build_comparison(baseline_run, kortex_run)

    async def full_benchmark(
        self,
        datasets: list[TaskDataset] | None = None,
        policies: list[RoutingPolicy] | None = None,
    ) -> BenchmarkReport:
        """Run every policy against every baseline on every dataset.

        Args:
            datasets: Datasets to use. Defaults to mixed, cost_sensitive,
                latency_sensitive.
            policies: Policies to test. Defaults to cost_optimized,
                latency_optimized, quality_optimized.

        Returns:
            Full benchmark report with all comparisons.
        """
        if datasets is None:
            datasets = [
                TaskDataset.mixed_workload(),
                TaskDataset.cost_sensitive(),
                TaskDataset.latency_sensitive(),
            ]

        if policies is None:
            policies = [
                RoutingPolicy.cost_optimized(),
                RoutingPolicy.latency_optimized(),
                RoutingPolicy.quality_optimized(),
            ]

        baselines = [
            BaselineStrategy.ALWAYS_CHEAPEST,
            BaselineStrategy.ALWAYS_STRONGEST,
            BaselineStrategy.RANDOM,
        ]

        comparisons: list[BenchmarkComparison] = []

        for ds in datasets:
            for policy in policies:
                for baseline in baselines:
                    comparison = await self.compare(
                        ds, policy, baseline,
                    )
                    comparisons.append(comparison)

        # Build top-level summary
        cost_savings = [
            c.cost_delta_pct for c in comparisons
            if c.cost_delta_pct < 0
        ]
        if cost_savings:
            min_save = min(abs(s) for s in cost_savings)
            max_save = max(abs(s) for s in cost_savings)
            save_range = f"{min_save:.0f}-{max_save:.0f}%"
        else:
            save_range = "0%"

        total_cap_improvements = sum(
            c.capability_match_improvement for c in comparisons
            if c.capability_match_improvement > 0
        )

        summary = (
            f"Across {len(datasets)} dataset(s) and {len(policies)} "
            f"policy/policies, Kortex reduced estimated cost by {save_range} "
            f"vs static baselines with {total_cap_improvements} total "
            f"capability-match improvements."
        )

        return BenchmarkReport(
            comparisons=comparisons,
            generated_at=datetime.now(timezone.utc).isoformat(),
            model_registry_size=len(self._models),
            summary=summary,
        )

    # -- internal helpers ----------------------------------------------------

    def _pick_baseline_model(
        self,
        strategy: BaselineStrategy,
        target: ProviderModel | None = None,
    ) -> ProviderModel | None:
        """Select a model using a static baseline strategy."""
        if strategy == BaselineStrategy.ALWAYS_CHEAPEST:
            return self._by_cost[0]
        elif strategy == BaselineStrategy.ALWAYS_STRONGEST:
            return self._by_power[0]
        elif strategy == BaselineStrategy.RANDOM:
            return self._rng.choice(self._models)
        elif strategy == BaselineStrategy.SINGLE_MODEL:
            return target
        return None

    @staticmethod
    def _aggregate(
        strategy_name: str,
        dataset_name: str,
        results: list[BenchmarkStepResult],
        failures: int,
        wall_time_ms: float,
    ) -> BenchmarkRun:
        """Aggregate step results into a BenchmarkRun."""
        total_cost = sum(r.estimated_cost_usd for r in results)
        total_latency = sum(r.estimated_latency_ms for r in results)
        n = len(results) or 1
        cap_mismatches = sum(1 for r in results if not r.capabilities_met)

        return BenchmarkRun(
            strategy_name=strategy_name,
            dataset_name=dataset_name,
            results=results,
            total_estimated_cost_usd=total_cost,
            total_estimated_latency_ms=total_latency,
            avg_cost_per_task=total_cost / n,
            avg_latency_per_task=total_latency / n,
            routing_failures=failures,
            capability_mismatches=cap_mismatches,
            wall_time_ms=wall_time_ms,
        )

    @staticmethod
    def _build_comparison(
        baseline: BenchmarkRun, kortex: BenchmarkRun
    ) -> BenchmarkComparison:
        """Build a comparison between baseline and Kortex runs."""
        if baseline.total_estimated_cost_usd > 0:
            cost_delta_pct = (
                (kortex.total_estimated_cost_usd
                 - baseline.total_estimated_cost_usd)
                / baseline.total_estimated_cost_usd
                * 100
            )
        else:
            cost_delta_pct = 0.0

        if baseline.total_estimated_latency_ms > 0:
            latency_delta_pct = (
                (kortex.total_estimated_latency_ms
                 - baseline.total_estimated_latency_ms)
                / baseline.total_estimated_latency_ms
                * 100
            )
        else:
            latency_delta_pct = 0.0

        cap_improvement = (
            baseline.capability_mismatches - kortex.capability_mismatches
        )

        total_tasks = len(kortex.results) + kortex.routing_failures
        cost_sign = "saved" if cost_delta_pct < 0 else "increased"
        summary = (
            f"Kortex {kortex.strategy_name} vs {baseline.strategy_name}: "
            f"{abs(cost_delta_pct):.0f}% cost {cost_sign}, "
            f"{cap_improvement} fewer capability mismatches, "
            f"on {total_tasks} {kortex.dataset_name} tasks."
        )

        return BenchmarkComparison(
            baseline=baseline,
            kortex=kortex,
            cost_delta_pct=cost_delta_pct,
            latency_delta_pct=latency_delta_pct,
            capability_match_improvement=cap_improvement,
            summary=summary,
        )
