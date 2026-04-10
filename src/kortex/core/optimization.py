"""Auto-optimization playground for routing policy tuning.

Runs a grid search over ``RoutingPolicy`` cost/latency weight combinations
against a trace dataset, computes a Pareto frontier of cost vs latency
trade-offs, and returns the Pareto-optimal policy configurations.

The ``OptimizationPlayground`` is designed for offline analysis:
- Give it a list of historical ``TaskTrace`` objects
- It replays each trace under every grid-search policy
- It identifies Pareto-optimal policies (no other policy is strictly
  better on both cost and latency)
- It returns the full results plus the Pareto set

Example::

    from kortex.core.optimization import OptimizationPlayground

    playground = OptimizationPlayground(router)
    result = await playground.optimize(traces, cost_weights=[0.3, 0.5, 0.7])
    for cfg in result.pareto_frontier:
        print(cfg)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

from kortex.core.policy import RoutingPolicy
from kortex.core.replay import ReplayEngine
from kortex.core.trace import TaskTrace

logger = structlog.get_logger(component="optimization")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PolicyEvaluation:
    """Result of evaluating one policy configuration against a trace set.

    Args:
        policy_name: Name of the evaluated policy.
        policy_dict: Serialized policy configuration.
        avg_cost_usd: Average estimated cost per task.
        avg_latency_ms: Average estimated latency per task.
        total_cost_usd: Total estimated cost across all tasks.
        total_latency_ms: Total estimated latency across all tasks.
        num_tasks: Number of tasks evaluated.
        capability_mismatches: Number of steps where capabilities weren't met.
        pareto_optimal: Whether this config is on the Pareto frontier.
    """

    policy_name: str
    policy_dict: dict[str, Any]
    avg_cost_usd: float
    avg_latency_ms: float
    total_cost_usd: float
    total_latency_ms: float
    num_tasks: int
    capability_mismatches: int = 0
    pareto_optimal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_dict": self.policy_dict,
            "avg_cost_usd": round(self.avg_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "num_tasks": self.num_tasks,
            "capability_mismatches": self.capability_mismatches,
            "pareto_optimal": self.pareto_optimal,
        }

    def __str__(self) -> str:
        flag = " [PARETO]" if self.pareto_optimal else ""
        return (
            f"{self.policy_name}{flag}: "
            f"avg_cost=${self.avg_cost_usd:.5f} "
            f"avg_latency={self.avg_latency_ms:.0f}ms"
        )


@dataclass
class OptimizationResult:
    """Complete result of an optimization run.

    Args:
        evaluations: All evaluated policy configurations, sorted by cost.
        pareto_frontier: Subset of evaluations that are Pareto-optimal.
        best_cost: Policy with lowest average cost.
        best_latency: Policy with lowest average latency.
        best_balanced: Policy closest to the Pareto "knee point".
        num_traces: Number of traces used.
        num_policies_evaluated: Total policy configurations evaluated.
    """

    evaluations: list[PolicyEvaluation]
    pareto_frontier: list[PolicyEvaluation]
    best_cost: PolicyEvaluation | None
    best_latency: PolicyEvaluation | None
    best_balanced: PolicyEvaluation | None
    num_traces: int
    num_policies_evaluated: int

    def summary(self) -> str:
        lines = [
            f"Optimization over {self.num_traces} traces, "
            f"{self.num_policies_evaluated} policies:",
        ]
        if self.best_cost:
            lines.append(f"  Best cost:     {self.best_cost}")
        if self.best_latency:
            lines.append(f"  Best latency:  {self.best_latency}")
        if self.best_balanced:
            lines.append(f"  Best balanced: {self.best_balanced}")
        lines.append(f"  Pareto frontier: {len(self.pareto_frontier)} configs")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pareto computation
# ---------------------------------------------------------------------------


def _pareto_frontier(
    evaluations: list[PolicyEvaluation],
) -> list[PolicyEvaluation]:
    """Return the Pareto-optimal subset of evaluations.

    A policy is Pareto-optimal if no other policy has strictly lower cost
    AND strictly lower latency simultaneously.

    Args:
        evaluations: All evaluated policies.

    Returns:
        Pareto-optimal subset.
    """
    pareto: list[PolicyEvaluation] = []
    for candidate in evaluations:
        dominated = any(
            other.avg_cost_usd <= candidate.avg_cost_usd
            and other.avg_latency_ms <= candidate.avg_latency_ms
            and (
                other.avg_cost_usd < candidate.avg_cost_usd
                or other.avg_latency_ms < candidate.avg_latency_ms
            )
            for other in evaluations
            if other is not candidate
        )
        if not dominated:
            pareto.append(candidate)
    return sorted(pareto, key=lambda e: e.avg_cost_usd)


def _knee_point(frontier: list[PolicyEvaluation]) -> PolicyEvaluation | None:
    """Find the "knee point" — the policy closest to the ideal (0,0) point.

    Normalises cost and latency to [0,1] and finds the point with
    minimum Euclidean distance to the origin.

    Args:
        frontier: Pareto-optimal evaluations.

    Returns:
        The balanced policy, or None if frontier is empty.
    """
    if not frontier:
        return None

    costs = [e.avg_cost_usd for e in frontier]
    latencies = [e.avg_latency_ms for e in frontier]
    max_c = max(costs) or 1.0
    max_l = max(latencies) or 1.0

    best = min(frontier, key=lambda e: (e.avg_cost_usd / max_c) ** 2 + (e.avg_latency_ms / max_l) ** 2)
    return best


# ---------------------------------------------------------------------------
# OptimizationPlayground
# ---------------------------------------------------------------------------


class OptimizationPlayground:
    """Grid search over RoutingPolicy configurations to find Pareto-optimal setups.

    Args:
        router: A Router with registered models used for replay evaluation.
        max_concurrent: Maximum concurrent replay coroutines per policy.
    """

    def __init__(
        self,
        router: Any,
        max_concurrent: int = 5,
    ) -> None:
        self._router = router
        self._max_concurrent = max_concurrent
        self._log = structlog.get_logger(component="optimization")

    async def optimize(
        self,
        traces: list[TaskTrace],
        cost_weights: list[float] | None = None,
        latency_weights: list[float] | None = None,
        extra_policies: list[RoutingPolicy] | None = None,
    ) -> OptimizationResult:
        """Run grid search over policy weight combinations.

        For each (cost_weight, latency_weight) pair, creates a balanced
        RoutingPolicy and replays all traces. Returns the full evaluation
        set plus the Pareto frontier.

        Args:
            traces: Historical TaskTrace objects to replay.
            cost_weights: Cost minimisation weight values to try (0–1).
                Defaults to [0.1, 0.3, 0.5, 0.7, 0.9].
            latency_weights: Latency minimisation weight values to try (0–1).
                Defaults to same as cost_weights (grid is cost × latency).
            extra_policies: Additional explicit RoutingPolicy objects to include.

        Returns:
            OptimizationResult with all evaluations and Pareto frontier.
        """
        if not traces:
            return OptimizationResult(
                evaluations=[], pareto_frontier=[],
                best_cost=None, best_latency=None, best_balanced=None,
                num_traces=0, num_policies_evaluated=0,
            )

        cost_weights = cost_weights or [0.1, 0.3, 0.5, 0.7, 0.9]
        latency_weights = latency_weights or cost_weights

        # Build policy grid
        policies: list[RoutingPolicy] = []

        from kortex.core.policy import RoutingObjective

        # Cost-optimized
        policies.append(RoutingPolicy.cost_optimized())
        # Quality-optimized
        policies.append(RoutingPolicy.quality_optimized())
        # Latency-optimized
        policies.append(RoutingPolicy.latency_optimized())

        # Grid
        for cw in cost_weights:
            for lw in latency_weights:
                name = f"grid_c{cw:.1f}_l{lw:.1f}"
                minimize = "cost" if cw >= lw else "latency"
                prefer_tier: str = (
                    "fast" if cw >= lw else
                    "balanced" if cw > 0.3 else
                    "powerful"
                )
                p = RoutingPolicy(
                    name=name,
                    objective=RoutingObjective(
                        minimize=minimize,  # type: ignore[arg-type]
                        prefer_tier=prefer_tier,  # type: ignore[arg-type]
                    ),
                )
                policies.append(p)

        if extra_policies:
            policies.extend(extra_policies)

        # Deduplicate by serialization
        seen: set[str] = set()
        unique_policies: list[RoutingPolicy] = []
        for p in policies:
            key = str(p.to_dict())
            if key not in seen:
                seen.add(key)
                unique_policies.append(p)

        # Evaluate all policies concurrently (per policy, traces run concurrently)
        sem = asyncio.Semaphore(self._max_concurrent)

        async def evaluate_policy(policy: RoutingPolicy) -> PolicyEvaluation:
            return await self._evaluate_policy(policy, traces, sem)

        all_evals = await asyncio.gather(*[evaluate_policy(p) for p in unique_policies])

        # Sort by cost
        all_evals_list = sorted(all_evals, key=lambda e: e.avg_cost_usd)

        # Pareto frontier
        frontier = _pareto_frontier(all_evals_list)
        for e in frontier:
            e.pareto_optimal = True

        best_cost = min(all_evals_list, key=lambda e: e.avg_cost_usd) if all_evals_list else None
        best_lat = min(all_evals_list, key=lambda e: e.avg_latency_ms) if all_evals_list else None
        best_balanced = _knee_point(frontier)

        self._log.info(
            "optimization_complete",
            num_policies=len(all_evals_list),
            pareto_count=len(frontier),
        )

        return OptimizationResult(
            evaluations=all_evals_list,
            pareto_frontier=frontier,
            best_cost=best_cost,
            best_latency=best_lat,
            best_balanced=best_balanced,
            num_traces=len(traces),
            num_policies_evaluated=len(all_evals_list),
        )

    async def _evaluate_policy(
        self,
        policy: RoutingPolicy,
        traces: list[TaskTrace],
        sem: asyncio.Semaphore,
    ) -> PolicyEvaluation:
        """Replay all traces under ``policy`` and return aggregate metrics."""
        engine = ReplayEngine(self._router)
        total_cost = 0.0
        total_latency = 0.0
        count = 0

        async def replay_one(trace: TaskTrace) -> tuple[float, float]:
            async with sem:
                try:
                    result = await engine.replay(trace, policy=policy)
                    c = sum(s.replayed_estimated_cost for s in result.replayed_steps)
                    l = sum(
                        self._router._models.get(
                            f"{s.replayed_provider}::{s.replayed_model}",
                            None,
                        ).avg_latency_ms
                        if f"{s.replayed_provider}::{s.replayed_model}" in self._router._models
                        else 0.0
                        for s in result.replayed_steps
                    )
                    return c, l
                except Exception:
                    return 0.0, 0.0

        results = await asyncio.gather(*[replay_one(t) for t in traces])
        for c, l in results:
            total_cost += c
            total_latency += l
            count += 1

        n = max(count, 1)
        return PolicyEvaluation(
            policy_name=policy.name,
            policy_dict=policy.to_dict(),
            avg_cost_usd=total_cost / n,
            avg_latency_ms=total_latency / n,
            total_cost_usd=total_cost,
            total_latency_ms=total_latency,
            num_tasks=n,
        )
