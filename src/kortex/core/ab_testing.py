"""Auto A/B testing framework for routing policy comparison.

Splits incoming tasks between two routing policies (control vs treatment),
tracks per-policy metrics, and automatically promotes the treatment policy
when it achieves statistically significant improvement (p < 0.05, >10%
improvement) over the control.

Statistical significance is determined via Welch's t-test on per-request
cost samples.

Example::

    from kortex.core.ab_testing import ABTest, ExperimentConfig

    config = ExperimentConfig(
        name="cost_experiment",
        control_policy=RoutingPolicy.cost_optimized(),
        treatment_policy=RoutingPolicy.latency_optimized(),
        traffic_split=0.5,  # 50% to treatment
        min_samples=30,
        improvement_threshold=0.10,
    )
    experiment = ABTest(config)

    # In your coordination loop:
    policy = experiment.split_traffic()  # returns control or treatment policy
    # ... run coordination with policy ...
    experiment.record_result(policy.name, cost=0.0025, latency_ms=340.0)

    # Check if we should promote
    if experiment.should_promote():
        winner = experiment.promote()
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from kortex.core.policy import RoutingPolicy

logger = structlog.get_logger(component="ab_testing")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PolicyMetrics:
    """Accumulated metrics for one policy arm in an experiment.

    Args:
        policy_name: Name of the policy.
        costs: Per-request estimated costs.
        latencies: Per-request estimated latencies.
        request_count: Total requests routed to this policy.
    """

    policy_name: str
    costs: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    request_count: int = 0

    @property
    def avg_cost(self) -> float:
        return sum(self.costs) / len(self.costs) if self.costs else 0.0

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def sample_count(self) -> int:
        return len(self.costs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "request_count": self.request_count,
            "sample_count": self.sample_count,
            "avg_cost": round(self.avg_cost, 6),
            "avg_latency": round(self.avg_latency, 2),
        }


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment.

    Args:
        name: Experiment name.
        control_policy: The current ("A") policy.
        treatment_policy: The new ("B") policy being tested.
        traffic_split: Fraction of traffic sent to treatment (0–1).
        min_samples: Minimum samples per arm before significance testing.
        significance_level: p-value threshold for statistical significance.
        improvement_threshold: Minimum relative improvement to auto-promote.
        metric: Which metric to optimise on ("cost" or "latency").
        auto_promote: Whether to automatically promote when conditions are met.
    """

    name: str
    control_policy: RoutingPolicy
    treatment_policy: RoutingPolicy
    traffic_split: float = 0.5
    min_samples: int = 30
    significance_level: float = 0.05
    improvement_threshold: float = 0.10
    metric: Literal["cost", "latency"] = "cost"
    auto_promote: bool = True


@dataclass
class ExperimentResult:
    """Snapshot of an experiment's state.

    Args:
        name: Experiment name.
        control: Metrics for the control arm.
        treatment: Metrics for the treatment arm.
        significant: Whether the difference is statistically significant.
        p_value: Welch t-test p-value.
        improvement_pct: Relative improvement (negative = treatment is better).
        winner: Which policy won ("control", "treatment", or None if no winner yet).
        promoted: Whether the treatment was auto-promoted.
        started_at: Unix timestamp when the experiment started.
    """

    name: str
    control: PolicyMetrics
    treatment: PolicyMetrics
    significant: bool
    p_value: float
    improvement_pct: float
    winner: Literal["control", "treatment"] | None
    promoted: bool
    started_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "control": self.control.to_dict(),
            "treatment": self.treatment.to_dict(),
            "significant": self.significant,
            "p_value": round(self.p_value, 4),
            "improvement_pct": round(self.improvement_pct, 2),
            "winner": self.winner,
            "promoted": self.promoted,
            "started_at": self.started_at,
        }

    def summary(self) -> str:
        parts = [f"Experiment '{self.name}':"]
        parts.append(
            f"  Control ({self.control.policy_name}): "
            f"n={self.control.sample_count}, "
            f"avg_cost=${self.control.avg_cost:.5f}, "
            f"avg_latency={self.control.avg_latency:.0f}ms"
        )
        parts.append(
            f"  Treatment ({self.treatment.policy_name}): "
            f"n={self.treatment.sample_count}, "
            f"avg_cost=${self.treatment.avg_cost:.5f}, "
            f"avg_latency={self.treatment.avg_latency:.0f}ms"
        )
        if self.significant and self.winner:
            direction = "BETTER" if self.winner == "treatment" else "WORSE"
            parts.append(
                f"  Result: Treatment is {direction} by {abs(self.improvement_pct):.1f}% "
                f"(p={self.p_value:.4f})"
            )
            if self.promoted:
                parts.append("  Status: PROMOTED")
        else:
            parts.append(
                f"  Result: Not yet significant (p={self.p_value:.4f}, "
                f"need {self.control.sample_count}/{self.control.request_count} "
                f"and {self.treatment.sample_count}/{self.treatment.request_count} samples)"
            )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Welch t-test
# ---------------------------------------------------------------------------


def _welch_t_test(a: list[float], b: list[float]) -> float:
    """Compute two-sided Welch t-test p-value for two independent samples.

    Uses a simple t-distribution approximation (Welch–Satterthwaite df).
    Falls back to p=1.0 if either sample is too small.

    Args:
        a: Sample from group A.
        b: Sample from group B.

    Returns:
        Two-sided p-value (0–1). Lower = more significant.
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 1.0

    mean1 = sum(a) / n1
    mean2 = sum(b) / n2
    var1 = sum((x - mean1) ** 2 for x in a) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in b) / (n2 - 1)

    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return 1.0 if mean1 == mean2 else 0.0

    t_stat = (mean1 - mean2) / se

    # Welch–Satterthwaite degrees of freedom
    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else n1 + n2 - 2

    # Approximate p-value using the regularized incomplete beta function
    # via a simple approximation for the t-distribution CDF
    p_one_sided = _t_dist_cdf_approx(abs(t_stat), df)
    return 2.0 * (1.0 - p_one_sided)


def _t_dist_cdf_approx(t: float, df: float) -> float:
    """Approximate the t-distribution CDF using Hill's method.

    Accurate for df > 1. Sufficient for our use case.
    """
    x = df / (df + t * t)
    # Regularized incomplete beta function approximation
    p = _beta_inc(x, df / 2, 0.5)
    return 1.0 - p / 2.0


def _beta_inc(x: float, a: float, b: float) -> float:
    """Compute the regularized incomplete beta function I_x(a, b).

    Uses a continued fraction approximation (Lentz's algorithm).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Use symmetry for numerical stability
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _beta_inc(1.0 - x, b, a)

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    prefix = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a

    # Continued fraction (max 200 iterations)
    cf = 1.0
    d = 1.0
    for m in range(1, 201):
        # Even step
        c_num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + c_num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c_den = 1.0 + c_num / cf
        if abs(c_den) < 1e-30:
            c_den = 1e-30
        d = 1.0 / d
        cf *= d * c_den

        # Odd step
        c_num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + c_num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c_den = 1.0 + c_num / cf
        if abs(c_den) < 1e-30:
            c_den = 1e-30
        d = 1.0 / d
        delta = d * c_den
        cf *= delta

        if abs(delta - 1.0) < 1e-8:
            break

    return min(1.0, prefix * cf)


# ---------------------------------------------------------------------------
# ABTest
# ---------------------------------------------------------------------------


class ABTest:
    """Manages a single A/B experiment between two routing policies.

    Args:
        config: Experiment configuration.
        rng_seed: Optional random seed for reproducible traffic splits.
    """

    def __init__(self, config: ExperimentConfig, rng_seed: int | None = None) -> None:
        self._config = config
        self._control_metrics = PolicyMetrics(policy_name=config.control_policy.name)
        self._treatment_metrics = PolicyMetrics(policy_name=config.treatment_policy.name)
        self._promoted = False
        self._winner: Literal["control", "treatment"] | None = None
        self._started_at = time.time()
        self._rng = random.Random(rng_seed)
        self._log = structlog.get_logger(component="ab_testing")

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def is_complete(self) -> bool:
        """Return True if the experiment has been concluded (promoted or stopped)."""
        return self._promoted

    # ------------------------------------------------------------------
    # Traffic splitting
    # ------------------------------------------------------------------

    def split_traffic(self) -> RoutingPolicy:
        """Return either the control or treatment policy based on traffic split.

        Args:
            None

        Returns:
            The RoutingPolicy for this request (control or treatment).
        """
        if self._promoted:
            # Experiment concluded — always use winner
            if self._winner == "treatment":
                return self._config.treatment_policy
            return self._config.control_policy

        use_treatment = self._rng.random() < self._config.traffic_split
        if use_treatment:
            self._treatment_metrics.request_count += 1
            return self._config.treatment_policy
        else:
            self._control_metrics.request_count += 1
            return self._config.control_policy

    # ------------------------------------------------------------------
    # Recording results
    # ------------------------------------------------------------------

    def record_result(
        self,
        policy_name: str,
        cost: float,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a routing result for the given policy arm.

        Args:
            policy_name: The policy name ("control" name or "treatment" name).
            cost: Estimated or actual cost for this request.
            latency_ms: Estimated or actual latency for this request.
        """
        if policy_name == self._config.control_policy.name:
            self._control_metrics.costs.append(cost)
            self._control_metrics.latencies.append(latency_ms)
        elif policy_name == self._config.treatment_policy.name:
            self._treatment_metrics.costs.append(cost)
            self._treatment_metrics.latencies.append(latency_ms)
        else:
            self._log.warning(
                "ab_test_unknown_policy",
                policy_name=policy_name,
                expected=[
                    self._config.control_policy.name,
                    self._config.treatment_policy.name,
                ],
            )

        # Auto-promote if conditions are met
        if self._config.auto_promote and not self._promoted:
            if self.should_promote():
                self.promote()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def should_promote(self) -> bool:
        """Check whether the treatment policy should be auto-promoted.

        Conditions:
        1. Both arms have at least ``min_samples`` samples
        2. The difference is statistically significant (p < significance_level)
        3. Treatment is at least ``improvement_threshold`` better on the metric

        Returns:
            True if treatment should replace control.
        """
        cm = self._control_metrics
        tm = self._treatment_metrics

        if cm.sample_count < self._config.min_samples:
            return False
        if tm.sample_count < self._config.min_samples:
            return False

        if self._config.metric == "cost":
            control_vals = cm.costs
            treatment_vals = tm.costs
        else:
            control_vals = cm.latencies
            treatment_vals = tm.latencies

        p_value = _welch_t_test(control_vals, treatment_vals)
        if p_value >= self._config.significance_level:
            return False

        control_mean = sum(control_vals) / len(control_vals)
        treatment_mean = sum(treatment_vals) / len(treatment_vals)

        if control_mean == 0:
            return False

        # Improvement: positive = treatment is better (lower cost/latency)
        improvement = (control_mean - treatment_mean) / control_mean
        return improvement >= self._config.improvement_threshold

    def get_result(self) -> ExperimentResult:
        """Return the current experiment state as an ExperimentResult.

        Returns:
            ExperimentResult snapshot.
        """
        cm = self._control_metrics
        tm = self._treatment_metrics

        if self._config.metric == "cost":
            control_vals = cm.costs
            treatment_vals = tm.costs
        else:
            control_vals = cm.latencies
            treatment_vals = tm.latencies

        p_value = 1.0
        improvement_pct = 0.0
        significant = False

        if len(control_vals) >= 2 and len(treatment_vals) >= 2:
            p_value = _welch_t_test(control_vals, treatment_vals)
            significant = p_value < self._config.significance_level

            c_mean = sum(control_vals) / len(control_vals) if control_vals else 0.0
            t_mean = sum(treatment_vals) / len(treatment_vals) if treatment_vals else 0.0
            if c_mean > 0:
                improvement_pct = (c_mean - t_mean) / c_mean * 100

        return ExperimentResult(
            name=self._config.name,
            control=cm,
            treatment=tm,
            significant=significant,
            p_value=p_value,
            improvement_pct=improvement_pct,
            winner=self._winner,
            promoted=self._promoted,
            started_at=self._started_at,
        )

    def promote(self) -> RoutingPolicy:
        """Conclude the experiment and promote the treatment policy.

        Returns:
            The promoted (treatment) RoutingPolicy.
        """
        self._promoted = True
        self._winner = "treatment"
        self._log.info(
            "ab_test_promoted",
            experiment=self._config.name,
            control=self._config.control_policy.name,
            treatment=self._config.treatment_policy.name,
        )
        return self._config.treatment_policy

    def reset(self) -> None:
        """Reset the experiment metrics (keeps config)."""
        self._control_metrics = PolicyMetrics(policy_name=self._config.control_policy.name)
        self._treatment_metrics = PolicyMetrics(policy_name=self._config.treatment_policy.name)
        self._promoted = False
        self._winner = None
        self._started_at = time.time()
