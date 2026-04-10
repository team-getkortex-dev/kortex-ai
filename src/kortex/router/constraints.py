"""Constraint-based model filtering for the Kortex router.

Constraints are composable hard filters applied *before* the heuristic or
policy scoring step. Any model that fails a hard constraint is excluded from
consideration entirely — unlike soft objectives, which merely shift scores.

Usage::

    from kortex.router.constraints import ConstraintSet, LatencyConstraint, CostConstraint

    cs = ConstraintSet()
    cs.add(LatencyConstraint(max_ms=500))
    cs.add(CostConstraint(max_usd=0.001))

    passed, failures = cs.filter(router.models)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kortex.core.router import ProviderModel


class Constraint(ABC):
    """Abstract base for a single hard model constraint.

    Subclasses implement ``evaluate`` and ``describe``; the default
    ``failure_reason`` formats a generic message that subclasses may override
    for better diagnostics.
    """

    @abstractmethod
    def evaluate(self, model: "ProviderModel") -> bool:
        """Return True if the model satisfies this constraint."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this constraint."""
        ...

    def failure_reason(self, model: "ProviderModel") -> str:
        """Return a human-readable reason why the model failed.

        Args:
            model: The model that failed the constraint.

        Returns:
            A descriptive failure string.
        """
        return f"{model.provider}::{model.model} failed {self.describe()}"


class LatencyConstraint(Constraint):
    """Reject models whose average latency exceeds a ceiling.

    Args:
        max_ms: Maximum allowed average latency in milliseconds.
    """

    def __init__(self, max_ms: float) -> None:
        self.max_ms = max_ms

    def evaluate(self, model: "ProviderModel") -> bool:
        return model.avg_latency_ms <= self.max_ms

    def describe(self) -> str:
        return f"LatencyConstraint(max={self.max_ms}ms)"

    def failure_reason(self, model: "ProviderModel") -> str:
        return (
            f"{model.provider}::{model.model} avg_latency={model.avg_latency_ms}ms "
            f"> max={self.max_ms}ms"
        )


class CostConstraint(Constraint):
    """Reject models whose estimated cost per request exceeds a ceiling.

    Args:
        max_usd: Maximum allowed estimated cost in USD.
    """

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd

    def evaluate(self, model: "ProviderModel") -> bool:
        return model.estimated_cost() <= self.max_usd

    def describe(self) -> str:
        return f"CostConstraint(max=${self.max_usd:.4f})"

    def failure_reason(self, model: "ProviderModel") -> str:
        return (
            f"{model.provider}::{model.model} cost=${model.estimated_cost():.4f} "
            f"> max=${self.max_usd:.4f}"
        )


class CapabilityConstraint(Constraint):
    """Reject models that lack one or more required capabilities.

    Args:
        required: List of capability strings the model must possess.
    """

    def __init__(self, required: list[str]) -> None:
        self.required: set[str] = set(required)

    def evaluate(self, model: "ProviderModel") -> bool:
        return self.required.issubset(set(model.capabilities))

    def describe(self) -> str:
        return f"CapabilityConstraint(required={sorted(self.required)})"

    def failure_reason(self, model: "ProviderModel") -> str:
        missing = self.required - set(model.capabilities)
        return (
            f"{model.provider}::{model.model} missing capabilities: {sorted(missing)}"
        )


class ProviderConstraint(Constraint):
    """Reject models from providers not in an explicit allow-list.

    Args:
        allowed_providers: Names of providers that are permitted.
    """

    def __init__(self, allowed_providers: list[str]) -> None:
        self.allowed: set[str] = set(allowed_providers)

    def evaluate(self, model: "ProviderModel") -> bool:
        return model.provider in self.allowed

    def describe(self) -> str:
        return f"ProviderConstraint(allowed={sorted(self.allowed)})"

    def failure_reason(self, model: "ProviderModel") -> str:
        return (
            f"{model.provider}::{model.model} provider not in allowed: "
            f"{sorted(self.allowed)}"
        )


class ConstraintSet:
    """An ordered set of constraints applied as a conjunction (AND).

    All constraints must pass for a model to be accepted. Models that fail
    any constraint are collected in the ``failures`` dict for diagnostics.

    Usage::

        cs = ConstraintSet()
        cs.add(LatencyConstraint(500)).add(CostConstraint(0.001))
        passed, failures = cs.filter(models)
    """

    def __init__(self) -> None:
        self._constraints: list[Constraint] = []

    def add(self, constraint: Constraint) -> "ConstraintSet":
        """Append a constraint and return ``self`` for chaining.

        Args:
            constraint: The constraint to add.

        Returns:
            This ``ConstraintSet`` instance.
        """
        self._constraints.append(constraint)
        return self

    def evaluate_all(
        self, model: "ProviderModel"
    ) -> tuple[bool, list[str]]:
        """Test all constraints against a single model.

        Args:
            model: The model to test.

        Returns:
            A ``(passed, failure_reasons)`` tuple. ``passed`` is True only
            when all constraints pass.
        """
        failures = [
            c.failure_reason(model)
            for c in self._constraints
            if not c.evaluate(model)
        ]
        return len(failures) == 0, failures

    def filter(
        self, models: "list[ProviderModel]"
    ) -> tuple[list["ProviderModel"], dict[str, list[str]]]:
        """Partition models into passing and failing.

        Args:
            models: Candidate models to filter.

        Returns:
            A ``(passed, failures_dict)`` tuple where ``failures_dict``
            maps ``model_identity_key -> list[failure_reason]``.
        """
        passed: list["ProviderModel"] = []
        failures: dict[str, list[str]] = {}
        for m in models:
            ok, reasons = self.evaluate_all(m)
            if ok:
                passed.append(m)
            else:
                failures[m.identity.key] = reasons
        return passed, failures

    @property
    def constraints(self) -> list[Constraint]:
        """Return a copy of the constraint list."""
        return list(self._constraints)

    def __len__(self) -> int:
        return len(self._constraints)
