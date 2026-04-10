"""Composable routing policy engine for Kortex.

Provides declarative, serializable routing policies that separate hard
constraints (what is allowed) from soft objectives (what is preferred)
and fallback rules (what to do when the primary fails).

Policies are deterministic: same task + same models + same policy = same result.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import structlog

from kortex.core.capabilities import validate_capabilities
from kortex.core.exceptions import RoutingFailedError
from kortex.core.types import RoutingDecision, TaskSpec

# Avoid circular import — ProviderModel is referenced by type only at
# module level and imported concretely inside methods that need it.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kortex.core.router import ProviderModel

logger = structlog.get_logger(component="policy")

# Rough estimate: average task uses ~1k input tokens and ~500 output tokens.
_DEFAULT_INPUT_TOKENS = 1000
_DEFAULT_OUTPUT_TOKENS = 500


# ---------------------------------------------------------------------------
# Policy building blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingConstraint:
    """Hard constraints that eliminate models from consideration.

    All constraints are AND-combined: a model must satisfy every non-None
    constraint to remain a candidate.

    Args:
        max_cost_usd: Maximum estimated cost per request (None = no limit).
        max_latency_ms: Maximum average latency (None = no limit).
        required_capabilities: Model must have all of these.
        allowed_providers: If set, only these providers are permitted.
        denied_providers: These providers are always excluded.
        allowed_models: If set, only these model names are permitted.
        denied_models: These model names are always excluded.
    """

    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    required_capabilities: list[str] = field(default_factory=list)
    allowed_providers: list[str] | None = None
    denied_providers: list[str] = field(default_factory=list)
    allowed_models: list[str] | None = None
    denied_models: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RoutingObjective:
    """Soft objectives that score and rank surviving candidates.

    Args:
        minimize: Primary optimization axis ("cost", "latency", or "none").
        prefer_tier: Boost models in this tier ("fast", "balanced",
            "powerful", or "any" for no preference).
        prefer_provider: Boost models from this provider (None = no pref).
    """

    minimize: Literal["cost", "latency", "none"] = "cost"
    prefer_tier: Literal["fast", "balanced", "powerful", "any"] = "any"
    prefer_provider: str | None = None


@dataclass(frozen=True)
class FallbackRule:
    """How to pick a fallback model when the primary fails.

    Args:
        strategy: Selection method for the fallback.
        explicit_model_identity: Composite key (provider::model) used
            when strategy is "explicit".
    """

    strategy: Literal["next_cheapest", "next_fastest", "same_tier", "explicit"] = (
        "next_cheapest"
    )
    explicit_model_identity: str | None = None


# ---------------------------------------------------------------------------
# RoutingPolicy — the top-level composable unit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingPolicy:
    """A complete, serializable routing policy.

    Combines constraints, objectives, and fallback rules into a single
    declarative unit that fully determines model selection.

    Args:
        name: Human-readable policy name.
        constraints: Hard constraints for candidate elimination.
        objective: Soft objectives for candidate scoring.
        fallback: Rule for selecting the fallback model.
        budget_ceiling_usd: Total budget across all pipeline steps (None = unlimited).
        description: Free-text description of the policy's intent.
    """

    name: str = "default"
    constraints: RoutingConstraint = field(default_factory=RoutingConstraint)
    objective: RoutingObjective = field(default_factory=RoutingObjective)
    fallback: FallbackRule = field(default_factory=FallbackRule)
    budget_ceiling_usd: float | None = None
    description: str = ""

    # -- factory methods ----------------------------------------------------

    @staticmethod
    def cost_optimized() -> RoutingPolicy:
        """Return a policy that minimizes cost."""
        return RoutingPolicy(
            name="cost_optimized",
            objective=RoutingObjective(minimize="cost", prefer_tier="fast"),
            fallback=FallbackRule(strategy="next_cheapest"),
            description="Minimize cost, prefer fast-tier models.",
        )

    @staticmethod
    def latency_optimized() -> RoutingPolicy:
        """Return a policy that minimizes latency."""
        return RoutingPolicy(
            name="latency_optimized",
            objective=RoutingObjective(minimize="latency", prefer_tier="fast"),
            fallback=FallbackRule(strategy="next_fastest"),
            description="Minimize latency, prefer fast-tier models.",
        )

    @staticmethod
    def quality_optimized() -> RoutingPolicy:
        """Return a policy that prefers powerful-tier models."""
        return RoutingPolicy(
            name="quality_optimized",
            objective=RoutingObjective(minimize="none", prefer_tier="powerful"),
            fallback=FallbackRule(strategy="same_tier"),
            description="Prefer powerful-tier models for maximum quality.",
        )

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "budget_ceiling_usd": self.budget_ceiling_usd,
            "constraints": {
                "max_cost_usd": self.constraints.max_cost_usd,
                "max_latency_ms": self.constraints.max_latency_ms,
                "required_capabilities": list(self.constraints.required_capabilities),
                "allowed_providers": (
                    list(self.constraints.allowed_providers)
                    if self.constraints.allowed_providers is not None
                    else None
                ),
                "denied_providers": list(self.constraints.denied_providers),
                "allowed_models": (
                    list(self.constraints.allowed_models)
                    if self.constraints.allowed_models is not None
                    else None
                ),
                "denied_models": list(self.constraints.denied_models),
            },
            "objective": {
                "minimize": self.objective.minimize,
                "prefer_tier": self.objective.prefer_tier,
                "prefer_provider": self.objective.prefer_provider,
            },
            "fallback": {
                "strategy": self.fallback.strategy,
                "explicit_model_identity": self.fallback.explicit_model_identity,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingPolicy:
        """Deserialize from a dict (inverse of ``to_dict``)."""
        c = data.get("constraints", {})
        o = data.get("objective", {})
        f = data.get("fallback", {})

        return cls(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            budget_ceiling_usd=data.get("budget_ceiling_usd"),
            constraints=RoutingConstraint(
                max_cost_usd=c.get("max_cost_usd"),
                max_latency_ms=c.get("max_latency_ms"),
                required_capabilities=c.get("required_capabilities", []),
                allowed_providers=c.get("allowed_providers"),
                denied_providers=c.get("denied_providers", []),
                allowed_models=c.get("allowed_models"),
                denied_models=c.get("denied_models", []),
            ),
            objective=RoutingObjective(
                minimize=o.get("minimize", "cost"),
                prefer_tier=o.get("prefer_tier", "any"),
                prefer_provider=o.get("prefer_provider"),
            ),
            fallback=FallbackRule(
                strategy=f.get("strategy", "next_cheapest"),
                explicit_model_identity=f.get("explicit_model_identity"),
            ),
        )

    def to_toml(self) -> str:
        """Serialize to a TOML string."""
        lines: list[str] = []
        lines.append(f'name = "{self.name}"')
        lines.append(f'description = "{self.description}"')
        if self.budget_ceiling_usd is not None:
            lines.append(f"budget_ceiling_usd = {self.budget_ceiling_usd}")
        else:
            lines.append("# budget_ceiling_usd not set")

        lines.append("")
        lines.append("[constraints]")
        if self.constraints.max_cost_usd is not None:
            lines.append(f"max_cost_usd = {self.constraints.max_cost_usd}")
        if self.constraints.max_latency_ms is not None:
            lines.append(f"max_latency_ms = {self.constraints.max_latency_ms}")
        if self.constraints.required_capabilities:
            caps = ", ".join(f'"{c}"' for c in self.constraints.required_capabilities)
            lines.append(f"required_capabilities = [{caps}]")
        if self.constraints.allowed_providers is not None:
            provs = ", ".join(f'"{p}"' for p in self.constraints.allowed_providers)
            lines.append(f"allowed_providers = [{provs}]")
        if self.constraints.denied_providers:
            provs = ", ".join(f'"{p}"' for p in self.constraints.denied_providers)
            lines.append(f"denied_providers = [{provs}]")
        if self.constraints.allowed_models is not None:
            models = ", ".join(f'"{m}"' for m in self.constraints.allowed_models)
            lines.append(f"allowed_models = [{models}]")
        if self.constraints.denied_models:
            models = ", ".join(f'"{m}"' for m in self.constraints.denied_models)
            lines.append(f"denied_models = [{models}]")

        lines.append("")
        lines.append("[objective]")
        lines.append(f'minimize = "{self.objective.minimize}"')
        lines.append(f'prefer_tier = "{self.objective.prefer_tier}"')
        if self.objective.prefer_provider is not None:
            lines.append(f'prefer_provider = "{self.objective.prefer_provider}"')

        lines.append("")
        lines.append("[fallback]")
        lines.append(f'strategy = "{self.fallback.strategy}"')
        if self.fallback.explicit_model_identity is not None:
            lines.append(
                f'explicit_model_identity = "{self.fallback.explicit_model_identity}"'
            )

        lines.append("")
        return "\n".join(lines)

    @classmethod
    def from_toml(cls, path: str) -> RoutingPolicy:
        """Deserialize from a TOML file.

        Args:
            path: Path to the .toml file.

        Returns:
            A RoutingPolicy parsed from the file.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)
        # Support both flat keys and [policy]-wrapped format
        if "policy" in data and isinstance(data["policy"], dict):
            data = data["policy"]
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Evaluation result types
# ---------------------------------------------------------------------------


@dataclass
class ScoredCandidate:
    """A candidate model with its computed score and breakdown.

    Args:
        model: The provider model.
        score: Composite score (higher is better).
        score_breakdown: Individual score components.
    """

    model: "ProviderModel"
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class EliminatedCandidate:
    """A model that was eliminated during constraint filtering.

    Args:
        model: The eliminated provider model.
        reason: Human-readable explanation for elimination.
    """

    model: "ProviderModel"
    reason: str


@dataclass
class PolicyEvaluation:
    """Complete result of evaluating a policy against available models.

    Provides full transparency: who was chosen, who was eliminated and why,
    and a human-readable explanation of the decision.

    Args:
        chosen: The selected model.
        fallback: The fallback model (if any).
        all_candidates: All surviving models ranked by score (descending).
        eliminated: All eliminated models with reasons.
        explanation: Human-readable decision explanation.
        policy_name: Name of the policy that produced this evaluation.
        constraints_applied: Summary of constraints that were active.
    """

    chosen: "ProviderModel"
    fallback: "ProviderModel | None"
    all_candidates: list[ScoredCandidate]
    eliminated: list[EliminatedCandidate]
    explanation: str
    policy_name: str
    constraints_applied: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PolicyRouter — the evaluation engine
# ---------------------------------------------------------------------------


class PolicyRouter:
    """Evaluates a routing policy against registered models.

    Deterministic: same task + same models + same policy = same result.
    Tie-breaking uses model identity key (lexicographic) for stability.

    Args:
        policy: The routing policy to evaluate.
        models: The registered provider models.
    """

    def __init__(self, policy: RoutingPolicy, models: list["ProviderModel"]) -> None:
        self._policy = policy
        self._models = models
        self._log = structlog.get_logger(component="policy_router")

    async def evaluate(self, task: TaskSpec) -> PolicyEvaluation:
        """Evaluate the policy for the given task.

        Args:
            task: The task specification to route.

        Returns:
            A PolicyEvaluation with chosen model, fallback, scores, and
            elimination details.

        Raises:
            RoutingFailedError: If all candidates are eliminated.
        """
        candidates = list(self._models)
        eliminated: list[EliminatedCandidate] = []
        constraints = self._policy.constraints

        # Merge task-level constraints into policy constraints
        effective_max_cost = constraints.max_cost_usd
        if task.cost_ceiling_usd is not None:
            if effective_max_cost is None:
                effective_max_cost = task.cost_ceiling_usd
            else:
                effective_max_cost = min(effective_max_cost, task.cost_ceiling_usd)

        effective_max_latency = constraints.max_latency_ms
        if task.latency_sla_ms is not None:
            if effective_max_latency is None:
                effective_max_latency = task.latency_sla_ms
            else:
                effective_max_latency = min(effective_max_latency, task.latency_sla_ms)

        all_required_caps = set(constraints.required_capabilities)
        all_required_caps.update(task.required_capabilities)

        # -- Step 1: Apply hard constraints (filter) -----------------------
        remaining: list["ProviderModel"] = []
        for m in candidates:
            reason = self._check_constraints(
                m,
                max_cost=effective_max_cost,
                max_latency=effective_max_latency,
                required_caps=all_required_caps,
                allowed_providers=constraints.allowed_providers,
                denied_providers=constraints.denied_providers,
                allowed_models=constraints.allowed_models,
                denied_models=constraints.denied_models,
            )
            if reason is not None:
                eliminated.append(EliminatedCandidate(model=m, reason=reason))
            else:
                remaining.append(m)

        if not remaining:
            elim_details = "; ".join(
                f"{e.model.provider}/{e.model.model}: {e.reason}" for e in eliminated
            )
            raise RoutingFailedError(
                f"Policy '{self._policy.name}' eliminated all {len(candidates)} "
                f"candidate(s) for task {task.task_id}. "
                f"Eliminations: {elim_details}."
            )

        # -- Step 2: Score remaining candidates ----------------------------
        scored = self._score_candidates(remaining)

        # -- Step 3: Sort by score descending, tie-break by identity key ---
        scored.sort(key=lambda s: (-s.score, s.model.identity.key))

        chosen_model = scored[0].model

        # -- Step 4: Select fallback based on FallbackRule -----------------
        fallback_model = self._select_fallback(scored, chosen_model)

        # -- Step 5: Build explanation -------------------------------------
        explanation = self._build_explanation(
            chosen_model, fallback_model, scored, eliminated
        )

        # -- Build constraints_applied summary -----------------------------
        constraints_applied: dict[str, Any] = {}
        if effective_max_cost is not None:
            constraints_applied["max_cost_usd"] = effective_max_cost
        if effective_max_latency is not None:
            constraints_applied["max_latency_ms"] = effective_max_latency
        if all_required_caps:
            constraints_applied["required_capabilities"] = sorted(all_required_caps)
        if constraints.allowed_providers is not None:
            constraints_applied["allowed_providers"] = constraints.allowed_providers
        if constraints.denied_providers:
            constraints_applied["denied_providers"] = constraints.denied_providers
        if constraints.allowed_models is not None:
            constraints_applied["allowed_models"] = constraints.allowed_models
        if constraints.denied_models:
            constraints_applied["denied_models"] = constraints.denied_models

        return PolicyEvaluation(
            chosen=chosen_model,
            fallback=fallback_model,
            all_candidates=scored,
            eliminated=eliminated,
            explanation=explanation,
            policy_name=self._policy.name,
            constraints_applied=constraints_applied,
        )

    # -- constraint checking -----------------------------------------------

    @staticmethod
    def _check_constraints(
        model: "ProviderModel",
        *,
        max_cost: float | None,
        max_latency: float | None,
        required_caps: set[str],
        allowed_providers: list[str] | None,
        denied_providers: list[str],
        allowed_models: list[str] | None,
        denied_models: list[str],
    ) -> str | None:
        """Return elimination reason if model violates a constraint, else None."""
        if max_cost is not None and model.estimated_cost() > max_cost:
            return (
                f"exceeded max_cost_usd of {max_cost} "
                f"(estimated ${model.estimated_cost():.6f})"
            )
        if max_latency is not None and model.avg_latency_ms > max_latency:
            return (
                f"exceeded max_latency_ms of {max_latency} "
                f"(avg {model.avg_latency_ms}ms)"
            )
        if required_caps:
            missing = required_caps - set(model.capabilities)
            if missing:
                return f"missing capabilities: {sorted(missing)}"
        if denied_providers and model.provider in denied_providers:
            return f"provider '{model.provider}' is denied"
        if allowed_providers is not None and model.provider not in allowed_providers:
            return f"provider '{model.provider}' not in allowed list"
        if denied_models and model.model in denied_models:
            return f"model '{model.model}' is denied"
        if allowed_models is not None and model.model not in allowed_models:
            return f"model '{model.model}' not in allowed list"
        return None

    # -- scoring -----------------------------------------------------------

    def _score_candidates(
        self, models: list["ProviderModel"]
    ) -> list[ScoredCandidate]:
        """Score each candidate according to the policy's objective."""
        obj = self._policy.objective
        scored: list[ScoredCandidate] = []

        # Compute normalizing ranges for cost and latency
        costs = [m.estimated_cost() for m in models]
        latencies = [m.avg_latency_ms for m in models]
        cost_range = max(costs) - min(costs) if len(costs) > 1 else 1.0
        latency_range = max(latencies) - min(latencies) if len(latencies) > 1 else 1.0

        # Prevent division by zero when all models have the same cost/latency
        if cost_range == 0:
            cost_range = 1.0
        if latency_range == 0:
            latency_range = 1.0

        min_cost = min(costs)
        min_latency = min(latencies)

        for m in models:
            breakdown: dict[str, float] = {}

            # Cost score: 0..1, 1 = cheapest
            cost_score = 1.0 - (m.estimated_cost() - min_cost) / cost_range
            breakdown["cost_score"] = round(cost_score, 4)

            # Latency score: 0..1, 1 = fastest
            latency_score = 1.0 - (m.avg_latency_ms - min_latency) / latency_range
            breakdown["latency_score"] = round(latency_score, 4)

            # Primary objective weight
            if obj.minimize == "cost":
                base_score = cost_score
            elif obj.minimize == "latency":
                base_score = latency_score
            else:
                base_score = 0.5  # neutral when minimize="none"

            breakdown["base_score"] = round(base_score, 4)

            # Tier bonus
            tier_bonus = 0.0
            if obj.prefer_tier != "any" and m.tier == obj.prefer_tier:
                tier_bonus = 0.1
            breakdown["tier_bonus"] = round(tier_bonus, 4)

            # Provider bonus
            provider_bonus = 0.0
            if obj.prefer_provider is not None and m.provider == obj.prefer_provider:
                provider_bonus = 0.05
            breakdown["provider_bonus"] = round(provider_bonus, 4)

            total = base_score + tier_bonus + provider_bonus
            scored.append(
                ScoredCandidate(
                    model=m,
                    score=round(total, 4),
                    score_breakdown=breakdown,
                )
            )

        return scored

    # -- fallback selection ------------------------------------------------

    def _select_fallback(
        self,
        scored: list[ScoredCandidate],
        chosen: "ProviderModel",
    ) -> "ProviderModel | None":
        """Select a fallback model according to the policy's fallback rule."""
        others = [s for s in scored if s.model.identity != chosen.identity]
        if not others:
            return None

        strategy = self._policy.fallback.strategy

        if strategy == "next_cheapest":
            return min(others, key=lambda s: s.model.estimated_cost()).model

        if strategy == "next_fastest":
            return min(others, key=lambda s: s.model.avg_latency_ms).model

        if strategy == "same_tier":
            same_tier = [s for s in others if s.model.tier == chosen.tier]
            if same_tier:
                return min(same_tier, key=lambda s: s.model.estimated_cost()).model
            # Fall through to cheapest if no same-tier alternative
            return min(others, key=lambda s: s.model.estimated_cost()).model

        if strategy == "explicit":
            identity_key = self._policy.fallback.explicit_model_identity
            if identity_key is not None:
                for s in others:
                    if s.model.identity.key == identity_key:
                        return s.model
            # Explicit model not found among candidates — return None
            return None

        return None

    # -- explanation -------------------------------------------------------

    def _build_explanation(
        self,
        chosen: "ProviderModel",
        fallback: "ProviderModel | None",
        scored: list[ScoredCandidate],
        eliminated: list[EliminatedCandidate],
    ) -> str:
        """Build a human-readable explanation of the routing decision."""
        parts: list[str] = []

        # What was chosen and why
        chosen_score = next(
            (s for s in scored if s.model.identity == chosen.identity), None
        )
        num_candidates = len(scored)
        reason_parts: list[str] = []

        obj = self._policy.objective
        if obj.minimize == "cost":
            reason_parts.append(
                f"lowest cost (${chosen.estimated_cost():.6f}/req)"
            )
        elif obj.minimize == "latency":
            reason_parts.append(f"lowest latency ({chosen.avg_latency_ms}ms)")
        if obj.prefer_tier != "any" and chosen.tier == obj.prefer_tier:
            reason_parts.append(f"{obj.prefer_tier}-tier preference")
        if obj.prefer_provider and chosen.provider == obj.prefer_provider:
            reason_parts.append(f"preferred provider '{obj.prefer_provider}'")

        why = ", ".join(reason_parts) if reason_parts else "highest composite score"

        parts.append(
            f"Selected {chosen.provider}::{chosen.model} because: {why} "
            f"among {num_candidates} candidate(s) meeting all constraints."
        )

        # What was eliminated
        if eliminated:
            elim_items = [
                f"{e.model.provider}/{e.model.model} {e.reason}" for e in eliminated
            ]
            parts.append(
                f"Eliminated {len(eliminated)} model(s): "
                + ", ".join(elim_items)
                + "."
            )

        return " ".join(parts)
