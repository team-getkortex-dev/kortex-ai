"""Task routing engine for Kortex.

Routes tasks to the most appropriate model/provider based on cost, latency,
capability constraints, and complexity hints. Supports pluggable routing
strategies via the RoutingStrategy protocol.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4

import structlog

from kortex.core.capabilities import validate_capabilities
from kortex.core.exceptions import RoutingFailedError
from kortex.core.types import ExecutionEvent, ModelIdentity, RoutingDecision, TaskSpec

if TYPE_CHECKING:
    from kortex.core.metrics import ObservedMetrics
    from kortex.core.policy import RoutingPolicy
    from kortex.router.constraints import ConstraintSet
    from kortex.router.decision_cache import RoutingDecisionCache

logger = structlog.get_logger()

# Rough estimate: average task uses ~1k input tokens and ~500 output tokens.
_DEFAULT_INPUT_TOKENS = 1000
_DEFAULT_OUTPUT_TOKENS = 500


@dataclass(frozen=True)
class ProviderModel:
    """An available model that the router can select from.

    Args:
        provider: Provider name (e.g. "anthropic", "openai").
        model: Model identifier (e.g. "claude-sonnet-4-20250514").
        cost_per_1k_input_tokens: Cost in USD per 1k input tokens.
        cost_per_1k_output_tokens: Cost in USD per 1k output tokens.
        avg_latency_ms: Average end-to-end latency in milliseconds.
        capabilities: List of capability tags the model supports.
        max_context_tokens: Maximum context window size.
        tier: Performance tier — "fast", "balanced", or "powerful".
    """

    provider: str
    model: str
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    avg_latency_ms: float
    capabilities: list[str] = field(default_factory=list)
    max_context_tokens: int = 128_000
    tier: Literal["fast", "balanced", "powerful"] = "balanced"
    model_version: str = ""
    endpoint_id: str = ""

    @property
    def identity(self) -> ModelIdentity:
        """Return a composite identity that uniquely identifies this model."""
        return ModelIdentity(
            provider=self.provider,
            model_name=self.model,
            model_version=self.model_version,
            endpoint_id=self.endpoint_id,
        )

    def estimated_cost(
        self,
        input_tokens: int = _DEFAULT_INPUT_TOKENS,
        output_tokens: int = _DEFAULT_OUTPUT_TOKENS,
    ) -> float:
        """Estimate total cost for a request with the given token counts."""
        return (
            self.cost_per_1k_input_tokens * input_tokens / 1000
            + self.cost_per_1k_output_tokens * output_tokens / 1000
        )


class RoutingStrategy(Protocol):
    """Protocol for pluggable routing strategies."""

    async def select(
        self, task: TaskSpec, candidates: list[ProviderModel]
    ) -> RoutingDecision:
        """Select the best model for the given task from the candidate list."""
        ...


def _cheapest(models: list[ProviderModel]) -> ProviderModel:
    return min(models, key=lambda m: m.estimated_cost())


def _most_capable(models: list[ProviderModel]) -> ProviderModel:
    return max(models, key=lambda m: (m.max_context_tokens, -m.estimated_cost()))


class HeuristicRoutingStrategy:
    """Rule-based routing strategy.

    Filters candidates by cost ceiling, latency SLA, and required capabilities,
    then selects based on the task's complexity hint.
    """

    async def select(
        self, task: TaskSpec, candidates: list[ProviderModel]
    ) -> RoutingDecision:
        """Select the best model using heuristic rules.

        Raises:
            RoutingFailedError: If no candidates survive filtering.
        """
        remaining = list(candidates)
        filter_log: list[str] = []

        # --- cost ceiling filter ---
        if task.cost_ceiling_usd is not None:
            before = len(remaining)
            remaining = [
                m for m in remaining if m.estimated_cost() <= task.cost_ceiling_usd
            ]
            removed = before - len(remaining)
            if removed:
                filter_log.append(
                    f"{removed} model(s) exceeded cost ceiling ${task.cost_ceiling_usd}"
                )

        # --- latency SLA filter ---
        if task.latency_sla_ms is not None:
            before = len(remaining)
            remaining = [m for m in remaining if m.avg_latency_ms <= task.latency_sla_ms]
            removed = before - len(remaining)
            if removed:
                filter_log.append(
                    f"{removed} model(s) exceeded latency SLA {task.latency_sla_ms}ms"
                )

        # --- capabilities filter ---
        if task.required_capabilities:
            required = set(task.required_capabilities)
            before = len(remaining)
            remaining = [m for m in remaining if required.issubset(set(m.capabilities))]
            removed = before - len(remaining)
            if removed:
                filter_log.append(
                    f"{removed} model(s) missing required capabilities {sorted(required)}"
                )

        if not remaining:
            raise RoutingFailedError(
                f"No models available for task {task.task_id}. "
                f"Filters applied: {'; '.join(filter_log) if filter_log else 'none'}. "
                f"Started with {len(candidates)} candidate(s)."
            )

        # --- tier-based selection ---
        chosen, reasoning = self._select_by_complexity(task.complexity_hint, remaining)

        # --- fallback: next-best candidate that isn't the chosen one ---
        others = [m for m in remaining if m.identity != chosen.identity]
        fallback = _cheapest(others).model if others else None
        fallback_provider = _cheapest(others).provider if others else None

        if filter_log:
            reasoning += " Filters: " + "; ".join(filter_log) + "."

        return RoutingDecision(
            task_id=task.task_id,
            chosen_provider=chosen.provider,
            chosen_model=chosen.model,
            chosen_model_identity=chosen.identity.key,
            reasoning=reasoning,
            estimated_cost_usd=chosen.estimated_cost(),
            estimated_latency_ms=chosen.avg_latency_ms,
            fallback_model=fallback,
            fallback_provider=fallback_provider,
        )

    def _select_by_complexity(
        self,
        complexity: Literal["simple", "moderate", "complex"],
        models: list[ProviderModel],
    ) -> tuple[ProviderModel, str]:
        """Pick a model matching the complexity tier, with fallback to adjacent tiers."""
        tier_map: dict[str, list[ProviderModel]] = {"fast": [], "balanced": [], "powerful": []}
        for m in models:
            tier_map[m.tier].append(m)

        if complexity == "simple":
            if tier_map["fast"]:
                chosen = _cheapest(tier_map["fast"])
                return chosen, f"Simple task -> cheapest fast-tier model: {chosen.model}."
            chosen = _cheapest(models)
            return chosen, f"Simple task -> no fast-tier available, cheapest overall: {chosen.model}."

        if complexity == "moderate":
            if tier_map["balanced"]:
                chosen = _cheapest(tier_map["balanced"])
                return chosen, f"Moderate task -> cheapest balanced-tier model: {chosen.model}."
            if tier_map["fast"]:
                chosen = _cheapest(tier_map["fast"])
                return chosen, f"Moderate task -> no balanced-tier, fell back to fast: {chosen.model}."
            chosen = _cheapest(models)
            return chosen, f"Moderate task -> cheapest available: {chosen.model}."

        # complex
        if tier_map["powerful"]:
            chosen = _most_capable(tier_map["powerful"])
            return chosen, f"Complex task -> most capable powerful-tier model: {chosen.model}."
        if tier_map["balanced"]:
            chosen = _most_capable(tier_map["balanced"])
            return chosen, f"Complex task -> no powerful-tier, fell back to balanced: {chosen.model}."
        chosen = _most_capable(models)
        return chosen, f"Complex task -> most capable available: {chosen.model}."


class Router:
    """Central routing engine that maps tasks to models.

    Supports two routing modes:
    - **Policy-based** (via ``set_policy``): Uses ``PolicyRouter`` for
      composable, explainable, deterministic routing.
    - **Strategy-based** (default): Uses a ``RoutingStrategy`` implementation
      (defaults to ``HeuristicRoutingStrategy``).

    If a policy is set, it takes precedence over the strategy.

    Args:
        strategy: The routing strategy to use. Defaults to HeuristicRoutingStrategy.
    """

    def __init__(
        self,
        strategy: RoutingStrategy | None = None,
        metrics: ObservedMetrics | None = None,
        enable_decision_cache: bool = True,
        decision_cache_size: int = 10_000,
    ) -> None:
        self._strategy: RoutingStrategy = strategy or HeuristicRoutingStrategy()
        self._models: dict[str, ProviderModel] = {}
        self._policy: "RoutingPolicy | None" = None
        self._metrics: ObservedMetrics | None = metrics
        self._constraint_set: "ConstraintSet | None" = None
        self._log = structlog.get_logger(component="router")

        # Routing decision cache — skips full computation for repeated (task, policy)
        self._decision_cache: RoutingDecisionCache | None = None
        if enable_decision_cache:
            from kortex.router.decision_cache import RoutingDecisionCache

            self._decision_cache = RoutingDecisionCache(max_size=decision_cache_size)

    @property
    def models(self) -> list[ProviderModel]:
        """Return all registered models."""
        return list(self._models.values())

    def register_model(self, model: ProviderModel) -> None:
        """Add a model to the registry.

        Uses the composite identity key (provider::model_name) so that
        two providers serving the same model name do not collide.

        Validates that the model's capabilities are all canonical values.

        Args:
            model: The provider model to register.

        Raises:
            ValueError: If the model has invalid capabilities.
        """
        if model.capabilities:
            validate_capabilities(model.capabilities)
        self._models[model.identity.key] = model

    def remove_model(self, model_name: str, provider: str | None = None) -> None:
        """Remove a model from the registry.

        Args:
            model_name: The model identifier to remove.
            provider: If given, remove only the entry for this provider.
                If omitted, remove the first entry whose model field matches
                (backward-compatible behavior).
        """
        if provider is not None:
            identity_key = f"{provider}::{model_name}"
            self._models.pop(identity_key, None)
        else:
            # Backward compat: remove first match by model name
            key_to_remove = None
            for key, m in self._models.items():
                if m.model == model_name:
                    key_to_remove = key
                    break
            if key_to_remove is not None:
                self._models.pop(key_to_remove, None)

    def set_policy(self, policy: "RoutingPolicy") -> None:
        """Set a routing policy. When set, policy-based routing takes precedence.

        Args:
            policy: The routing policy to use.
        """
        from kortex.core.policy import RoutingPolicy  # noqa: F811

        self._policy = policy
        self._log.info("policy_set", policy_name=policy.name)

    def get_policy(self) -> "RoutingPolicy | None":
        """Return the current routing policy, or None if not set."""
        return self._policy

    def set_constraints(self, constraint_set: "ConstraintSet") -> None:
        """Attach a ConstraintSet that filters candidates before routing.

        Once set, every ``route()`` call applies these constraints before
        heuristic or policy selection. Models that fail any constraint are
        excluded and reported via ``RoutingFailedError`` diagnostics.

        Args:
            constraint_set: The constraint set to apply.
        """
        self._constraint_set = constraint_set
        self._log.info("constraint_set_attached", count=len(constraint_set))

    def get_constraints(self) -> "ConstraintSet | None":
        """Return the attached ConstraintSet, or None if not configured.

        Returns:
            The active ConstraintSet, or None.
        """
        return self._constraint_set

    def set_metrics(self, metrics: ObservedMetrics) -> None:
        """Attach an ObservedMetrics tracker to enable EWMA-based calibration.

        Once set, the router adjusts candidate model metadata (latency and
        cost) using observed values before scoring. Models with no
        observations are left unchanged.

        Args:
            metrics: The ObservedMetrics instance to use.
        """
        self._metrics = metrics
        self._log.info("observed_metrics_attached")

    def get_metrics(self) -> ObservedMetrics | None:
        """Return the attached ObservedMetrics, or None.

        Returns:
            The ObservedMetrics instance, or None if not configured.
        """
        return self._metrics

    def _apply_observed_metrics(
        self,
        models: list[ProviderModel],
        complexity_class: str = "medium",
    ) -> list[ProviderModel]:
        """Return models with latency/cost adjusted by EWMA observations.

        For each model that has at least one observation in the attached
        metrics, replaces ``avg_latency_ms`` with the EWMA latency, and
        scales the per-token cost rates so that ``estimated_cost()`` returns
        approximately the observed cost.

        Looks up observations for the given ``complexity_class`` first; the
        ObservedMetrics tracker falls back to the ``"medium"`` bucket when no
        stratified data exists for this class. Models with no observations at
        all are returned unchanged.

        Args:
            models: The candidate model list to adjust.
            complexity_class: Task complexity class for stratified lookup.

        Returns:
            A new list with adjusted ProviderModel instances.
        """
        import dataclasses

        if self._metrics is None:
            return models

        adjusted: list[ProviderModel] = []
        for m in models:
            obs_lat = self._metrics.get_latency(m.identity.key, complexity_class)
            obs_cost = self._metrics.get_cost(m.identity.key, complexity_class)

            if obs_lat is None and obs_cost is None:
                adjusted.append(m)
                continue

            new_lat = obs_lat if obs_lat is not None else m.avg_latency_ms

            # Scale cost rates proportionally so estimated_cost() ≈ observed cost
            if obs_cost is not None and obs_cost > 0:
                estimated = m.estimated_cost()
                if estimated > 0:
                    ratio = obs_cost / estimated
                    new_input = m.cost_per_1k_input_tokens * ratio
                    new_output = m.cost_per_1k_output_tokens * ratio
                else:
                    new_input = m.cost_per_1k_input_tokens
                    new_output = m.cost_per_1k_output_tokens
            else:
                new_input = m.cost_per_1k_input_tokens
                new_output = m.cost_per_1k_output_tokens

            adjusted.append(dataclasses.replace(
                m,
                avg_latency_ms=new_lat,
                cost_per_1k_input_tokens=new_input,
                cost_per_1k_output_tokens=new_output,
            ))

        return adjusted

    async def route(self, task: TaskSpec) -> RoutingDecision:
        """Route a task to the best available model.

        If a policy is set, uses ``PolicyRouter`` for composable evaluation.
        Otherwise, falls back to the configured ``RoutingStrategy``.

        Args:
            task: The task specification to route.

        Returns:
            The RoutingDecision for the task.

        Raises:
            RoutingFailedError: If no suitable model is found.
        """
        # Fast path: return cached decision for identical (task, policy) combos.
        # Skip cache when adaptive EWMA metrics are attached — metrics change
        # over time, so a cached decision can become stale after any provider call.
        if self._decision_cache is not None and self._metrics is None:
            cached = self._decision_cache.get(task, self._policy)
            if cached is not None:
                return cached

        candidates = self._apply_observed_metrics(
            list(self._models.values()),
            complexity_class=task.complexity_class,
        )

        # Apply hard constraints before heuristic / policy selection
        constraint_failures: dict[str, list[str]] = {}
        if self._constraint_set is not None and len(self._constraint_set) > 0:
            candidates, constraint_failures = self._constraint_set.filter(candidates)
            if not candidates:
                from kortex.router.diagnostics import RoutingDiagnostics

                all_models = list(self._models.values())
                msg = RoutingDiagnostics().explain_failure(
                    task,
                    all_models,
                    constraint_failures=constraint_failures,
                )
                failed_models = [
                    (k, "; ".join(v)) for k, v in constraint_failures.items()
                ]
                # Find the closest model (fewest failures)
                closest: str | None = None
                if constraint_failures:
                    best_key = min(constraint_failures, key=lambda k: len(constraint_failures[k]))
                    closest = best_key
                raise RoutingFailedError(
                    msg,
                    failed_models=failed_models,
                    closest_model=closest,
                    suggestion=(
                        "Relax constraints or register models that satisfy all constraints."
                    ),
                )

        if self._policy is not None:
            decision = await self._route_with_policy(task, candidates)
        else:
            decision = await self._strategy.select(task, candidates)

        self._log.info(
            "routing_decision",
            task_id=decision.task_id,
            chosen_provider=decision.chosen_provider,
            chosen_model=decision.chosen_model,
            reasoning=decision.reasoning,
            estimated_cost_usd=decision.estimated_cost_usd,
            estimated_latency_ms=decision.estimated_latency_ms,
            fallback_model=decision.fallback_model,
        )

        # Store in decision cache for future identical requests (static routing only)
        if self._decision_cache is not None and self._metrics is None:
            self._decision_cache.set(task, self._policy, decision)

        return decision

    async def route_batch(self, tasks: list[TaskSpec]) -> list[RoutingDecision]:
        """Route multiple tasks concurrently.

        Uses ``asyncio.gather`` to run all ``route()`` calls in parallel.
        Individual failures raise ``RoutingFailedError`` for that task but
        do not prevent other tasks from being routed — exceptions propagate
        from ``gather`` in index order.

        Args:
            tasks: The tasks to route.

        Returns:
            A list of ``RoutingDecision`` objects in the same order as
            ``tasks``.

        Raises:
            RoutingFailedError: If any task cannot be routed.
        """
        return list(await asyncio.gather(*[self.route(t) for t in tasks]))

    async def _route_with_policy(
        self, task: TaskSpec, candidates: list[ProviderModel]
    ) -> RoutingDecision:
        """Route using the PolicyRouter and convert to RoutingDecision."""
        from kortex.core.policy import PolicyRouter

        assert self._policy is not None
        router = PolicyRouter(self._policy, candidates)
        evaluation = await router.evaluate(task)

        return RoutingDecision(
            task_id=task.task_id,
            chosen_provider=evaluation.chosen.provider,
            chosen_model=evaluation.chosen.model,
            chosen_model_identity=evaluation.chosen.identity.key,
            reasoning=evaluation.explanation,
            estimated_cost_usd=evaluation.chosen.estimated_cost(),
            estimated_latency_ms=evaluation.chosen.avg_latency_ms,
            fallback_model=evaluation.fallback.model if evaluation.fallback else None,
            fallback_provider=(
                evaluation.fallback.provider if evaluation.fallback else None
            ),
            metadata={
                "policy_name": evaluation.policy_name,
                "constraints_applied": evaluation.constraints_applied,
                "candidates_count": len(evaluation.all_candidates),
                "eliminated_count": len(evaluation.eliminated),
                "score_breakdown": (
                    evaluation.all_candidates[0].score_breakdown
                    if evaluation.all_candidates
                    else {}
                ),
            },
        )
