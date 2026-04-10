"""Provider cost arbitrage for Kortex routing.

Maintains a registry of equivalent models across providers (e.g.
``gpt-4o-mini`` ≈ ``claude-haiku``) and automatically routes to the
cheapest available equivalent when pricing changes.

Also tracks savings achieved by arbitrage decisions so you can measure
the ROI of multi-provider setups.

Example::

    from kortex.router.cost_arbitrage import CostArbitrage

    arbitrage = CostArbitrage()
    arbitrage.register_equivalent_models("gpt-4o-mini", "claude-haiku-4-5")
    arbitrage.register_equivalent_models("gpt-4o", "claude-sonnet-4-20250514")

    # Integrate with router
    router = Router()
    arbitrage.attach(router)

    # Prices updated from provider pricing API
    arbitrage.update_price("openai", "gpt-4o-mini", input_per_1k=0.00015, output_per_1k=0.0006)
    arbitrage.update_price("anthropic", "claude-haiku-4-5", input_per_1k=0.00025, output_per_1k=0.00125)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(component="cost_arbitrage")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelPrice:
    """Live pricing snapshot for a model.

    Args:
        provider: Provider name.
        model: Model name.
        input_per_1k: Cost per 1k input tokens in USD.
        output_per_1k: Cost per 1k output tokens in USD.
    """

    provider: str
    model: str
    input_per_1k: float
    output_per_1k: float

    def estimated_cost(
        self,
        input_tokens: int = 1000,
        output_tokens: int = 500,
    ) -> float:
        """Estimate cost for the given token counts."""
        return (
            self.input_per_1k * input_tokens / 1000
            + self.output_per_1k * output_tokens / 1000
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_per_1k": self.input_per_1k,
            "output_per_1k": self.output_per_1k,
            "estimated_cost_default": self.estimated_cost(),
        }


@dataclass
class ArbitrageDecision:
    """Result of an arbitrage lookup.

    Args:
        requested_model: The originally requested model name.
        chosen_provider: The provider chosen by arbitrage.
        chosen_model: The model chosen by arbitrage.
        original_provider: The provider that would have been used without arbitrage.
        original_cost: Estimated cost without arbitrage.
        arbitrage_cost: Estimated cost with arbitrage.
        savings_usd: Savings achieved (may be negative if cheaper wasn't available).
        reason: Human-readable explanation.
    """

    requested_model: str
    chosen_provider: str
    chosen_model: str
    original_provider: str
    original_cost: float
    arbitrage_cost: float
    savings_usd: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "chosen_provider": self.chosen_provider,
            "chosen_model": self.chosen_model,
            "original_provider": self.original_provider,
            "original_cost": round(self.original_cost, 6),
            "arbitrage_cost": round(self.arbitrage_cost, 6),
            "savings_usd": round(self.savings_usd, 6),
            "reason": self.reason,
        }


@dataclass
class SavingsReport:
    """Cumulative arbitrage savings report.

    Args:
        total_decisions: Total arbitrage decisions made.
        total_savings_usd: Total USD saved.
        total_original_cost_usd: What costs would have been without arbitrage.
        total_arbitrage_cost_usd: Actual costs with arbitrage.
        savings_pct: Percentage savings.
        decisions: History of arbitrage decisions.
    """

    total_decisions: int
    total_savings_usd: float
    total_original_cost_usd: float
    total_arbitrage_cost_usd: float
    savings_pct: float
    decisions: list[ArbitrageDecision]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "total_savings_usd": round(self.total_savings_usd, 6),
            "total_original_cost_usd": round(self.total_original_cost_usd, 6),
            "total_arbitrage_cost_usd": round(self.total_arbitrage_cost_usd, 6),
            "savings_pct": round(self.savings_pct, 2),
        }

    def summary(self) -> str:
        return (
            f"Arbitrage savings: ${self.total_savings_usd:.4f} "
            f"({self.savings_pct:.1f}%) over {self.total_decisions} decisions. "
            f"Original: ${self.total_original_cost_usd:.4f}, "
            f"Actual: ${self.total_arbitrage_cost_usd:.4f}"
        )


# ---------------------------------------------------------------------------
# CostArbitrage
# ---------------------------------------------------------------------------


class CostArbitrage:
    """Registry of equivalent models with dynamic pricing and arbitrage routing.

    Maintains:
    - Groups of equivalent models (same quality tier, different providers)
    - Live pricing for each model
    - A running savings ledger

    Args:
        input_tokens: Default number of input tokens for cost estimation.
        output_tokens: Default number of output tokens for cost estimation.
    """

    def __init__(
        self,
        input_tokens: int = 1000,
        output_tokens: int = 500,
    ) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        # model_name → group_id
        self._model_to_group: dict[str, int] = {}
        # group_id → set of model names
        self._groups: dict[int, set[str]] = {}
        self._next_group_id = 0
        # (provider, model) → ModelPrice
        self._prices: dict[tuple[str, str], ModelPrice] = {}
        # (model_name) → primary provider (first one registered)
        self._primary_provider: dict[str, str] = {}
        # Savings tracking
        self._decisions: list[ArbitrageDecision] = []
        self._total_savings = 0.0
        self._total_original = 0.0
        self._total_arbitrage = 0.0
        self._log = structlog.get_logger(component="cost_arbitrage")

    # ------------------------------------------------------------------
    # Model equivalence groups
    # ------------------------------------------------------------------

    def register_equivalent_models(self, *model_names: str) -> None:
        """Register a group of equivalent models.

        All models in the group are considered interchangeable for arbitrage.
        If any model is already in a group, the groups are merged.

        Args:
            *model_names: Model names to group together.
        """
        if len(model_names) < 2:
            raise ValueError("Need at least 2 models to form an equivalence group")

        # Find existing groups to merge
        existing_group_ids: set[int] = set()
        for name in model_names:
            if name in self._model_to_group:
                existing_group_ids.add(self._model_to_group[name])

        if not existing_group_ids:
            # Create new group
            gid = self._next_group_id
            self._next_group_id += 1
            self._groups[gid] = set(model_names)
            for name in model_names:
                self._model_to_group[name] = gid
        else:
            # Merge into the first existing group
            target_gid = min(existing_group_ids)
            for gid in existing_group_ids:
                if gid != target_gid:
                    # Absorb this group into target
                    for name in self._groups.pop(gid, set()):
                        self._model_to_group[name] = target_gid
                        self._groups[target_gid].add(name)
            # Add new models to target group
            for name in model_names:
                self._model_to_group[name] = target_gid
                self._groups[target_gid].add(name)

        self._log.debug(
            "equivalent_models_registered",
            models=list(model_names),
        )

    def get_equivalent_models(self, model_name: str) -> list[str]:
        """Return all models equivalent to the given model.

        Args:
            model_name: The model to look up.

        Returns:
            List of equivalent model names (including the model itself).
            Empty list if the model has no registered equivalents.
        """
        gid = self._model_to_group.get(model_name)
        if gid is None:
            return []
        return list(self._groups[gid])

    # ------------------------------------------------------------------
    # Dynamic pricing
    # ------------------------------------------------------------------

    def update_price(
        self,
        provider: str,
        model: str,
        input_per_1k: float,
        output_per_1k: float,
    ) -> None:
        """Update the live price for a specific provider/model.

        Args:
            provider: Provider name.
            model: Model name.
            input_per_1k: Updated cost per 1k input tokens in USD.
            output_per_1k: Updated cost per 1k output tokens in USD.
        """
        key = (provider, model)
        self._prices[key] = ModelPrice(
            provider=provider,
            model=model,
            input_per_1k=input_per_1k,
            output_per_1k=output_per_1k,
        )
        if model not in self._primary_provider:
            self._primary_provider[model] = provider
        self._log.debug(
            "price_updated",
            provider=provider,
            model=model,
            input_per_1k=input_per_1k,
        )

    def get_price(self, provider: str, model: str) -> ModelPrice | None:
        """Return the current price for a provider/model, or None.

        Args:
            provider: Provider name.
            model: Model name.

        Returns:
            ModelPrice or None if no price is registered.
        """
        return self._prices.get((provider, model))

    def list_prices(self) -> list[ModelPrice]:
        """Return all registered prices.

        Returns:
            List of ModelPrice objects.
        """
        return list(self._prices.values())

    # ------------------------------------------------------------------
    # Arbitrage
    # ------------------------------------------------------------------

    def find_cheapest(
        self,
        model_name: str,
        excluded_providers: list[str] | None = None,
    ) -> ArbitrageDecision | None:
        """Find the cheapest equivalent model across all providers.

        Args:
            model_name: The model to find an equivalent for.
            excluded_providers: Providers to exclude from consideration.

        Returns:
            ArbitrageDecision with the cheapest option, or None if no
            equivalents have registered prices.
        """
        equivalents = self.get_equivalent_models(model_name)
        if not equivalents:
            return None

        excluded = set(excluded_providers or [])

        # Gather all (provider, model, cost) candidates
        candidates: list[tuple[str, str, float]] = []
        for eq_model in equivalents:
            for (prov, mdl), price in self._prices.items():
                if mdl == eq_model and prov not in excluded:
                    cost = price.estimated_cost(self._input_tokens, self._output_tokens)
                    candidates.append((prov, mdl, cost))

        if not candidates:
            return None

        # Sort by cost ascending
        candidates.sort(key=lambda x: x[2])
        best_prov, best_model, best_cost = candidates[0]

        # Determine what the original cost would have been
        original_prov = self._primary_provider.get(model_name, "")
        original_price = self._prices.get((original_prov, model_name))
        original_cost = (
            original_price.estimated_cost(self._input_tokens, self._output_tokens)
            if original_price
            else best_cost
        )

        savings = original_cost - best_cost

        if best_prov == original_prov and best_model == model_name:
            reason = f"'{original_prov}::{model_name}' is already the cheapest equivalent"
        else:
            reason = (
                f"Arbitrage: '{best_prov}::{best_model}' is cheaper than "
                f"'{original_prov}::{model_name}' "
                f"(${best_cost:.5f} vs ${original_cost:.5f})"
            )

        decision = ArbitrageDecision(
            requested_model=model_name,
            chosen_provider=best_prov,
            chosen_model=best_model,
            original_provider=original_prov,
            original_cost=original_cost,
            arbitrage_cost=best_cost,
            savings_usd=savings,
            reason=reason,
        )

        # Record the decision
        self._decisions.append(decision)
        self._total_savings += savings
        self._total_original += original_cost
        self._total_arbitrage += best_cost

        self._log.info(
            "arbitrage_decision",
            model=model_name,
            chosen=f"{best_prov}::{best_model}",
            savings=f"${savings:.5f}",
        )
        return decision

    # ------------------------------------------------------------------
    # Savings tracking
    # ------------------------------------------------------------------

    def savings_report(self) -> SavingsReport:
        """Return a cumulative savings report.

        Returns:
            SavingsReport with total and per-decision stats.
        """
        savings_pct = (
            self._total_savings / self._total_original * 100
            if self._total_original > 0
            else 0.0
        )
        return SavingsReport(
            total_decisions=len(self._decisions),
            total_savings_usd=self._total_savings,
            total_original_cost_usd=self._total_original,
            total_arbitrage_cost_usd=self._total_arbitrage,
            savings_pct=savings_pct,
            decisions=list(self._decisions),
        )

    def reset_savings(self) -> None:
        """Reset the savings ledger (keeps model registry and prices)."""
        self._decisions.clear()
        self._total_savings = 0.0
        self._total_original = 0.0
        self._total_arbitrage = 0.0
