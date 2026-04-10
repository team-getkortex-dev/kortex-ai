"""Routing failure diagnostics for Kortex.

``RoutingDiagnostics`` analyses why a routing attempt failed and produces
a structured, human-readable explanation with concrete suggestions. It is
called automatically by the ``Router`` when no candidates survive
constraint filtering or heuristic selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kortex.security.redaction import scan_and_redact

if TYPE_CHECKING:
    from kortex.core.router import ProviderModel
    from kortex.core.types import TaskSpec


class RoutingDiagnostics:
    """Analyses routing failures and generates actionable diagnostics.

    All methods are stateless — create a new instance per failure or reuse
    a single shared instance freely.
    """

    def explain_failure(
        self,
        task: "TaskSpec",
        candidates: "list[ProviderModel]",
        constraint_failures: dict[str, list[str]] | None = None,
        heuristic_failures: list[str] | None = None,
    ) -> str:
        """Build a multi-line explanation of a routing failure.

        Args:
            task: The task that could not be routed.
            candidates: All registered candidate models (before filtering).
            constraint_failures: Mapping from model key to list of failure
                reasons produced by ``ConstraintSet.filter()``.
            heuristic_failures: Filter log entries from the heuristic
                strategy (cost ceiling, latency SLA, capability mismatches).

        Returns:
            A human-readable multi-line string explaining the failure and
            offering concrete suggestions.
        """
        lines: list[str] = []
        n_candidates = len(candidates)
        n_failed = len(constraint_failures) if constraint_failures else 0

        lines.append(f"Routing failed for task '{task.task_id}'.")

        task_desc = f"  Task: complexity={task.complexity_hint}"
        if task.cost_ceiling_usd is not None:
            task_desc += f", cost_ceiling=${task.cost_ceiling_usd:.4f}"
        if task.latency_sla_ms is not None:
            task_desc += f", latency_sla={task.latency_sla_ms}ms"
        if task.required_capabilities:
            task_desc += f", capabilities={task.required_capabilities}"
        lines.append(task_desc)

        lines.append(
            f"  Available models: {n_candidates}, "
            f"eliminated by constraints: {n_failed}"
        )

        if constraint_failures:
            shown = list(constraint_failures.items())[:3]
            for model_key, reasons in shown:
                lines.append(f"    {model_key}:")
                for r in reasons:
                    lines.append(f"      • {r}")
            if n_failed > 3:
                lines.append(f"    ... and {n_failed - 3} more")

        if heuristic_failures:
            for f in heuristic_failures[:3]:
                lines.append(f"  Filter: {f}")

        closest = self._find_closest(candidates, constraint_failures or {})
        if closest:
            lines.append(f"  Closest match: {closest}")

        suggestions = self._generate_suggestions(
            task, candidates, constraint_failures or {}
        )
        if suggestions:
            lines.append("  Suggestions:")
            for s in suggestions:
                lines.append(f"    → {s}")

        return scan_and_redact("\n".join(lines))

    # -- helpers --------------------------------------------------------------

    def _find_closest(
        self,
        candidates: "list[ProviderModel]",
        constraint_failures: dict[str, list[str]],
    ) -> str | None:
        """Return a description of the model that came closest to passing."""
        if not candidates:
            return None
        scored = sorted(
            (len(constraint_failures.get(m.identity.key, [])), m)
            for m in candidates
        )
        count, model = scored[0]
        if count > 0:
            return f"{model.provider}::{model.model} (failed {count} constraint(s))"
        return None

    def _generate_suggestions(
        self,
        task: "TaskSpec",
        candidates: "list[ProviderModel]",
        constraint_failures: dict[str, list[str]],
    ) -> list[str]:
        """Generate concrete suggestions for resolving the failure."""
        suggestions: list[str] = []

        if not candidates:
            suggestions.append(
                "Register models using Router.register_model() before routing."
            )
            return suggestions

        if task.required_capabilities:
            all_caps = set(cap for m in candidates for cap in m.capabilities)
            missing = set(task.required_capabilities) - all_caps
            if missing:
                suggestions.append(
                    f"No registered models support: {sorted(missing)}. "
                    "Register models with these capabilities."
                )

        if task.cost_ceiling_usd is not None:
            cheapest = min(candidates, key=lambda m: m.estimated_cost(), default=None)
            if cheapest is not None and cheapest.estimated_cost() > task.cost_ceiling_usd:
                suggestions.append(
                    f"Cheapest model ({cheapest.provider}::{cheapest.model}) costs "
                    f"${cheapest.estimated_cost():.4f}, exceeding ceiling "
                    f"${task.cost_ceiling_usd:.4f}. Raise cost_ceiling_usd or "
                    "register cheaper models."
                )

        if task.latency_sla_ms is not None:
            fastest = min(candidates, key=lambda m: m.avg_latency_ms, default=None)
            if fastest is not None and fastest.avg_latency_ms > task.latency_sla_ms:
                suggestions.append(
                    f"Fastest model ({fastest.provider}::{fastest.model}) "
                    f"avg_latency={fastest.avg_latency_ms}ms exceeds SLA "
                    f"{task.latency_sla_ms}ms. Raise latency_sla_ms or add "
                    "faster models."
                )

        if constraint_failures:
            suggestions.append(
                "Relax constraints or register models that satisfy all constraints."
            )

        return suggestions
