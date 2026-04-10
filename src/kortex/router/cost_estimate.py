"""Cost estimation data model for Kortex batch routing.

``CostEstimate`` is a pure-data container returned by
``KortexRuntime.estimate_cost()``. It summarises predicted spend across a
batch of tasks before any provider calls are made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostEstimate:
    """Predicted cost for a batch of tasks.

    Args:
        total_usd: Sum of estimated costs across all successfully routed tasks.
        per_model: Mapping from model identity key to total estimated cost for
            tasks routed to that model.
        per_task: Per-task estimated cost in pipeline order (0.0 for tasks
            that could not be routed).
        task_count: Total number of tasks submitted for estimation.
        routing_failures: Number of tasks that could not be routed and
            therefore contribute $0 to the estimate.
    """

    total_usd: float
    per_model: dict[str, float] = field(default_factory=dict)
    per_task: list[float] = field(default_factory=list)
    task_count: int = 0
    routing_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "total_usd": self.total_usd,
            "per_model": dict(self.per_model),
            "per_task": list(self.per_task),
            "task_count": self.task_count,
            "routing_failures": self.routing_failures,
        }

    def summary(self) -> str:
        """Return a human-readable one-paragraph summary.

        Returns:
            A string suitable for CLI display or logging.
        """
        if self.task_count == 0:
            return "No tasks to estimate."

        avg = self.total_usd / max(1, self.task_count - self.routing_failures)
        top_models = ", ".join(
            f"{k}: ${v:.4f}"
            for k, v in sorted(self.per_model.items(), key=lambda x: -x[1])[:3]
        )
        parts = [
            f"Estimated cost for {self.task_count} task(s): "
            f"${self.total_usd:.4f} total (avg ${avg:.4f}/task)."
        ]
        if self.routing_failures:
            parts.append(f" {self.routing_failures} task(s) could not be routed.")
        if top_models:
            parts.append(f" Top models: {top_models}.")
        return "".join(parts)
