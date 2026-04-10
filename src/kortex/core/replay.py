"""Replay engine for Kortex traces.

Answers: "What happened, why, and what would change under a different policy?"

The ``ReplayEngine`` takes a recorded ``TaskTrace`` and re-evaluates every
routing decision against current models — optionally under a different
``RoutingPolicy``. This enables policy comparison, what-if analysis, and
cost optimisation without re-executing any LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from kortex.core.policy import RoutingPolicy
from kortex.core.router import Router
from kortex.core.trace import TaskTrace, TraceStep
from kortex.core.types import TaskSpec

logger = structlog.get_logger(component="replay")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ReplayedStep:
    """Comparison of one step under original vs replayed routing.

    Args:
        step_index: Position in the pipeline (0-based).
        agent_id: The agent for this step.
        original_model: Model chosen in the original trace.
        original_provider: Provider chosen in the original trace.
        replayed_model: Model chosen under the replay policy.
        replayed_provider: Provider chosen under the replay policy.
        model_changed: Whether the model selection changed.
        original_estimated_cost: Estimated cost from the original trace.
        replayed_estimated_cost: Estimated cost under the replay policy.
        cost_delta: Positive = more expensive, negative = savings.
        explanation: Human-readable explanation from PolicyEvaluation.
    """

    step_index: int
    agent_id: str
    original_model: str
    original_provider: str
    replayed_model: str
    replayed_provider: str
    model_changed: bool
    original_estimated_cost: float
    replayed_estimated_cost: float
    cost_delta: float
    explanation: str


@dataclass
class ReplayResult:
    """Complete result of replaying a trace under a (potentially different) policy.

    Args:
        original_trace: The trace that was replayed.
        replayed_steps: Per-step comparison.
        policy_used: Serialized policy used for the replay.
        summary: Human-readable comparison summary.
        from_step: Step index where replay started (0 = full replay).
    """

    original_trace: TaskTrace
    replayed_steps: list[ReplayedStep]
    policy_used: dict[str, Any]
    summary: str
    from_step: int = 0

    def diff(self, other: ReplayResult) -> dict[str, Any]:
        """Compare this replay result to another.

        Returns a step-by-step diff showing which models changed and the
        overall cost delta between the two replays.

        Args:
            other: The other ReplayResult to compare against.

        Returns:
            Dict with ``step_diffs``, ``total_cost_self``, ``total_cost_other``,
            ``cost_delta``, and ``changed_steps``.
        """
        self_by_idx = {s.step_index: s for s in self.replayed_steps}
        other_by_idx = {s.step_index: s for s in other.replayed_steps}
        all_indices = sorted(set(self_by_idx) | set(other_by_idx))

        step_diffs: list[dict[str, Any]] = []
        for idx in all_indices:
            s = self_by_idx.get(idx)
            o = other_by_idx.get(idx)
            model_differs = (
                (s.replayed_model if s else None)
                != (o.replayed_model if o else None)
                or (s.replayed_provider if s else None)
                != (o.replayed_provider if o else None)
            )
            step_diffs.append({
                "step_index": idx,
                "self_model": f"{s.replayed_provider}::{s.replayed_model}" if s else None,
                "other_model": f"{o.replayed_provider}::{o.replayed_model}" if o else None,
                "model_differs": model_differs,
                "self_cost": s.replayed_estimated_cost if s else 0.0,
                "other_cost": o.replayed_estimated_cost if o else 0.0,
                "cost_delta": (o.replayed_estimated_cost if o else 0.0)
                              - (s.replayed_estimated_cost if s else 0.0),
            })

        total_self = sum(s.replayed_estimated_cost for s in self.replayed_steps)
        total_other = sum(s.replayed_estimated_cost for s in other.replayed_steps)
        changed = sum(1 for d in step_diffs if d["model_differs"])

        return {
            "step_diffs": step_diffs,
            "total_cost_self": round(total_self, 6),
            "total_cost_other": round(total_other, 6),
            "cost_delta": round(total_other - total_self, 6),
            "changed_steps": changed,
            "policy_self": self.policy_used.get("name", "?"),
            "policy_other": other.policy_used.get("name", "?"),
        }


# ---------------------------------------------------------------------------
# ReplayEngine
# ---------------------------------------------------------------------------


class ReplayEngine:
    """Replays recorded traces against current models and policies.

    Args:
        router: A Router with registered models. The engine uses these
            models to re-evaluate routing decisions.
    """

    def __init__(self, router: Router) -> None:
        self._router = router
        self._log = structlog.get_logger(component="replay_engine")

    async def replay(
        self,
        trace: TaskTrace,
        policy: RoutingPolicy | None = None,
    ) -> ReplayResult:
        """Replay a trace, optionally under a different policy.

        For each step in the trace:
        1. Reconstruct the TaskSpec from the trace step data.
        2. If a policy is provided, temporarily set it on the router.
           Otherwise, restore the original policy from the trace snapshot.
        3. Route the reconstructed task.
        4. Compare the new decision to the original.

        Args:
            trace: The recorded trace to replay.
            policy: If provided, use this policy instead of the original.
                If None, reconstruct the original policy from the trace.

        Returns:
            A ReplayResult with per-step comparisons and summary.
        """
        # Determine which policy to use
        if policy is not None:
            replay_policy = policy
        elif trace.policy_snapshot:
            replay_policy = RoutingPolicy.from_dict(trace.policy_snapshot)
        else:
            replay_policy = None

        # Save and restore the router's policy
        original_router_policy = self._router.get_policy()
        try:
            if replay_policy is not None:
                self._router.set_policy(replay_policy)
            else:
                # Clear any existing policy to use heuristic
                self._router._policy = None

            replayed_steps: list[ReplayedStep] = []

            for step in trace.steps:
                replayed = await self._replay_step(step)
                replayed_steps.append(replayed)

        finally:
            # Restore original policy
            if original_router_policy is not None:
                self._router.set_policy(original_router_policy)
            else:
                self._router._policy = None

        policy_dict = replay_policy.to_dict() if replay_policy else {}
        summary = self._build_summary(trace, replayed_steps, policy_dict)

        return ReplayResult(
            original_trace=trace,
            replayed_steps=replayed_steps,
            policy_used=policy_dict,
            summary=summary,
        )

    async def policy_diff(
        self, trace: TaskTrace, new_policy: RoutingPolicy
    ) -> ReplayResult:
        """Replay under a new policy and highlight differences.

        Convenience method — calls ``replay()`` with the given policy.

        Args:
            trace: The recorded trace.
            new_policy: The policy to compare against.

        Returns:
            ReplayResult with summary highlighting changed steps and cost delta.
        """
        return await self.replay(trace, policy=new_policy)

    async def what_if(
        self, trace: TaskTrace, modifications: dict[str, Any]
    ) -> ReplayResult:
        """Apply modifications to the original policy and replay.

        Supports modifications like ``{"denied_providers": ["openai"]}`` to
        see what would happen under altered constraints.

        Args:
            trace: The recorded trace.
            modifications: Dict of fields to overlay on the original policy.
                Supports top-level policy fields and nested constraint/objective
                fields prefixed with their section name
                (e.g. ``{"denied_providers": ["openai"]}`` modifies constraints).

        Returns:
            ReplayResult showing what would change.
        """
        # Start from the original policy
        if trace.policy_snapshot:
            base_dict = dict(trace.policy_snapshot)
        else:
            base_dict = RoutingPolicy().to_dict()

        # Apply constraint-level modifications
        constraint_keys = {
            "max_cost_usd", "max_latency_ms", "required_capabilities",
            "allowed_providers", "denied_providers", "allowed_models",
            "denied_models",
        }
        objective_keys = {"minimize", "prefer_tier", "prefer_provider"}
        fallback_keys = {"strategy", "explicit_model_identity"}

        constraints = dict(base_dict.get("constraints", {}))
        objective = dict(base_dict.get("objective", {}))
        fallback = dict(base_dict.get("fallback", {}))

        for key, value in modifications.items():
            if key in constraint_keys:
                constraints[key] = value
            elif key in objective_keys:
                objective[key] = value
            elif key in fallback_keys:
                fallback[key] = value
            else:
                base_dict[key] = value

        base_dict["constraints"] = constraints
        base_dict["objective"] = objective
        base_dict["fallback"] = fallback

        modified_policy = RoutingPolicy.from_dict(base_dict)
        return await self.replay(trace, policy=modified_policy)

    # -- internal ----------------------------------------------------------

    async def _replay_step(self, step: TraceStep) -> ReplayedStep:
        """Replay a single trace step and compare to original."""
        # Reconstruct the task from trace data
        original_decision = step.routing_decision
        original_model = original_decision.get("chosen_model", "")
        original_provider = original_decision.get("chosen_provider", "")
        original_cost = original_decision.get("estimated_cost_usd", 0.0)

        task_content = step.input_payload.get(
            "content",
            step.input_payload.get("task_id", "replayed task"),
        )

        task = TaskSpec(
            content=str(task_content),
            complexity_hint=original_decision.get("complexity_hint", "moderate"),
            required_capabilities=original_decision.get(
                "required_capabilities", []
            ),
            cost_ceiling_usd=original_decision.get("cost_ceiling_usd"),
            latency_sla_ms=original_decision.get("latency_sla_ms"),
        )

        # Route under the current router (which has the replay policy set)
        decision = await self._router.route(task)

        model_changed = (
            decision.chosen_model != original_model
            or decision.chosen_provider != original_provider
        )
        cost_delta = decision.estimated_cost_usd - original_cost

        return ReplayedStep(
            step_index=step.step_index,
            agent_id=step.agent_id,
            original_model=original_model,
            original_provider=original_provider,
            replayed_model=decision.chosen_model,
            replayed_provider=decision.chosen_provider,
            model_changed=model_changed,
            original_estimated_cost=original_cost,
            replayed_estimated_cost=decision.estimated_cost_usd,
            cost_delta=cost_delta,
            explanation=decision.reasoning,
        )

    @staticmethod
    def _build_summary(
        trace: TaskTrace,
        steps: list[ReplayedStep],
        policy_dict: dict[str, Any],
    ) -> str:
        """Build a human-readable summary of the replay."""
        total_steps = len(steps)
        changed = sum(1 for s in steps if s.model_changed)
        total_cost_delta = sum(s.cost_delta for s in steps)
        original_total = sum(s.original_estimated_cost for s in steps)

        parts: list[str] = []

        policy_name = policy_dict.get("name", "heuristic")
        parts.append(f"Policy '{policy_name}':")

        if changed == 0:
            parts.append(
                f" {total_steps} step(s), no model changes."
            )
        else:
            parts.append(
                f" {changed} of {total_steps} step(s) would change model selection."
            )

        if total_cost_delta != 0 and original_total > 0:
            if total_cost_delta < 0:
                pct = abs(total_cost_delta) / original_total * 100
                parts.append(
                    f" Estimated savings: ${abs(total_cost_delta):.6f}"
                    f" ({pct:.0f}% reduction)."
                )
            else:
                pct = total_cost_delta / original_total * 100
                parts.append(
                    f" Estimated increase: ${total_cost_delta:.6f}"
                    f" ({pct:.0f}% more)."
                )
        elif total_cost_delta == 0:
            parts.append(" No cost change.")

        return "".join(parts)

    async def replay_from_step(
        self,
        trace: TaskTrace,
        from_step: int,
        policy: RoutingPolicy | None = None,
    ) -> ReplayResult:
        """Replay only the steps from ``from_step`` onward.

        The steps before ``from_step`` are copied verbatim from the original
        trace. This is the core of the time-travel debugging workflow: you
        can "rewind" to a specific step and explore what would happen if a
        different model had been chosen there.

        Args:
            trace: The recorded trace.
            from_step: Step index to start replaying from (0-based).
            policy: Optional policy override for the replayed suffix.

        Returns:
            ReplayResult containing only the replayed suffix steps.

        Raises:
            IndexError: If ``from_step`` is out of range.
        """
        if from_step < 0 or from_step >= len(trace.steps):
            raise IndexError(
                f"from_step {from_step} out of range "
                f"(trace has {len(trace.steps)} steps)"
            )

        # Determine replay policy
        if policy is not None:
            replay_policy = policy
        elif trace.policy_snapshot:
            replay_policy = RoutingPolicy.from_dict(trace.policy_snapshot)
        else:
            replay_policy = None

        original_router_policy = self._router.get_policy()
        try:
            if replay_policy is not None:
                self._router.set_policy(replay_policy)
            else:
                self._router._policy = None

            replayed_steps: list[ReplayedStep] = []
            # Replay only the suffix
            for step in trace.steps[from_step:]:
                replayed = await self._replay_step(step)
                replayed_steps.append(replayed)

        finally:
            if original_router_policy is not None:
                self._router.set_policy(original_router_policy)
            else:
                self._router._policy = None

        policy_dict = replay_policy.to_dict() if replay_policy else {}
        summary = self._build_summary(trace, replayed_steps, policy_dict)
        suffix_summary = (
            f"(from step {from_step}/{len(trace.steps) - 1}) " + summary
        )

        return ReplayResult(
            original_trace=trace,
            replayed_steps=replayed_steps,
            policy_used=policy_dict,
            summary=suffix_summary,
            from_step=from_step,
        )
