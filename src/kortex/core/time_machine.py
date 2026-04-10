"""Time-travel debugging for Kortex traces.

The ``TimeMachine`` lets you "rewind" a ``TaskTrace`` to any step and
explore what would happen if a different routing decision had been made
at that point. Key concepts:

- ``ExecutionSnapshot`` — a frozen copy of the system state at one step
- ``TimeMachine.snapshot(step_index)`` — create a snapshot at a step
- ``TimeMachine.restore(snapshot)`` — build a task context to re-route from
- ``ReplayEngine.replay_from_step()`` — replay only the suffix of a trace
  starting from a specific step index

Combined with ``ReplayResult.diff()`` you can compare two replay runs
side-by-side to understand exactly what changed.

Example::

    from kortex.core.time_machine import TimeMachine
    from kortex.core.replay import ReplayEngine

    tm = TimeMachine(trace)
    snapshot = tm.snapshot(step_index=1)  # freeze state after step 1
    context = tm.restore(snapshot)        # get task + policy to resume

    engine = ReplayEngine(router)
    result = await engine.replay_from_step(trace, from_step=1, policy=new_policy)
    print(result.summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kortex.core.trace import TaskTrace, TraceStep


# ---------------------------------------------------------------------------
# ExecutionSnapshot
# ---------------------------------------------------------------------------


@dataclass
class ExecutionSnapshot:
    """Frozen state of the system at one step in a coordination trace.

    Can be used to "restore" execution context and re-route from that
    exact point under a different policy.

    Args:
        step_index: The step this snapshot was taken at (0-based).
        agent_id: The agent running at this step.
        task_content: The task content visible to this agent.
        complexity_hint: Complexity hint at this step.
        required_capabilities: Capabilities required at this step.
        policy_snapshot: The routing policy active at this step.
        routing_decision: The actual routing decision made.
        input_payload: The boundary-crossing payload at this step.
        cumulative_cost_usd: Estimated cost accumulated before this step.
        cumulative_latency_ms: Estimated latency accumulated before this step.
        trace_id: The trace this snapshot belongs to.
        task_id: The task this snapshot belongs to.
    """

    step_index: int
    agent_id: str
    task_content: str
    complexity_hint: str
    required_capabilities: list[str]
    policy_snapshot: dict[str, Any]
    routing_decision: dict[str, Any]
    input_payload: dict[str, Any]
    cumulative_cost_usd: float
    cumulative_latency_ms: float
    trace_id: str
    task_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "agent_id": self.agent_id,
            "task_content": self.task_content,
            "complexity_hint": self.complexity_hint,
            "required_capabilities": self.required_capabilities,
            "policy_snapshot": self.policy_snapshot,
            "routing_decision": self.routing_decision,
            "input_payload": self.input_payload,
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "cumulative_latency_ms": self.cumulative_latency_ms,
            "trace_id": self.trace_id,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionSnapshot:
        return cls(
            step_index=data["step_index"],
            agent_id=data["agent_id"],
            task_content=data.get("task_content", ""),
            complexity_hint=data.get("complexity_hint", "moderate"),
            required_capabilities=data.get("required_capabilities", []),
            policy_snapshot=data.get("policy_snapshot", {}),
            routing_decision=data.get("routing_decision", {}),
            input_payload=data.get("input_payload", {}),
            cumulative_cost_usd=data.get("cumulative_cost_usd", 0.0),
            cumulative_latency_ms=data.get("cumulative_latency_ms", 0.0),
            trace_id=data.get("trace_id", ""),
            task_id=data.get("task_id", ""),
        )


# ---------------------------------------------------------------------------
# TimeMachine
# ---------------------------------------------------------------------------


class TimeMachine:
    """Rewind and inspect execution at any step in a trace.

    Args:
        trace: The TaskTrace to navigate.
    """

    def __init__(self, trace: TaskTrace) -> None:
        self._trace = trace

    @property
    def trace(self) -> TaskTrace:
        """The underlying trace."""
        return self._trace

    @property
    def num_steps(self) -> int:
        """Total number of steps in the trace."""
        return len(self._trace.steps)

    def snapshot(self, step_index: int) -> ExecutionSnapshot:
        """Create an ExecutionSnapshot at the given step.

        Args:
            step_index: Which step to snapshot (0-based).

        Returns:
            ExecutionSnapshot capturing state at that step.

        Raises:
            IndexError: If ``step_index`` is out of range.
        """
        if step_index < 0 or step_index >= len(self._trace.steps):
            raise IndexError(
                f"step_index {step_index} is out of range "
                f"(trace has {len(self._trace.steps)} steps)"
            )

        step = self._trace.steps[step_index]

        # Cumulative cost/latency up to (but not including) this step
        prev_steps = self._trace.steps[:step_index]
        cumulative_cost = sum(
            s.routing_decision.get("estimated_cost_usd", 0.0)
            for s in prev_steps
        )
        cumulative_latency = sum(
            s.routing_decision.get("estimated_latency_ms", 0.0)
            for s in prev_steps
        )

        task_content = step.input_payload.get(
            "content", self._trace.task_content or ""
        )

        return ExecutionSnapshot(
            step_index=step_index,
            agent_id=step.agent_id,
            task_content=task_content,
            complexity_hint=self._trace.task_complexity or "moderate",
            required_capabilities=step.input_payload.get("required_capabilities", []),
            policy_snapshot=step.policy_snapshot,
            routing_decision=step.routing_decision,
            input_payload=dict(step.input_payload),
            cumulative_cost_usd=cumulative_cost,
            cumulative_latency_ms=cumulative_latency,
            trace_id=self._trace.trace_id,
            task_id=self._trace.task_id,
        )

    def restore(self, snapshot: ExecutionSnapshot) -> dict[str, Any]:
        """Build an execution context dict suitable for re-routing.

        The returned dict contains everything needed to resume routing from
        ``snapshot.step_index`` onward:
        - ``task_content``, ``complexity_hint``, ``required_capabilities``
        - ``remaining_agents``: agent IDs from snapshot.step_index onward
        - ``policy_snapshot``: the policy to restore
        - ``from_step``: the step index to resume from

        Args:
            snapshot: The snapshot to restore.

        Returns:
            Dict with execution context for resumption.
        """
        remaining_agents = [
            s.agent_id for s in self._trace.steps[snapshot.step_index:]
        ]
        return {
            "task_content": snapshot.task_content,
            "complexity_hint": snapshot.complexity_hint,
            "required_capabilities": snapshot.required_capabilities,
            "policy_snapshot": snapshot.policy_snapshot,
            "from_step": snapshot.step_index,
            "remaining_agents": remaining_agents,
            "trace_id": snapshot.trace_id,
            "task_id": snapshot.task_id,
        }

    def snapshots(self) -> list[ExecutionSnapshot]:
        """Return snapshots for every step in the trace.

        Returns:
            List of ExecutionSnapshot, one per step.
        """
        return [self.snapshot(i) for i in range(len(self._trace.steps))]

    def step_summary(self, step_index: int) -> str:
        """Return a concise text summary of one step.

        Args:
            step_index: Step to summarise.

        Returns:
            Human-readable line.
        """
        snap = self.snapshot(step_index)
        rd = snap.routing_decision
        chosen = f"{rd.get('chosen_provider', '?')}::{rd.get('chosen_model', '?')}"
        cost = rd.get("estimated_cost_usd", 0.0)
        latency = rd.get("estimated_latency_ms", 0.0)
        return (
            f"[{step_index}] {snap.agent_id} → {chosen} "
            f"~${cost:.4f} ~{latency:.0f}ms "
            f"(cum. ${snap.cumulative_cost_usd:.4f})"
        )

    def full_summary(self) -> str:
        """Return a multi-line summary of the entire trace.

        Returns:
            Human-readable trace walkthrough.
        """
        lines = [
            f"Trace: {self._trace.trace_id}",
            f"Task:  {self._trace.task_id}",
            f"Steps: {len(self._trace.steps)}",
            f"Total cost: ${self._trace.total_estimated_cost_usd:.4f}",
            f"Duration:   {self._trace.total_duration_ms:.0f}ms",
            f"Success:    {self._trace.success}",
            "",
        ]
        for i in range(len(self._trace.steps)):
            lines.append("  " + self.step_summary(i))
        return "\n".join(lines)
