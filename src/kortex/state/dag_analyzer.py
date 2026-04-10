"""DAG dependency analyzer for HandoffContext objects.

Identifies which handoffs in a batch are independent (no shared ancestry
within the batch) so they can be persisted concurrently with
``asyncio.gather``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kortex.core.types import HandoffContext


class DAGAnalyzer:
    """Analyzes dependency relationships among a set of handoffs.

    Each handoff may declare a ``parent_checkpoint_id``.  If that parent is
    also present in the batch, the handoff *depends* on it and must be
    persisted after the parent.  Handoffs whose parents are outside the batch
    (or have no parent at all) are treated as roots and can run immediately.

    The analyzer performs a topological sort and returns execution *groups*:
    handoffs within the same group are independent and can be saved in
    parallel; groups must be processed sequentially relative to each other.
    """

    def analyze_dependencies(
        self, handoffs: list[HandoffContext]
    ) -> dict[str, set[str]]:
        """Build a dependency map for the given handoffs.

        Args:
            handoffs: The handoffs to analyze.

        Returns:
            A dict mapping each ``checkpoint_id`` to the set of
            ``checkpoint_id`` values it depends on (i.e., must be saved after).
            Only intra-batch dependencies are recorded.
        """
        checkpoint_ids: set[str] = {h.checkpoint_id for h in handoffs}
        deps: dict[str, set[str]] = {}
        for h in handoffs:
            dep: set[str] = set()
            if h.parent_checkpoint_id and h.parent_checkpoint_id in checkpoint_ids:
                dep.add(h.parent_checkpoint_id)
            deps[h.checkpoint_id] = dep
        return deps

    def get_execution_groups(
        self, handoffs: list[HandoffContext]
    ) -> list[list[HandoffContext]]:
        """Topologically sort handoffs into parallel execution groups.

        Handoffs within the same group have no dependencies on each other and
        can be persisted concurrently.  Groups are ordered so that every
        dependency is in an earlier group.

        Args:
            handoffs: The handoffs to group.

        Returns:
            A list of groups; each group is a list of independent
            ``HandoffContext`` objects that can be saved in parallel.
        """
        if not handoffs:
            return []

        deps = self.analyze_dependencies(handoffs)
        id_to_handoff: dict[str, HandoffContext] = {h.checkpoint_id: h for h in handoffs}
        completed: set[str] = set()
        remaining: set[str] = set(deps.keys())
        groups: list[list[HandoffContext]] = []

        while remaining:
            # All nodes whose dependencies have already been satisfied
            ready = [
                cid
                for cid in remaining
                if deps[cid].issubset(completed)
            ]
            if not ready:
                # Cycle or unresolvable — flush the rest sequentially to avoid
                # an infinite loop.  This should never happen with valid input.
                groups.append([id_to_handoff[cid] for cid in sorted(remaining)])
                break

            group = [id_to_handoff[cid] for cid in ready]
            groups.append(group)
            completed.update(ready)
            remaining -= set(ready)

        return groups
