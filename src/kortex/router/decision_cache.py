"""LRU cache for routing decisions.

Caches ``RoutingDecision`` objects keyed by a hash of the task spec and
active routing policy.  Identical (task, policy) combinations skip the full
routing computation, saving ~5ms per repeated call.

Uses xxhash XXH3-64 for fast, non-cryptographic key generation and an
``OrderedDict`` for O(1) LRU eviction.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import xxhash

if TYPE_CHECKING:
    from kortex.core.policy import RoutingPolicy
    from kortex.core.types import RoutingDecision, TaskSpec


class RoutingDecisionCache:
    """LRU cache for routing decisions.

    Args:
        max_size: Maximum number of decisions to hold before evicting the
            least-recently-used entry.  Default 10 000.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._cache: OrderedDict[str, RoutingDecision] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def _make_key(
        self,
        task: TaskSpec,
        policy: RoutingPolicy | None,
    ) -> str:
        """Compute a deterministic xxh3-64 key for (task, policy).

        Args:
            task: The task specification.
            policy: The active routing policy, or ``None``.

        Returns:
            A 16-character hex digest.
        """
        policy_data: dict[str, Any] | None = None
        if policy is not None:
            if hasattr(policy, "model_dump"):
                policy_data = policy.model_dump(mode="json")  # type: ignore[union-attr]
            else:
                import dataclasses

                if dataclasses.is_dataclass(policy):
                    policy_data = dataclasses.asdict(policy)  # type: ignore[arg-type]
                else:
                    policy_data = {"id": id(policy), "name": getattr(policy, "name", str(policy))}

        payload: dict[str, Any] = {
            "task": task.model_dump(mode="json"),
            "policy": policy_data,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return xxhash.xxh3_64(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    def get(
        self,
        task: TaskSpec,
        policy: RoutingPolicy | None,
    ) -> RoutingDecision | None:
        """Return a cached decision, or ``None`` on miss.

        Marks the entry as most-recently-used on hit.

        Args:
            task: The task specification.
            policy: The active routing policy.

        Returns:
            The cached ``RoutingDecision``, or ``None``.
        """
        key = self._make_key(task, policy)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(
        self,
        task: TaskSpec,
        policy: RoutingPolicy | None,
        decision: RoutingDecision,
    ) -> None:
        """Store a routing decision, evicting LRU entry when full.

        Args:
            task: The task specification.
            policy: The active routing policy.
            decision: The ``RoutingDecision`` to cache.
        """
        key = self._make_key(task, policy)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = decision
        else:
            self._cache[key] = decision
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached decisions and reset hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        """Total cache hits since creation or last ``clear()``."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses since creation or last ``clear()``."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that returned a cached decision."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)
