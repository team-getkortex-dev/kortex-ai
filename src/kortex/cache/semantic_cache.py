"""Semantic cache for Kortex coordination results.

Caches ``CoordinationResult`` dicts keyed by a hash of the task spec and
context messages. Identical tasks hitting the cache skip all routing and
provider calls, yielding dramatic speedups for repeated queries.
"""

from __future__ import annotations

import json

import xxhash
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kortex.cache.backends import MemoryCache, RedisCache
    from kortex.core.types import TaskSpec


class SemanticCache:
    """Cache layer for coordination results.

    Wraps a pluggable backend (``MemoryCache`` or ``RedisCache``) and
    provides a stable hashing scheme so that semantically identical tasks
    always map to the same key.

    Args:
        backend: A cache backend implementing ``get``/``set``/``clear``.
        ttl_seconds: Optional TTL applied when writing entries. ``None``
            means entries never expire (subject to LRU eviction for
            ``MemoryCache``).
    """

    def __init__(
        self,
        backend: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        self._backend = backend
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _make_key(
        self,
        task: TaskSpec,
        context: list[str],
        execute: bool,
    ) -> str:
        """Compute a deterministic xxh3-64 cache key.

        Uses xxhash XXH3-64 for ~100x faster hashing than SHA-256.
        Cryptographic strength is not required for cache keys.

        The key covers the full task spec, the context message list, and
        whether this is an execute (vs dry-run) call, so dry-run and live
        results are stored separately.

        Args:
            task: The task specification.
            context: Prior conversation messages or context strings.
            execute: Whether the call uses execute mode.

        Returns:
            A 16-character hex digest.
        """
        payload: dict[str, Any] = {
            "task": task.model_dump(mode="json"),
            "messages": context,
            "execute": execute,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return xxhash.xxh3_64(serialized.encode()).hexdigest()

    async def get(
        self,
        task: TaskSpec,
        context: list[str],
        execute: bool = False,
    ) -> dict[str, Any] | None:
        """Look up a cached result.

        Args:
            task: The task specification.
            context: Context messages.
            execute: Whether this is an execute call.

        Returns:
            The cached ``CoordinationResult`` dict, or ``None`` on miss.
        """
        key = self._make_key(task, context, execute)
        data = await self._backend.get(key)
        if data is None:
            self._misses += 1
            return None
        self._hits += 1
        return json.loads(data)

    async def set(
        self,
        task: TaskSpec,
        context: list[str],
        result_dict: dict[str, Any],
        execute: bool = False,
    ) -> None:
        """Store a result in the cache.

        Args:
            task: The task specification.
            context: Context messages.
            result_dict: The ``CoordinationResult`` dict to cache.
            execute: Whether this is an execute call.
        """
        key = self._make_key(task, context, execute)
        data = json.dumps(result_dict, default=str).encode()
        await self._backend.set(key, data, ttl_seconds=self._ttl)

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that returned a cached result."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def hits(self) -> int:
        """Total number of cache hits since creation or last clear."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total number of cache misses since creation or last clear."""
        return self._misses

    async def clear(self) -> None:
        """Clear all cached entries and reset hit/miss counters."""
        await self._backend.clear()
        self._hits = 0
        self._misses = 0
