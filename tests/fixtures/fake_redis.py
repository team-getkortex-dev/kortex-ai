"""Fake async Redis client for testing RedisStateStore without a real Redis server.

Implements the subset of redis.asyncio methods used by RedisStateStore:
get, set, delete, keys, zadd, zrangebyscore, zrem, expire, ping, scan_iter, flushdb, aclose.
"""

from __future__ import annotations

import fnmatch
from typing import Any, AsyncIterator


class FakeRedis:
    """In-memory fake Redis that mirrors the redis.asyncio interface."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._sorted_sets: dict[str, dict[str, float]] = {}
        self._ttls: dict[str, int] = {}

    async def ping(self) -> bool:
        return True

    async def set(self, key: str, value: str) -> None:
        self._data[key] = value

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
            if key in self._sorted_sets:
                del self._sorted_sets[key]
                count += 1
        return count

    async def expire(self, key: str, seconds: int) -> bool:
        if key in self._data:
            self._ttls[key] = seconds
            return True
        return False

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        if key not in self._sorted_sets:
            self._sorted_sets[key] = {}
        added = 0
        for member, score in mapping.items():
            if member not in self._sorted_sets[key]:
                added += 1
            self._sorted_sets[key][member] = score
        return added

    async def zrangebyscore(
        self, key: str, min_score: str | float, max_score: str | float
    ) -> list[str]:
        if key not in self._sorted_sets:
            return []
        items = self._sorted_sets[key]
        # Return all members sorted by score (we treat "-inf"/"+inf" as bounds)
        sorted_members = sorted(items.items(), key=lambda x: x[1])
        return [member for member, _score in sorted_members]

    async def zrem(self, key: str, *members: str) -> int:
        if key not in self._sorted_sets:
            return 0
        count = 0
        for member in members:
            if member in self._sorted_sets[key]:
                del self._sorted_sets[key][member]
                count += 1
        return count

    async def scan_iter(self, match: str = "*") -> AsyncIterator[str]:
        """Async iterator over keys matching a glob pattern."""
        for key in list(self._data.keys()):
            if fnmatch.fnmatch(key, match):
                yield key

    async def flushdb(self) -> None:
        self._data.clear()
        self._sorted_sets.clear()
        self._ttls.clear()

    async def aclose(self) -> None:
        pass

    # -- Test helpers (not part of Redis API) --------------------------------

    def get_ttl(self, key: str) -> int | None:
        """Return the TTL set for a key, or None if no TTL."""
        return self._ttls.get(key)

    def key_count(self, prefix: str = "") -> int:
        """Count keys matching a prefix."""
        return sum(1 for k in self._data if k.startswith(prefix))
