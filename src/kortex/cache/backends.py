"""Cache backend implementations for Kortex semantic cache.

Provides two backends:
- ``MemoryCache``: In-process LRU cache backed by ``OrderedDict``.
- ``RedisCache``: Distributed cache backed by Redis with optional TTL.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class MemoryCache:
    """In-memory LRU cache.

    Uses an ``OrderedDict`` to maintain access order; the least-recently-used
    entry is evicted when the cache reaches ``max_size``.

    Args:
        max_size: Maximum number of entries before eviction begins.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, bytes] = OrderedDict()

    async def get(self, key: str) -> bytes | None:
        """Retrieve a value, promoting it to most-recently-used.

        Args:
            key: Cache key.

        Returns:
            The cached bytes, or None if the key is not present.
        """
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        """Store a value, evicting the LRU entry if over capacity.

        Args:
            key: Cache key.
            value: Bytes to store.
            ttl_seconds: Ignored for in-memory cache (no expiry support).
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    async def delete(self, key: str) -> None:
        """Remove a key from the cache.

        Args:
            key: Cache key to remove.
        """
        self._cache.pop(key, None)

    async def clear(self) -> None:
        """Remove all entries from the cache."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Return the current number of entries."""
        return len(self._cache)


class RedisCache:
    """Distributed cache backed by Redis with optional TTL.

    Lazy-initializes the Redis client on first use so that tests and
    environments without Redis installed can still import this module.

    Args:
        redis_url: Redis connection URL.
        key_prefix: Namespace prefix prepended to all keys.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "kortex:cache:",
    ) -> None:
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as redis  # type: ignore[import-untyped]

            self._client = redis.from_url(self._redis_url)
        return self._client

    async def get(self, key: str) -> bytes | None:
        """Retrieve a value by key.

        Args:
            key: Cache key (without prefix).

        Returns:
            The cached bytes, or None if not found or expired.
        """
        client = await self._get_client()
        return await client.get(f"{self._key_prefix}{key}")

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        """Store a value with optional TTL.

        Args:
            key: Cache key (without prefix).
            value: Bytes to store.
            ttl_seconds: Seconds until expiry, or None for no expiry.
        """
        client = await self._get_client()
        await client.set(f"{self._key_prefix}{key}", value, ex=ttl_seconds)

    async def delete(self, key: str) -> None:
        """Remove a key.

        Args:
            key: Cache key (without prefix).
        """
        client = await self._get_client()
        await client.delete(f"{self._key_prefix}{key}")

    async def clear(self) -> None:
        """Remove all keys matching this instance's prefix."""
        client = await self._get_client()
        async for key in client.scan_iter(f"{self._key_prefix}*"):
            await client.delete(key)

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
