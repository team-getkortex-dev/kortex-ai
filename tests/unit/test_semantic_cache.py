"""Tests for semantic cache backends and SemanticCache."""

from __future__ import annotations

import pytest

from kortex.cache.backends import MemoryCache
from kortex.cache.semantic_cache import SemanticCache
from kortex.core.types import TaskSpec


# ---------------------------------------------------------------------------
# MemoryCache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_cache_get_miss() -> None:
    cache = MemoryCache()
    assert await cache.get("missing") is None


@pytest.mark.asyncio
async def test_memory_cache_set_and_get() -> None:
    cache = MemoryCache()
    await cache.set("k", b"hello")
    assert await cache.get("k") == b"hello"


@pytest.mark.asyncio
async def test_memory_cache_lru_eviction() -> None:
    cache = MemoryCache(max_size=3)
    for i in range(3):
        await cache.set(str(i), f"v{i}".encode())

    # Access "0" so it becomes most-recently-used
    await cache.get("0")

    # Add a 4th entry — "1" should be evicted (LRU)
    await cache.set("3", b"v3")

    assert await cache.get("1") is None
    assert await cache.get("0") == b"v0"
    assert await cache.get("2") == b"v2"
    assert await cache.get("3") == b"v3"


@pytest.mark.asyncio
async def test_memory_cache_delete() -> None:
    cache = MemoryCache()
    await cache.set("k", b"v")
    await cache.delete("k")
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_memory_cache_clear() -> None:
    cache = MemoryCache()
    await cache.set("a", b"1")
    await cache.set("b", b"2")
    await cache.clear()
    assert cache.size == 0


@pytest.mark.asyncio
async def test_memory_cache_size() -> None:
    cache = MemoryCache()
    assert cache.size == 0
    await cache.set("k", b"v")
    assert cache.size == 1


@pytest.mark.asyncio
async def test_memory_cache_overwrite_promotes() -> None:
    """Overwriting an existing key should promote it to MRU."""
    cache = MemoryCache(max_size=2)
    await cache.set("a", b"1")
    await cache.set("b", b"2")
    # Overwrite "a" — it's now MRU; "b" is LRU
    await cache.set("a", b"new")
    # Add "c" — "b" should be evicted
    await cache.set("c", b"3")
    assert await cache.get("b") is None
    assert await cache.get("a") == b"new"


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


def _make_task(**kwargs: object) -> TaskSpec:
    return TaskSpec(content="Summarise this document", **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_semantic_cache_miss_on_empty() -> None:
    cache = SemanticCache(MemoryCache())
    task = _make_task()
    assert await cache.get(task, []) is None
    assert cache.misses == 1
    assert cache.hits == 0


@pytest.mark.asyncio
async def test_semantic_cache_hit_after_set() -> None:
    cache = SemanticCache(MemoryCache())
    task = _make_task()
    result = {"task_id": task.task_id, "success": True}
    await cache.set(task, [], result)
    retrieved = await cache.get(task, [])
    assert retrieved is not None
    assert retrieved["success"] is True
    assert cache.hits == 1


@pytest.mark.asyncio
async def test_semantic_cache_different_tasks_different_keys() -> None:
    cache = SemanticCache(MemoryCache())
    t1 = TaskSpec(content="task one")
    t2 = TaskSpec(content="task two")
    await cache.set(t1, [], {"id": 1})
    assert await cache.get(t2, []) is None


@pytest.mark.asyncio
async def test_semantic_cache_execute_flag_separates_keys() -> None:
    cache = SemanticCache(MemoryCache())
    task = _make_task()
    await cache.set(task, [], {"dry": True}, execute=False)
    assert await cache.get(task, [], execute=True) is None
    assert await cache.get(task, [], execute=False) is not None


@pytest.mark.asyncio
async def test_semantic_cache_hit_rate() -> None:
    cache = SemanticCache(MemoryCache())
    task = _make_task()
    # 2 misses
    await cache.get(task, [])
    await cache.get(task, [])
    # store + 1 hit
    await cache.set(task, [], {"ok": True})
    await cache.get(task, [])
    assert cache.hits == 1
    assert cache.misses == 2
    assert abs(cache.hit_rate - 1 / 3) < 0.001


@pytest.mark.asyncio
async def test_semantic_cache_clear_resets_counters() -> None:
    cache = SemanticCache(MemoryCache())
    task = _make_task()
    await cache.set(task, [], {"x": 1})
    await cache.get(task, [])
    await cache.clear()
    assert cache.hits == 0
    assert cache.misses == 0
    assert await cache.get(task, []) is None


@pytest.mark.asyncio
async def test_semantic_cache_with_context() -> None:
    cache = SemanticCache(MemoryCache())
    task = _make_task()
    await cache.set(task, ["ctx1"], {"result": "a"})
    # Different context — should miss
    assert await cache.get(task, ["ctx2"]) is None
    # Same context — should hit
    assert await cache.get(task, ["ctx1"]) is not None


@pytest.mark.asyncio
async def test_semantic_cache_ttl_ignored_for_memory() -> None:
    """SemanticCache with ttl_seconds passes it to the backend; MemoryCache ignores it."""
    cache = SemanticCache(MemoryCache(), ttl_seconds=60)
    task = _make_task()
    await cache.set(task, [], {"ok": True})
    result = await cache.get(task, [])
    assert result is not None


# ---------------------------------------------------------------------------
# xxhash key generation
# ---------------------------------------------------------------------------


def test_xxhash_key_consistent_for_identical_inputs() -> None:
    """Same task + context + execute flag must always produce the same key."""
    cache = SemanticCache(MemoryCache())
    task = _make_task()
    k1 = cache._make_key(task, ["a", "b"], False)
    k2 = cache._make_key(task, ["a", "b"], False)
    assert k1 == k2


def test_xxhash_different_inputs_produce_different_keys() -> None:
    cache = SemanticCache(MemoryCache())
    task = _make_task()
    k1 = cache._make_key(task, ["a"], False)
    k2 = cache._make_key(task, ["b"], False)
    k3 = cache._make_key(task, ["a"], True)
    assert k1 != k2
    assert k1 != k3
    assert k2 != k3


def test_xxhash_key_is_16_hex_chars() -> None:
    """XXH3-64 hexdigest is 16 characters (8 bytes = 64 bits)."""
    cache = SemanticCache(MemoryCache())
    key = cache._make_key(_make_task(), [], False)
    assert len(key) == 16
    assert all(c in "0123456789abcdef" for c in key)


def test_xxhash_key_generation_speed() -> None:
    """Key generation must complete 10 000 iterations in under 500ms total."""
    import time

    cache = SemanticCache(MemoryCache())
    task = _make_task()
    start = time.monotonic()
    for _ in range(10_000):
        cache._make_key(task, [], False)
    elapsed_ms = (time.monotonic() - start) * 1000
    # 10k iterations well under 500ms — proves ~0.05ms/op not ~2ms
    assert elapsed_ms < 500, f"10k hashes took {elapsed_ms:.1f}ms (expected <500ms)"
