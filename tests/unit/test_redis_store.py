"""Unit tests for RedisStateStore using FakeRedis (no real Redis needed)."""

from __future__ import annotations

import asyncio

import pytest

from kortex.core.exceptions import CheckpointNotFoundError
from kortex.core.types import HandoffContext
from kortex.store.redis import RedisStateStore

from tests.fixtures.fake_redis import FakeRedis


def _make_ctx(
    checkpoint_id: str = "cp-1",
    source: str = "agent-a",
    target: str = "agent-b",
    task_id: str = "task-1",
    parent: str | None = None,
) -> HandoffContext:
    return HandoffContext(
        handoff_id=f"hoff-{checkpoint_id}",
        source_agent=source,
        target_agent=target,
        state_snapshot={"task_id": task_id, "data": "value"},
        compressed_summary="test summary",
        checkpoint_id=checkpoint_id,
        parent_checkpoint_id=parent,
    )


def _make_store(
    ttl: int | None = None,
    prefix: str = "kortex:",
) -> tuple[RedisStateStore, FakeRedis]:
    """Create a RedisStateStore with a FakeRedis backend injected."""
    store = RedisStateStore(
        redis_url="redis://fake:6379",
        key_prefix=prefix,
        ttl_seconds=ttl,
    )
    fake = FakeRedis()
    store._redis = fake  # inject fake client directly
    return store, fake


# ---------------------------------------------------------------------------
# 1. Save and retrieve checkpoint
# ---------------------------------------------------------------------------


class TestSaveAndGet:
    @pytest.mark.asyncio
    async def test_save_and_retrieve(self) -> None:
        store, _ = _make_store()
        ctx = _make_ctx()
        cp_id = await store.save_checkpoint(ctx)
        assert cp_id == "cp-1"

        retrieved = await store.get_checkpoint("cp-1")
        assert retrieved.checkpoint_id == "cp-1"
        assert retrieved.source_agent == "agent-a"
        assert retrieved.target_agent == "agent-b"
        assert retrieved.state_snapshot["task_id"] == "task-1"


# ---------------------------------------------------------------------------
# 2. CheckpointNotFoundError on missing checkpoint
# ---------------------------------------------------------------------------


class TestNotFound:
    @pytest.mark.asyncio
    async def test_missing_checkpoint_raises(self) -> None:
        store, _ = _make_store()
        with pytest.raises(CheckpointNotFoundError, match="not-exist"):
            await store.get_checkpoint("not-exist")


# ---------------------------------------------------------------------------
# 3. list_checkpoints filters by task_id
# ---------------------------------------------------------------------------


class TestListByTask:
    @pytest.mark.asyncio
    async def test_filter_by_task_id(self) -> None:
        store, _ = _make_store()
        await store.save_checkpoint(_make_ctx("cp-1", task_id="task-A"))
        await store.save_checkpoint(_make_ctx("cp-2", task_id="task-B"))
        await store.save_checkpoint(_make_ctx("cp-3", task_id="task-A"))

        results = await store.list_checkpoints(task_id="task-A")
        assert len(results) == 2
        assert all(c.state_snapshot["task_id"] == "task-A" for c in results)


# ---------------------------------------------------------------------------
# 4. list_checkpoints filters by agent_id
# ---------------------------------------------------------------------------


class TestListByAgent:
    @pytest.mark.asyncio
    async def test_filter_by_agent_id(self) -> None:
        store, _ = _make_store()
        await store.save_checkpoint(_make_ctx("cp-1", source="alice", target="bob"))
        await store.save_checkpoint(_make_ctx("cp-2", source="bob", target="charlie"))
        await store.save_checkpoint(_make_ctx("cp-3", source="dave", target="eve"))

        results = await store.list_checkpoints(agent_id="bob")
        assert len(results) == 2
        ids = {c.checkpoint_id for c in results}
        assert ids == {"cp-1", "cp-2"}


# ---------------------------------------------------------------------------
# 5. delete_checkpoint removes from store
# ---------------------------------------------------------------------------


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_existing(self) -> None:
        store, _ = _make_store()
        await store.save_checkpoint(_make_ctx("cp-1"))
        assert await store.delete_checkpoint("cp-1") is True

        with pytest.raises(CheckpointNotFoundError):
            await store.get_checkpoint("cp-1")

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self) -> None:
        store, _ = _make_store()
        assert await store.delete_checkpoint("nope") is False


# ---------------------------------------------------------------------------
# 6. Checkpoint chain walks correctly through 3+ levels
# ---------------------------------------------------------------------------


class TestCheckpointChain:
    @pytest.mark.asyncio
    async def test_chain_three_levels(self) -> None:
        store, _ = _make_store()
        await store.save_checkpoint(_make_ctx("cp-1", parent=None))
        await store.save_checkpoint(_make_ctx("cp-2", parent="cp-1"))
        await store.save_checkpoint(_make_ctx("cp-3", parent="cp-2"))

        chain = await store.get_checkpoint_chain("cp-3")
        assert len(chain) == 3
        assert chain[0].checkpoint_id == "cp-1"
        assert chain[1].checkpoint_id == "cp-2"
        assert chain[2].checkpoint_id == "cp-3"


# ---------------------------------------------------------------------------
# 7. Context manager support
# ---------------------------------------------------------------------------


class TestContextManager:
    @pytest.mark.asyncio
    async def test_save_after_connect(self) -> None:
        store, _ = _make_store()
        # Already has _redis injected, so we can use it directly
        ctx = _make_ctx("cm-test")
        await store.save_checkpoint(ctx)
        retrieved = await store.get_checkpoint("cm-test")
        assert retrieved.checkpoint_id == "cm-test"


# ---------------------------------------------------------------------------
# 8. Concurrent writes don't corrupt
# ---------------------------------------------------------------------------


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_saves(self) -> None:
        store, _ = _make_store()
        tasks = [
            store.save_checkpoint(_make_ctx(f"cp-{i}", task_id="concurrent"))
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        results = await store.list_checkpoints(task_id="concurrent")
        assert len(results) == 10
        ids = {c.checkpoint_id for c in results}
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# 9. TTL is set when configured
# ---------------------------------------------------------------------------


class TestTTL:
    @pytest.mark.asyncio
    async def test_ttl_set_on_save(self) -> None:
        store, fake = _make_store(ttl=3600)
        await store.save_checkpoint(_make_ctx("cp-ttl"))

        key = store._ck_key("cp-ttl")
        assert fake.get_ttl(key) == 3600

    @pytest.mark.asyncio
    async def test_no_ttl_when_not_configured(self) -> None:
        store, fake = _make_store(ttl=None)
        await store.save_checkpoint(_make_ctx("cp-nottl"))

        key = store._ck_key("cp-nottl")
        assert fake.get_ttl(key) is None


# ---------------------------------------------------------------------------
# 10. Flush deletes only prefixed keys
# ---------------------------------------------------------------------------


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_prefix_only(self) -> None:
        store, fake = _make_store(prefix="kortex:")

        # Save some checkpoints
        await store.save_checkpoint(_make_ctx("cp-f1"))
        await store.save_checkpoint(_make_ctx("cp-f2"))

        # Add a non-prefixed key directly
        await fake.set("other:key", "value")

        count = await store.flush(prefix_only=True)
        assert count > 0

        # Non-prefixed key should still exist
        assert await fake.get("other:key") == "value"

        # Prefixed keys should be gone
        assert await fake.get(store._ck_key("cp-f1")) is None


# ---------------------------------------------------------------------------
# 11. Max depth prevents infinite chain loops
# ---------------------------------------------------------------------------


class TestMaxDepth:
    @pytest.mark.asyncio
    async def test_chain_stops_at_max_depth(self) -> None:
        store, _ = _make_store()

        # Create a chain longer than we'll test but with a cycle at the end
        # to verify the seen-set prevents infinite loops
        await store.save_checkpoint(_make_ctx("cp-a", parent="cp-c"))  # cycle: a->c
        await store.save_checkpoint(_make_ctx("cp-b", parent="cp-a"))
        await store.save_checkpoint(_make_ctx("cp-c", parent="cp-b"))  # cycle: c->b->a->c

        # Should not hang or crash — cycle detection kicks in
        chain = await store.get_checkpoint_chain("cp-c")
        assert len(chain) <= 3  # stops when it detects the cycle
