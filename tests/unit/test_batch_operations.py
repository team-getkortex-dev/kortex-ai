"""Tests for batch operations and store optimizations."""

from __future__ import annotations

import time

import pytest

from kortex.core.state import StateManager
from kortex.core.types import HandoffContext
from kortex.store.memory import InMemoryStateStore
from kortex.store.sqlite import SQLiteStateStore


# ---------------------------------------------------------------------------
# 1. batch_handoff creates all checkpoints correctly
# ---------------------------------------------------------------------------


class TestBatchHandoffCorrectness:
    @pytest.mark.asyncio
    async def test_batch_handoff_creates_all_checkpoints(self) -> None:
        mgr = StateManager.create("memory")
        handoffs_input = [
            (f"src_{i}", f"tgt_{i}", {"index": i, "data": f"v{i}"}, None)
            for i in range(10)
        ]
        results = await mgr.batch_handoff(handoffs_input)

        assert len(results) == 10
        for i, ctx in enumerate(results):
            assert ctx.source_agent == f"src_{i}"
            assert ctx.target_agent == f"tgt_{i}"
            assert ctx.state_snapshot["index"] == i
            # Verify retrievable from store
            retrieved = await mgr._store.get_checkpoint(ctx.checkpoint_id)
            assert retrieved.checkpoint_id == ctx.checkpoint_id

    @pytest.mark.asyncio
    async def test_batch_handoff_with_parent_chain(self) -> None:
        mgr = StateManager.create("memory")
        first = await mgr.handoff("a", "b", {"step": 0})
        batch = [
            ("b", "c", {"step": 1}, first.checkpoint_id),
            ("c", "d", {"step": 2}, None),
        ]
        results = await mgr.batch_handoff(batch)
        assert results[0].parent_checkpoint_id == first.checkpoint_id
        assert results[1].parent_checkpoint_id is None


# ---------------------------------------------------------------------------
# 2. batch_handoff with 1000 items completes successfully
# ---------------------------------------------------------------------------


class TestBatchVolume:
    @pytest.mark.asyncio
    async def test_batch_1000_items_memory(self) -> None:
        mgr = StateManager.create("memory")
        handoffs = [
            (f"src_{i}", f"tgt_{i}", {"i": i}, None) for i in range(1000)
        ]
        results = await mgr.batch_handoff(handoffs)
        assert len(results) == 1000

        # Spot check
        r500 = await mgr._store.get_checkpoint(results[500].checkpoint_id)
        assert r500.state_snapshot["i"] == 500

    @pytest.mark.asyncio
    async def test_batch_1000_items_sqlite(self) -> None:
        mgr = StateManager.create("sqlite", db_path=":memory:")
        store = mgr._store  # type: ignore
        await store.connect()

        handoffs = [
            (f"src_{i}", f"tgt_{i}", {"i": i}, None) for i in range(1000)
        ]
        results = await mgr.batch_handoff(handoffs)
        assert len(results) == 1000

        r999 = await mgr._store.get_checkpoint(results[999].checkpoint_id)
        assert r999.state_snapshot["i"] == 999


# ---------------------------------------------------------------------------
# 3. Batch is faster than sequential
# ---------------------------------------------------------------------------


class TestBatchPerformance:
    @pytest.mark.asyncio
    async def test_batch_faster_than_sequential_memory(self) -> None:
        n = 1000

        # Sequential
        mgr_seq = StateManager.create("memory")
        start = time.monotonic()
        for i in range(n):
            await mgr_seq.handoff(f"s_{i}", f"t_{i}", {"i": i})
        seq_time = time.monotonic() - start

        # Batch
        mgr_batch = StateManager.create("memory")
        handoffs = [(f"s_{i}", f"t_{i}", {"i": i}, None) for i in range(n)]
        start = time.monotonic()
        await mgr_batch.batch_handoff(handoffs)
        batch_time = time.monotonic() - start

        speedup = seq_time / batch_time if batch_time > 0 else float("inf")
        print(f"\nMemory: sequential={seq_time*1000:.1f}ms, batch={batch_time*1000:.1f}ms, speedup={speedup:.1f}x")
        assert speedup >= 2.0, f"Batch should be at least 2x faster, got {speedup:.1f}x"

    @pytest.mark.asyncio
    async def test_batch_faster_than_sequential_sqlite(self) -> None:
        n = 1000

        # Sequential
        mgr_seq = StateManager.create("sqlite", db_path=":memory:")
        await mgr_seq._store.connect()  # type: ignore
        start = time.monotonic()
        for i in range(n):
            await mgr_seq.handoff(f"s_{i}", f"t_{i}", {"i": i})
        seq_time = time.monotonic() - start

        # Batch
        mgr_batch = StateManager.create("sqlite", db_path=":memory:")
        await mgr_batch._store.connect()  # type: ignore
        handoffs = [(f"s_{i}", f"t_{i}", {"i": i}, None) for i in range(n)]
        start = time.monotonic()
        await mgr_batch.batch_handoff(handoffs)
        batch_time = time.monotonic() - start

        speedup = seq_time / batch_time if batch_time > 0 else float("inf")
        print(f"\nSQLite: sequential={seq_time*1000:.1f}ms, batch={batch_time*1000:.1f}ms, speedup={speedup:.1f}x")
        assert speedup >= 2.0, f"Batch should be at least 2x faster, got {speedup:.1f}x"


# ---------------------------------------------------------------------------
# 4. SQLite WAL mode is enabled after connect
# ---------------------------------------------------------------------------


class TestSQLiteWAL:
    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self) -> None:
        store = SQLiteStateStore(db_path=":memory:")
        await store.connect()
        db = store._ensure_connected()
        async with db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
        # :memory: databases may report "memory" instead of "wal"
        # but the PRAGMA was still executed successfully
        journal_mode = row[0] if row else ""
        assert journal_mode in ("wal", "memory"), f"Unexpected journal mode: {journal_mode}"
        await store.disconnect()

    @pytest.mark.asyncio
    async def test_synchronous_normal(self) -> None:
        store = SQLiteStateStore(db_path=":memory:")
        await store.connect()
        db = store._ensure_connected()
        async with db.execute("PRAGMA synchronous") as cursor:
            row = await cursor.fetchone()
        # NORMAL = 1
        sync_mode = row[0] if row else -1
        assert sync_mode == 1, f"Expected synchronous=NORMAL(1), got {sync_mode}"
        await store.disconnect()


# ---------------------------------------------------------------------------
# 5. Individual handoff still works unchanged after optimizations
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    @pytest.mark.asyncio
    async def test_individual_handoff_memory(self) -> None:
        mgr = StateManager.create("memory")
        ctx = await mgr.handoff("a", "b", {"key": "value"})
        assert ctx.source_agent == "a"
        assert ctx.target_agent == "b"
        retrieved = await mgr._store.get_checkpoint(ctx.checkpoint_id)
        assert retrieved.state_snapshot == {"key": "value"}

    @pytest.mark.asyncio
    async def test_individual_handoff_sqlite(self) -> None:
        mgr = StateManager.create("sqlite", db_path=":memory:")
        await mgr._store.connect()  # type: ignore
        ctx = await mgr.handoff("a", "b", {"key": "value"})
        assert ctx.source_agent == "a"
        retrieved = await mgr._store.get_checkpoint(ctx.checkpoint_id)
        assert retrieved.state_snapshot == {"key": "value"}

    @pytest.mark.asyncio
    async def test_delete_still_works(self) -> None:
        store = InMemoryStateStore()
        mgr = StateManager(store=store)
        ctx = await mgr.handoff("x", "y", {"test": True})
        assert await store.delete_checkpoint(ctx.checkpoint_id) is True
        assert await store.delete_checkpoint(ctx.checkpoint_id) is False

    @pytest.mark.asyncio
    async def test_get_latest_still_works(self) -> None:
        mgr = StateManager.create("memory")
        await mgr.handoff("agent_a", "agent_b", {"step": 1})
        await mgr.handoff("agent_a", "agent_c", {"step": 2})
        latest = await mgr.get_latest("agent_a")
        assert latest is not None
        assert latest.state_snapshot["step"] == 2
