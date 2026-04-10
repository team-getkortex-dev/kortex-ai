"""Tests for trace storage backends."""

from __future__ import annotations

import pytest

from kortex.core.trace import TaskTrace, TraceStep
from kortex.core.trace_store import InMemoryTraceStore, SQLiteTraceStore


def _sample_trace(trace_id: str = "t1", task_id: str = "task-1") -> TaskTrace:
    return TaskTrace(
        trace_id=trace_id,
        task_id=task_id,
        task_content="Test task",
        task_complexity="moderate",
        pipeline=["a", "b"],
        steps=[
            TraceStep(
                step_index=0,
                agent_id="a",
                input_payload={"content": "test"},
                routing_decision={"chosen_model": "m1"},
                policy_snapshot={"name": "default"},
            ),
        ],
        total_estimated_cost_usd=0.01,
        success=True,
        created_at="2026-03-29T10:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# 12. InMemoryTraceStore saves and retrieves traces
# ---------------------------------------------------------------------------


class TestInMemorySaveRetrieve:
    @pytest.mark.asyncio
    async def test_save_and_get(self) -> None:
        store = InMemoryTraceStore()
        trace = _sample_trace()
        tid = await store.save_trace(trace)
        assert tid == "t1"

        retrieved = await store.get_trace("t1")
        assert retrieved.task_id == "task-1"
        assert retrieved.trace_id == "t1"

    @pytest.mark.asyncio
    async def test_get_missing_raises(self) -> None:
        store = InMemoryTraceStore()
        with pytest.raises(KeyError):
            await store.get_trace("nonexistent")

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        store = InMemoryTraceStore()
        await store.save_trace(_sample_trace())
        assert await store.delete_trace("t1")
        assert not await store.delete_trace("t1")


# ---------------------------------------------------------------------------
# 13. InMemoryTraceStore lists traces by task_id
# ---------------------------------------------------------------------------


class TestInMemoryListByTask:
    @pytest.mark.asyncio
    async def test_list_by_task_id(self) -> None:
        store = InMemoryTraceStore()
        await store.save_trace(_sample_trace("t1", "task-A"))
        await store.save_trace(_sample_trace("t2", "task-A"))
        await store.save_trace(_sample_trace("t3", "task-B"))

        task_a = await store.list_traces(task_id="task-A")
        assert len(task_a) == 2
        assert all(t.task_id == "task-A" for t in task_a)

        task_b = await store.list_traces(task_id="task-B")
        assert len(task_b) == 1

    @pytest.mark.asyncio
    async def test_list_all(self) -> None:
        store = InMemoryTraceStore()
        await store.save_trace(_sample_trace("t1", "task-A"))
        await store.save_trace(_sample_trace("t2", "task-B"))

        all_traces = await store.list_traces()
        assert len(all_traces) == 2


# ---------------------------------------------------------------------------
# 14. SQLiteTraceStore saves and retrieves (use :memory:)
# ---------------------------------------------------------------------------


class TestSQLiteSaveRetrieve:
    @pytest.mark.asyncio
    async def test_save_and_get(self) -> None:
        async with SQLiteTraceStore(":memory:") as store:
            trace = _sample_trace()
            tid = await store.save_trace(trace)
            assert tid == "t1"

            retrieved = await store.get_trace("t1")
            assert retrieved.task_id == "task-1"
            assert retrieved.trace_id == "t1"
            assert len(retrieved.steps) == 1

    @pytest.mark.asyncio
    async def test_get_missing_raises(self) -> None:
        async with SQLiteTraceStore(":memory:") as store:
            with pytest.raises(KeyError):
                await store.get_trace("nonexistent")


# ---------------------------------------------------------------------------
# 15. SQLiteTraceStore lists traces with limit
# ---------------------------------------------------------------------------


class TestSQLiteListWithLimit:
    @pytest.mark.asyncio
    async def test_list_with_limit(self) -> None:
        async with SQLiteTraceStore(":memory:") as store:
            for i in range(10):
                t = _sample_trace(f"t{i}", "task-1")
                # Vary created_at to test ordering
                t.created_at = f"2026-03-29T10:{i:02d}:00+00:00"
                await store.save_trace(t)

            limited = await store.list_traces(limit=3)
            assert len(limited) == 3

            all_traces = await store.list_traces(limit=50)
            assert len(all_traces) == 10

    @pytest.mark.asyncio
    async def test_list_by_task_id(self) -> None:
        async with SQLiteTraceStore(":memory:") as store:
            await store.save_trace(_sample_trace("t1", "task-A"))
            await store.save_trace(_sample_trace("t2", "task-B"))

            task_a = await store.list_traces(task_id="task-A")
            assert len(task_a) == 1
            assert task_a[0].task_id == "task-A"


# ---------------------------------------------------------------------------
# 16. Trace store lifecycle (connect/disconnect) works
# ---------------------------------------------------------------------------


class TestSQLiteLifecycle:
    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        store = SQLiteTraceStore(":memory:")
        await store.connect()

        await store.save_trace(_sample_trace())
        retrieved = await store.get_trace("t1")
        assert retrieved.task_id == "task-1"

        await store.disconnect()

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        async with SQLiteTraceStore(":memory:") as store:
            await store.save_trace(_sample_trace())
            assert await store.delete_trace("t1")
            assert not await store.delete_trace("t1")
