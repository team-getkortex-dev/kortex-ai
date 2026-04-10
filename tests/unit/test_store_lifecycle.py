"""Unit tests for store lifecycle management.

Verifies that StateManager and KortexRuntime properly handle
connect/disconnect lifecycle for all backends.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kortex.core.exceptions import StateError
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.store.memory import InMemoryStateStore


def _build_router() -> Router:
    router = Router()
    router.register_model(ProviderModel(
        provider="test",
        model="test-model",
        cost_per_1k_input_tokens=0.001,
        cost_per_1k_output_tokens=0.002,
        avg_latency_ms=200,
        capabilities=["reasoning"],
        tier="fast",
    ))
    return router


# ---------------------------------------------------------------------------
# 1. Memory backend works without start() (backward compatible)
# ---------------------------------------------------------------------------


class TestMemoryBackwardCompat:
    @pytest.mark.asyncio
    async def test_memory_works_without_start(self) -> None:
        mgr = StateManager.create("memory")
        ctx = await mgr.handoff("a", "b", {"key": "value"})
        assert ctx.source_agent == "a"
        assert ctx.target_agent == "b"

    @pytest.mark.asyncio
    async def test_memory_start_is_noop(self) -> None:
        mgr = StateManager.create("memory")
        await mgr.start()  # should not raise
        ctx = await mgr.handoff("a", "b", {"key": "value"})
        assert ctx.source_agent == "a"


# ---------------------------------------------------------------------------
# 2. SQLite FAILS if you handoff without start()
# ---------------------------------------------------------------------------


class TestSQLiteRequiresStart:
    @pytest.mark.asyncio
    async def test_sqlite_fails_without_start(self) -> None:
        mgr = StateManager.create("sqlite", db_path=":memory:")
        with pytest.raises(RuntimeError, match="not connected"):
            await mgr.handoff("a", "b", {"key": "value"})


# ---------------------------------------------------------------------------
# 3. create_and_connect works immediately after creation
# ---------------------------------------------------------------------------


class TestCreateAndConnect:
    @pytest.mark.asyncio
    async def test_sqlite_create_and_connect(self) -> None:
        mgr = await StateManager.create_and_connect("sqlite", db_path=":memory:")
        try:
            ctx = await mgr.handoff("a", "b", {"data": "test"})
            assert ctx.source_agent == "a"
            restored = await mgr.rollback(ctx.checkpoint_id)
            assert restored.checkpoint_id == ctx.checkpoint_id
        finally:
            await mgr.stop()

    @pytest.mark.asyncio
    async def test_memory_create_and_connect(self) -> None:
        mgr = await StateManager.create_and_connect("memory")
        try:
            ctx = await mgr.handoff("x", "y", {"key": "val"})
            assert ctx.source_agent == "x"
        finally:
            await mgr.stop()


# ---------------------------------------------------------------------------
# 4. StateManager as async context manager calls start/stop
# ---------------------------------------------------------------------------


class TestStateManagerContextManager:
    @pytest.mark.asyncio
    async def test_context_manager_start_stop(self) -> None:
        mgr = StateManager.create("sqlite", db_path=":memory:")

        async with mgr:
            ctx = await mgr.handoff("a", "b", {"key": "value"})
            assert ctx.source_agent == "a"

        # After exit, operations should fail
        with pytest.raises(StateError):
            await mgr.handoff("c", "d", {"key": "value"})

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self) -> None:
        mgr = StateManager.create("memory")
        async with mgr as m:
            assert m is mgr


# ---------------------------------------------------------------------------
# 5. KortexRuntime as async context manager starts/stops state manager
# ---------------------------------------------------------------------------


class TestRuntimeContextManager:
    @pytest.mark.asyncio
    async def test_runtime_context_manager(self) -> None:
        from kortex.core.types import TaskSpec

        state = StateManager.create("sqlite", db_path=":memory:")
        router = _build_router()
        runtime = KortexRuntime(router=router, state_manager=state)
        runtime.register_agent(AgentDescriptor("a", "A", "Agent A", ["reasoning"]))

        async with runtime:
            task = TaskSpec(content="test", complexity_hint="simple")
            result = await runtime.coordinate(task, ["a"])
            assert result.success is True

        # After exit, state manager is stopped
        with pytest.raises(StateError):
            await state.handoff("x", "y", {"k": "v"})

    @pytest.mark.asyncio
    async def test_runtime_context_manager_returns_self(self) -> None:
        state = StateManager(store=InMemoryStateStore())
        router = _build_router()
        runtime = KortexRuntime(router=router, state_manager=state)

        async with runtime as r:
            assert r is runtime


# ---------------------------------------------------------------------------
# 6. SQLite store connect() creates the table, disconnect() closes cleanly
# ---------------------------------------------------------------------------


class TestSQLiteConnectDisconnect:
    @pytest.mark.asyncio
    async def test_sqlite_connect_creates_table(self) -> None:
        from kortex.store.sqlite import SQLiteStateStore

        store = SQLiteStateStore(db_path=":memory:")
        await store.connect()
        try:
            # Verify we can write — which means the table was created
            from kortex.core.types import HandoffContext

            ctx = HandoffContext(
                source_agent="a",
                target_agent="b",
                state_snapshot={"key": "value"},
            )
            checkpoint_id = await store.save_checkpoint(ctx)
            assert checkpoint_id == ctx.checkpoint_id

            # Verify we can read it back
            restored = await store.get_checkpoint(checkpoint_id)
            assert restored.source_agent == "a"
        finally:
            await store.disconnect()

    @pytest.mark.asyncio
    async def test_sqlite_disconnect_closes_cleanly(self) -> None:
        from kortex.store.sqlite import SQLiteStateStore

        store = SQLiteStateStore(db_path=":memory:")
        await store.connect()
        await store.disconnect()

        # After disconnect, operations should fail
        with pytest.raises(RuntimeError, match="not connected"):
            from kortex.core.types import HandoffContext

            ctx = HandoffContext(
                source_agent="a",
                target_agent="b",
                state_snapshot={},
            )
            await store.save_checkpoint(ctx)


# ---------------------------------------------------------------------------
# 7. Redis store connect() verifies connectivity (mocked)
# ---------------------------------------------------------------------------


class TestRedisConnectMocked:
    @pytest.mark.asyncio
    async def test_redis_connect_pings(self) -> None:
        from kortex.store.redis import RedisStateStore

        store = RedisStateStore(redis_url="redis://localhost:6379")

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            await store.connect()
            mock_redis.ping.assert_awaited_once()
            await store.disconnect()
            mock_redis.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# 8. Double start() is safe (idempotent)
# ---------------------------------------------------------------------------


class TestDoubleStart:
    @pytest.mark.asyncio
    async def test_double_start_memory(self) -> None:
        mgr = StateManager.create("memory")
        await mgr.start()
        await mgr.start()  # should not raise
        ctx = await mgr.handoff("a", "b", {"key": "value"})
        assert ctx.source_agent == "a"

    @pytest.mark.asyncio
    async def test_double_start_sqlite(self) -> None:
        mgr = StateManager.create("sqlite", db_path=":memory:")
        await mgr.start()
        await mgr.start()  # idempotent — should not raise or reconnect
        ctx = await mgr.handoff("a", "b", {"key": "value"})
        assert ctx.source_agent == "a"
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self) -> None:
        mgr = StateManager.create("sqlite", db_path=":memory:")
        await mgr.start()
        await mgr.stop()
        await mgr.stop()  # should not raise


# ---------------------------------------------------------------------------
# 9. Operations after stop() raise a clear error
# ---------------------------------------------------------------------------


class TestOperationsAfterStop:
    @pytest.mark.asyncio
    async def test_handoff_after_stop(self) -> None:
        mgr = StateManager.create("memory")
        await mgr.start()
        await mgr.stop()

        with pytest.raises(StateError, match="stopped"):
            await mgr.handoff("a", "b", {"key": "value"})

    @pytest.mark.asyncio
    async def test_batch_handoff_after_stop(self) -> None:
        mgr = StateManager.create("memory")
        await mgr.start()
        await mgr.stop()

        with pytest.raises(StateError, match="stopped"):
            await mgr.batch_handoff([("a", "b", {"k": "v"}, None)])

    @pytest.mark.asyncio
    async def test_rollback_after_stop(self) -> None:
        mgr = StateManager.create("memory")
        await mgr.start()
        ctx = await mgr.handoff("a", "b", {"key": "value"})
        await mgr.stop()

        with pytest.raises(StateError, match="stopped"):
            await mgr.rollback(ctx.checkpoint_id)

    @pytest.mark.asyncio
    async def test_get_history_after_stop(self) -> None:
        mgr = StateManager.create("memory")
        await mgr.start()
        ctx = await mgr.handoff("a", "b", {"key": "value"})
        await mgr.stop()

        with pytest.raises(StateError, match="stopped"):
            await mgr.get_history(ctx.checkpoint_id)

    @pytest.mark.asyncio
    async def test_get_latest_after_stop(self) -> None:
        mgr = StateManager.create("memory")
        await mgr.start()
        await mgr.stop()

        with pytest.raises(StateError, match="stopped"):
            await mgr.get_latest("a")

    @pytest.mark.asyncio
    async def test_restart_after_stop_works(self) -> None:
        """Calling start() again after stop() should reconnect."""
        mgr = StateManager.create("memory")
        await mgr.start()
        await mgr.stop()

        # Restart
        await mgr.start()
        ctx = await mgr.handoff("a", "b", {"key": "value"})
        assert ctx.source_agent == "a"
        await mgr.stop()
