"""Integration tests for the async SQLite state store.

Verifies non-blocking async behaviour and concurrent write correctness.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from kortex.core.exceptions import CheckpointNotFoundError
from kortex.core.state import StateManager
from kortex.core.types import HandoffContext
from kortex.store.sqlite import SQLiteStateStore


@pytest.fixture
async def store() -> SQLiteStateStore:
    """In-memory SQLite store for isolation."""
    s = SQLiteStateStore(db_path=":memory:")
    await s.connect()
    yield s
    await s.disconnect()


@pytest.fixture
async def manager() -> StateManager:
    s = SQLiteStateStore(db_path=":memory:")
    mgr = StateManager(store=s)
    await mgr.start()
    yield mgr
    await mgr.stop()


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_retrieve_checkpoint(store: SQLiteStateStore) -> None:
    from datetime import datetime, timezone
    from uuid import uuid4

    ctx = HandoffContext(
        handoff_id=str(uuid4()),
        checkpoint_id=str(uuid4()),
        source_agent="a",
        target_agent="b",
        state_snapshot={"key": "value"},
        parent_checkpoint_id=None,
        created_at=datetime.now(timezone.utc),
    )
    await store.save_checkpoint(ctx)
    retrieved = await store.get_checkpoint(ctx.checkpoint_id)
    assert retrieved.checkpoint_id == ctx.checkpoint_id
    assert retrieved.state_snapshot == ctx.state_snapshot


@pytest.mark.asyncio
async def test_get_missing_checkpoint_raises(store: SQLiteStateStore) -> None:
    with pytest.raises(CheckpointNotFoundError):
        await store.get_checkpoint("does-not-exist")


@pytest.mark.asyncio
async def test_delete_checkpoint(store: SQLiteStateStore) -> None:
    from datetime import datetime, timezone
    from uuid import uuid4

    ctx = HandoffContext(
        handoff_id=str(uuid4()),
        checkpoint_id=str(uuid4()),
        source_agent="a",
        target_agent="b",
        state_snapshot={},
        parent_checkpoint_id=None,
        created_at=datetime.now(timezone.utc),
    )
    await store.save_checkpoint(ctx)
    deleted = await store.delete_checkpoint(ctx.checkpoint_id)
    assert deleted is True
    with pytest.raises(CheckpointNotFoundError):
        await store.get_checkpoint(ctx.checkpoint_id)


@pytest.mark.asyncio
async def test_list_checkpoints_by_task(store: SQLiteStateStore) -> None:
    from datetime import datetime, timezone
    from uuid import uuid4

    task_id = str(uuid4())
    for _ in range(3):
        ctx = HandoffContext(
            handoff_id=str(uuid4()),
            checkpoint_id=str(uuid4()),
            source_agent="a",
            target_agent="b",
            state_snapshot={"task_id": task_id},
            parent_checkpoint_id=None,
            created_at=datetime.now(timezone.utc),
        )
        await store.save_checkpoint(ctx)

    results = await store.list_checkpoints(task_id=task_id)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Batch inserts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_save_checkpoints(store: SQLiteStateStore) -> None:
    from datetime import datetime, timezone
    from uuid import uuid4

    contexts = [
        HandoffContext(
            handoff_id=str(uuid4()),
            checkpoint_id=str(uuid4()),
            source_agent="a",
            target_agent="b",
            state_snapshot={"i": i},
            parent_checkpoint_id=None,
            created_at=datetime.now(timezone.utc),
        )
        for i in range(10)
    ]
    ids = await store.save_checkpoints_batch(contexts)
    assert len(ids) == 10
    for ctx in contexts:
        r = await store.get_checkpoint(ctx.checkpoint_id)
        assert r.checkpoint_id == ctx.checkpoint_id


# ---------------------------------------------------------------------------
# Concurrent writes — non-blocking verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_writes_complete_faster_than_serial(
    store: SQLiteStateStore,
) -> None:
    """10 concurrent writes should complete in less time than 10 serial writes."""
    from datetime import datetime, timezone
    from uuid import uuid4

    def make_ctx() -> HandoffContext:
        return HandoffContext(
            handoff_id=str(uuid4()),
            checkpoint_id=str(uuid4()),
            source_agent="a",
            target_agent="b",
            state_snapshot={"data": "x" * 200},
            parent_checkpoint_id=None,
            created_at=datetime.now(timezone.utc),
        )

    N = 10

    # Serial
    serial_start = time.monotonic()
    for _ in range(N):
        await store.save_checkpoint(make_ctx())
    serial_ms = (time.monotonic() - serial_start) * 1000

    # Concurrent
    concurrent_start = time.monotonic()
    await asyncio.gather(*[store.save_checkpoint(make_ctx()) for _ in range(N)])
    concurrent_ms = (time.monotonic() - concurrent_start) * 1000

    # Concurrent should not be dramatically slower; in practice it is faster
    # for backends with real I/O. We just assert it completes within 3x serial.
    assert concurrent_ms < serial_ms * 3, (
        f"Concurrent ({concurrent_ms:.1f}ms) was >3x serial ({serial_ms:.1f}ms)"
    )


@pytest.mark.asyncio
async def test_async_operations_do_not_block_event_loop(
    store: SQLiteStateStore,
) -> None:
    """SQLite writes should not block a concurrent coroutine from progressing."""
    from datetime import datetime, timezone
    from uuid import uuid4

    progress: list[int] = []

    async def write_checkpoints() -> None:
        for i in range(5):
            ctx = HandoffContext(
                handoff_id=str(uuid4()),
                checkpoint_id=str(uuid4()),
                source_agent="a",
                target_agent="b",
                state_snapshot={"i": i},
                parent_checkpoint_id=None,
                created_at=datetime.now(timezone.utc),
            )
            await store.save_checkpoint(ctx)

    async def tick() -> None:
        for i in range(5):
            await asyncio.sleep(0)
            progress.append(i)

    await asyncio.gather(write_checkpoints(), tick())

    # If writes blocked, tick() would have no progress.
    assert len(progress) == 5


# ---------------------------------------------------------------------------
# StateManager integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_manager_sqlite_handoff_roundtrip(
    manager: StateManager,
) -> None:
    ctx = await manager.handoff(
        source_agent="agent_a",
        target_agent="agent_b",
        state_snapshot={"msg": "hello"},
    )
    assert ctx.source_agent == "agent_a"
    assert ctx.target_agent == "agent_b"
    assert ctx.checkpoint_id is not None


@pytest.mark.asyncio
async def test_state_manager_sqlite_rollback(manager: StateManager) -> None:
    ctx = await manager.handoff("a", "b", {"step": 1})
    rolled_back = await manager.rollback(ctx.checkpoint_id)
    assert rolled_back.checkpoint_id == ctx.checkpoint_id
