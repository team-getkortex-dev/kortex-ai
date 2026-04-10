"""Tests for the state manager and in-memory store."""

from __future__ import annotations

import asyncio

import pytest

from kortex.core.exceptions import CheckpointNotFoundError
from kortex.core.state import StateManager, _compress_snapshot
from kortex.core.types import HandoffContext
from kortex.store.memory import InMemoryStateStore


@pytest.fixture
def manager() -> StateManager:
    return StateManager(store=InMemoryStateStore())


# --- 1. Basic handoff ---


class TestBasicHandoff:
    @pytest.mark.asyncio
    async def test_handoff_creates_checkpoint(self, manager: StateManager) -> None:
        ctx = await manager.handoff(
            source_agent="planner",
            target_agent="executor",
            state_snapshot={"plan": "step1"},
        )
        assert isinstance(ctx, HandoffContext)
        assert ctx.source_agent == "planner"
        assert ctx.target_agent == "executor"
        assert ctx.state_snapshot == {"plan": "step1"}
        assert ctx.checkpoint_id is not None
        assert ctx.compressed_summary is not None

    @pytest.mark.asyncio
    async def test_handoff_is_retrievable(self, manager: StateManager) -> None:
        ctx = await manager.handoff(
            source_agent="a", target_agent="b", state_snapshot={"x": 1}
        )
        retrieved = await manager.rollback(ctx.checkpoint_id)
        assert retrieved.checkpoint_id == ctx.checkpoint_id


# --- 2. Rollback ---


class TestRollback:
    @pytest.mark.asyncio
    async def test_rollback_returns_correct_checkpoint(
        self, manager: StateManager
    ) -> None:
        ctx1 = await manager.handoff("a", "b", {"step": 1})
        ctx2 = await manager.handoff("b", "c", {"step": 2})
        rolled = await manager.rollback(ctx1.checkpoint_id)
        assert rolled.state_snapshot == {"step": 1}
        assert rolled.checkpoint_id == ctx1.checkpoint_id


# --- 3. Rollback invalid checkpoint ---


class TestRollbackInvalid:
    @pytest.mark.asyncio
    async def test_rollback_invalid_raises(self, manager: StateManager) -> None:
        with pytest.raises(CheckpointNotFoundError, match="not found"):
            await manager.rollback("nonexistent-id")


# --- 4. Checkpoint chain ---


class TestCheckpointChain:
    @pytest.mark.asyncio
    async def test_chain_through_three_handoffs(self, manager: StateManager) -> None:
        ctx1 = await manager.handoff("a", "b", {"step": 1})
        ctx2 = await manager.handoff(
            "b", "c", {"step": 2}, parent_checkpoint_id=ctx1.checkpoint_id
        )
        ctx3 = await manager.handoff(
            "c", "d", {"step": 3}, parent_checkpoint_id=ctx2.checkpoint_id
        )

        history = await manager.get_history(ctx3.checkpoint_id)
        assert len(history) == 3
        # root first
        assert history[0].checkpoint_id == ctx1.checkpoint_id
        assert history[1].checkpoint_id == ctx2.checkpoint_id
        assert history[2].checkpoint_id == ctx3.checkpoint_id


# --- 5. Compressed summary truncation ---


class TestCompressedSummary:
    def test_short_values_not_truncated(self) -> None:
        summary = _compress_snapshot({"key": "short"})
        assert summary == "key=short"

    def test_long_values_truncated(self) -> None:
        long_value = "x" * 600
        summary = _compress_snapshot({"data": long_value})
        assert "... [truncated]" in summary
        assert len(summary) < 600

    @pytest.mark.asyncio
    async def test_handoff_generates_summary(self, manager: StateManager) -> None:
        ctx = await manager.handoff(
            "a", "b", {"big_field": "y" * 1000, "small": "ok"}
        )
        assert ctx.compressed_summary is not None
        assert "... [truncated]" in ctx.compressed_summary
        assert "small=ok" in ctx.compressed_summary


# --- 6. get_latest ---


class TestGetLatest:
    @pytest.mark.asyncio
    async def test_returns_most_recent_for_agent(self, manager: StateManager) -> None:
        await manager.handoff("a", "b", {"v": 1})
        ctx2 = await manager.handoff("a", "b", {"v": 2})
        latest = await manager.get_latest("b")
        assert latest is not None
        assert latest.checkpoint_id == ctx2.checkpoint_id

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_agent(
        self, manager: StateManager
    ) -> None:
        result = await manager.get_latest("nobody")
        assert result is None

    @pytest.mark.asyncio
    async def test_matches_source_or_target(self, manager: StateManager) -> None:
        ctx = await manager.handoff("agent_x", "agent_y", {"v": 1})
        assert (await manager.get_latest("agent_x")) is not None
        assert (await manager.get_latest("agent_y")) is not None


# --- 7. Concurrent handoffs ---


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_handoffs_safe(self) -> None:
        store = InMemoryStateStore()
        manager = StateManager(store=store)

        async def do_handoff(i: int) -> HandoffContext:
            return await manager.handoff("src", f"tgt-{i}", {"i": i})

        results = await asyncio.gather(*[do_handoff(i) for i in range(10)])
        assert len(results) == 10

        # All checkpoint IDs are unique
        ids = [r.checkpoint_id for r in results]
        assert len(set(ids)) == 10

        # All are retrievable
        for ctx in results:
            retrieved = await manager.rollback(ctx.checkpoint_id)
            assert retrieved.checkpoint_id == ctx.checkpoint_id


# --- 8. Round-trip ---


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_save_get_compare(self, manager: StateManager) -> None:
        original = await manager.handoff(
            source_agent="writer",
            target_agent="reviewer",
            state_snapshot={"doc": "hello world", "version": 3},
        )
        retrieved = await manager.rollback(original.checkpoint_id)

        assert retrieved.handoff_id == original.handoff_id
        assert retrieved.source_agent == original.source_agent
        assert retrieved.target_agent == original.target_agent
        assert retrieved.state_snapshot == original.state_snapshot
        assert retrieved.compressed_summary == original.compressed_summary
        assert retrieved.checkpoint_id == original.checkpoint_id
        assert retrieved.parent_checkpoint_id == original.parent_checkpoint_id
        assert retrieved.created_at == original.created_at
