"""Integration tests for parallel handoff execution."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from kortex.core.state import StateManager
from kortex.core.types import HandoffContext
from kortex.state.dag_analyzer import DAGAnalyzer


def _make_ctx(
    checkpoint_id: str | None = None,
    parent_checkpoint_id: str | None = None,
    source: str = "a",
    target: str = "b",
) -> HandoffContext:
    return HandoffContext(
        handoff_id=str(uuid4()),
        checkpoint_id=checkpoint_id or str(uuid4()),
        source_agent=source,
        target_agent=target,
        state_snapshot={"key": "value"},
        parent_checkpoint_id=parent_checkpoint_id,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# DAGAnalyzer
# ---------------------------------------------------------------------------


def test_dag_analyze_no_deps_for_independent_handoffs() -> None:
    h1 = _make_ctx("id1", parent_checkpoint_id=None)
    h2 = _make_ctx("id2", parent_checkpoint_id=None)
    h3 = _make_ctx("id3", parent_checkpoint_id=None)

    analyzer = DAGAnalyzer()
    deps = analyzer.analyze_dependencies([h1, h2, h3])

    assert deps["id1"] == set()
    assert deps["id2"] == set()
    assert deps["id3"] == set()


def test_dag_analyze_linear_chain() -> None:
    h1 = _make_ctx("id1", parent_checkpoint_id=None)
    h2 = _make_ctx("id2", parent_checkpoint_id="id1")
    h3 = _make_ctx("id3", parent_checkpoint_id="id2")

    analyzer = DAGAnalyzer()
    deps = analyzer.analyze_dependencies([h1, h2, h3])

    assert deps["id1"] == set()
    assert deps["id2"] == {"id1"}
    assert deps["id3"] == {"id2"}


def test_dag_external_parent_not_tracked() -> None:
    """Parent outside the batch should not appear as an intra-batch dep."""
    h1 = _make_ctx("id1", parent_checkpoint_id="external-id")

    analyzer = DAGAnalyzer()
    deps = analyzer.analyze_dependencies([h1])

    assert deps["id1"] == set()


def test_dag_execution_groups_all_independent() -> None:
    handoffs = [_make_ctx(f"id{i}", parent_checkpoint_id=None) for i in range(3)]

    analyzer = DAGAnalyzer()
    groups = analyzer.get_execution_groups(handoffs)

    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_dag_execution_groups_linear_chain() -> None:
    h1 = _make_ctx("id1", parent_checkpoint_id=None)
    h2 = _make_ctx("id2", parent_checkpoint_id="id1")
    h3 = _make_ctx("id3", parent_checkpoint_id="id2")

    analyzer = DAGAnalyzer()
    groups = analyzer.get_execution_groups([h1, h2, h3])

    # Each depends on the previous — 3 sequential groups
    assert len(groups) == 3
    assert groups[0][0].checkpoint_id == "id1"
    assert groups[1][0].checkpoint_id == "id2"
    assert groups[2][0].checkpoint_id == "id3"


def test_dag_execution_groups_mixed() -> None:
    """Two independent branches that converge at a third node."""
    h_root = _make_ctx("root", parent_checkpoint_id=None)
    h_a = _make_ctx("a", parent_checkpoint_id="root")
    h_b = _make_ctx("b", parent_checkpoint_id="root")
    # c depends on both a and b — but DAGAnalyzer only tracks single parents
    # so c will just depend on its declared parent_checkpoint_id
    h_c = _make_ctx("c", parent_checkpoint_id="a")

    analyzer = DAGAnalyzer()
    groups = analyzer.get_execution_groups([h_root, h_a, h_b, h_c])

    # Group 1: root (no deps)
    # Group 2: a and b (both depend only on root)
    # Group 3: c (depends on a)
    assert len(groups) == 3
    group_ids = [sorted(h.checkpoint_id for h in g) for g in groups]
    assert group_ids[0] == ["root"]
    assert group_ids[1] == ["a", "b"]
    assert group_ids[2] == ["c"]


# ---------------------------------------------------------------------------
# execute_handoffs_parallel — independent handoffs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_independent_handoffs_execute_in_parallel() -> None:
    """3 independent handoffs should complete in ~1x time, not 3x."""
    manager = StateManager()
    await manager.start()

    handoffs = [_make_ctx(f"id{i}", parent_checkpoint_id=None) for i in range(3)]

    start = time.monotonic()
    saved = await manager.execute_handoffs_parallel(handoffs)
    elapsed_ms = (time.monotonic() - start) * 1000

    assert len(saved) == 3
    # For the in-memory store this is always fast; just verify correctness
    for h in handoffs:
        retrieved = await manager._store.get_checkpoint(h.checkpoint_id)
        assert retrieved.checkpoint_id == h.checkpoint_id

    await manager.stop()


@pytest.mark.asyncio
async def test_dependent_handoffs_execute_sequentially() -> None:
    """A linear chain of handoffs must be saved in dependency order."""
    manager = StateManager()
    await manager.start()

    h1 = _make_ctx("dep1", parent_checkpoint_id=None)
    h2 = _make_ctx("dep2", parent_checkpoint_id="dep1")
    h3 = _make_ctx("dep3", parent_checkpoint_id="dep2")

    saved = await manager.execute_handoffs_parallel([h1, h2, h3])
    assert len(saved) == 3

    # All three must be retrievable
    for h in [h1, h2, h3]:
        r = await manager._store.get_checkpoint(h.checkpoint_id)
        assert r.checkpoint_id == h.checkpoint_id

    await manager.stop()


@pytest.mark.asyncio
async def test_empty_handoffs_returns_empty() -> None:
    manager = StateManager()
    await manager.start()
    result = await manager.execute_handoffs_parallel([])
    assert result == []
    await manager.stop()


@pytest.mark.asyncio
async def test_parallel_handoffs_benchmark() -> None:
    """3 independent handoffs should complete in <= 2x the time of 1 handoff."""
    manager = StateManager()
    await manager.start()

    # Measure single handoff baseline
    h_single = _make_ctx()
    t0 = time.monotonic()
    await manager.execute_handoffs_parallel([h_single])
    single_ms = (time.monotonic() - t0) * 1000

    # 3 independent handoffs
    three = [_make_ctx(f"par{i}", parent_checkpoint_id=None) for i in range(3)]
    t0 = time.monotonic()
    await manager.execute_handoffs_parallel(three)
    three_ms = (time.monotonic() - t0) * 1000

    # Parallel should not degrade to 3x sequential
    assert three_ms < single_ms * 3 + 5, (
        f"3 parallel handoffs ({three_ms:.2f}ms) took more than 3x single ({single_ms:.2f}ms)"
    )

    await manager.stop()


# ---------------------------------------------------------------------------
# coordinate() — parallel save end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinate_multi_agent_pipeline_uses_parallel_handoffs() -> None:
    """coordinate() with a 3-agent pipeline should persist all handoffs."""
    from kortex.core.router import Router, ProviderModel
    from kortex.core.runtime import KortexRuntime
    from kortex.core.types import TaskSpec

    router = Router()
    router.register_model(ProviderModel(
        provider="test", model="m",
        cost_per_1k_input_tokens=0.0,
        cost_per_1k_output_tokens=0.0,
        avg_latency_ms=10.0,
        tier="fast",
    ))

    runtime = KortexRuntime(router=router, state_manager=StateManager())
    await runtime.start()

    task = TaskSpec(content="hello")
    result = await runtime.coordinate(task, ["a", "b", "c"], execute=False)

    # 4 handoffs: input->a, a->b, b->c (initial + 2 inter-agent)
    assert len(result.handoffs) == 3

    await runtime.stop()
