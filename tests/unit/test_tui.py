"""Tests for the TUI dashboard module."""

from __future__ import annotations

from collections import deque

import pytest

from kortex.dashboard.tui import DashboardMetrics, KortexTUI, _sparkline


# ---------------------------------------------------------------------------
# DashboardMetrics defaults
# ---------------------------------------------------------------------------


def test_dashboard_metrics_defaults() -> None:
    m = DashboardMetrics()
    assert m.total_tasks_routed == 0
    assert m.total_cost_usd == 0.0
    assert m.avg_latency_ms == 0.0
    assert m.cache_hits == 0
    assert m.cache_misses == 0
    assert m.active_tasks == []
    assert m.provider_health == {}
    assert m.model_usage == {}
    assert len(m.recent_decisions) == 0
    assert len(m.cost_history) == 0
    assert len(m.latency_history) == 0
    assert m.paused is False
    assert m.uptime_s == 0.0
    assert m.decision_cache_hit_rate == 0.0


def test_dashboard_metrics_mutable_defaults_are_independent() -> None:
    m1 = DashboardMetrics()
    m2 = DashboardMetrics()
    m1.active_tasks.append("task-1")
    assert m2.active_tasks == []


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------


def test_sparkline_empty() -> None:
    result = _sparkline(deque())
    assert len(result) == 20  # default width
    assert result == " " * 20


def test_sparkline_uniform_values() -> None:
    values = deque([1.0, 1.0, 1.0])
    result = _sparkline(values, width=3)
    assert len(result) == 3


def test_sparkline_ascending_trend() -> None:
    values = deque([0.0, 1.0, 2.0, 3.0])
    result = _sparkline(values, width=4)
    # Should show increasing block heights
    assert result[0] <= result[-1] or all(c == result[0] for c in result)


def test_sparkline_width_respected() -> None:
    values = deque([1.0] * 100)
    result = _sparkline(values, width=10)
    assert len(result) == 10


def test_sparkline_uses_last_n_samples() -> None:
    values = deque(range(100))
    result_20 = _sparkline(values, width=20)
    assert len(result_20) == 20


# ---------------------------------------------------------------------------
# KortexTUI instantiation
# ---------------------------------------------------------------------------


def test_kortex_tui_init() -> None:
    from unittest.mock import MagicMock

    mock_runtime = MagicMock()
    mock_runtime.get_dashboard_snapshot.return_value = DashboardMetrics()

    tui = KortexTUI(mock_runtime, refresh_rate=0.5)
    assert tui._refresh_rate == 0.5
    assert tui._running is False
    assert not tui._metrics.paused


def test_kortex_tui_key_handling_quit() -> None:
    """Pressing q should set _running = False."""
    from unittest.mock import MagicMock
    import asyncio

    mock_runtime = MagicMock()
    mock_runtime.get_dashboard_snapshot.return_value = DashboardMetrics()
    tui = KortexTUI(mock_runtime)
    tui._running = True

    asyncio.run(_put_and_handle(tui, "q"))
    assert tui._running is False


def test_kortex_tui_key_handling_pause_toggle() -> None:
    from unittest.mock import MagicMock
    import asyncio

    mock_runtime = MagicMock()
    mock_runtime.get_dashboard_snapshot.return_value = DashboardMetrics()
    tui = KortexTUI(mock_runtime)
    tui._running = True

    asyncio.run(_put_and_handle(tui, "p"))
    assert tui._metrics.paused is True

    asyncio.run(_put_and_handle(tui, "p"))
    assert tui._metrics.paused is False


def test_kortex_tui_key_handling_clear_log() -> None:
    from unittest.mock import MagicMock
    import asyncio

    mock_runtime = MagicMock()
    mock_runtime.get_dashboard_snapshot.return_value = DashboardMetrics()
    tui = KortexTUI(mock_runtime)
    tui._metrics.recent_decisions.append({"agent_id": "a"})
    assert len(tui._metrics.recent_decisions) == 1

    asyncio.run(_put_and_handle(tui, "c"))
    assert len(tui._metrics.recent_decisions) == 0


# ---------------------------------------------------------------------------
# Runtime get_dashboard_snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_get_dashboard_snapshot_initial() -> None:
    from kortex.core.router import ProviderModel, Router
    from kortex.core.runtime import KortexRuntime
    from kortex.core.state import StateManager

    router = Router()
    router.register_model(ProviderModel(
        provider="test", model="fast",
        cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
        avg_latency_ms=10.0, tier="fast",
    ))
    runtime = KortexRuntime(router=router, state_manager=StateManager())
    await runtime.start()

    snapshot = runtime.get_dashboard_snapshot()
    assert isinstance(snapshot, DashboardMetrics)
    assert snapshot.total_tasks_routed == 0
    assert snapshot.total_cost_usd == 0.0

    await runtime.stop()


@pytest.mark.asyncio
async def test_runtime_get_dashboard_snapshot_after_coordinate() -> None:
    from kortex.core.router import ProviderModel, Router
    from kortex.core.runtime import KortexRuntime
    from kortex.core.state import StateManager
    from kortex.core.types import TaskSpec

    router = Router()
    router.register_model(ProviderModel(
        provider="test", model="fast",
        cost_per_1k_input_tokens=0.001, cost_per_1k_output_tokens=0.002,
        avg_latency_ms=100.0, tier="fast",
    ))
    runtime = KortexRuntime(router=router, state_manager=StateManager())
    await runtime.start()

    task = TaskSpec(content="hello dashboard")
    await runtime.coordinate(task, ["agent_a"])

    snapshot = runtime.get_dashboard_snapshot()
    assert snapshot.total_tasks_routed >= 1
    assert snapshot.total_cost_usd >= 0.0
    assert snapshot.avg_latency_ms >= 0.0
    assert len(snapshot.recent_decisions) >= 1

    await runtime.stop()


@pytest.mark.asyncio
async def test_event_stream_populated_after_coordinate() -> None:
    from kortex.core.router import ProviderModel, Router
    from kortex.core.runtime import KortexRuntime
    from kortex.core.state import StateManager
    from kortex.core.types import TaskSpec

    router = Router()
    router.register_model(ProviderModel(
        provider="test", model="fast",
        cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
        avg_latency_ms=10.0, tier="fast",
    ))
    runtime = KortexRuntime(router=router, state_manager=StateManager())
    await runtime.start()

    task = TaskSpec(content="test event stream")
    await runtime.coordinate(task, ["agent_a"])

    # At least one routing decision should be in the event stream
    assert not runtime._event_stream.empty()
    event = runtime._event_stream.get_nowait()
    assert "chosen_model" in event
    assert "chosen_provider" in event

    await runtime.stop()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _put_and_handle(tui: KortexTUI, key: str) -> None:
    await tui._key_queue.put(key)
    await tui._handle_keys()
