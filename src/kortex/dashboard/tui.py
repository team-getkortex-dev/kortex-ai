"""Interactive TUI dashboard for Kortex runtime monitoring.

Real-time terminal UI built with Rich that shows:
- Active task pipeline progress
- Provider health status
- Routing decisions log stream
- Cache hit/miss metrics
- Running cost tracker with sparklines
- Keyboard controls for pause/clear/refresh

Usage::

    from kortex.dashboard.tui import KortexTUI

    async with runtime:
        await KortexTUI.start(runtime)
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kortex.core.runtime import KortexRuntime


# ---------------------------------------------------------------------------
# DashboardMetrics — live snapshot passed to/from runtime
# ---------------------------------------------------------------------------


@dataclass
class DashboardMetrics:
    """Snapshot of runtime metrics for the TUI.

    All fields are updated atomically via ``KortexRuntime.get_dashboard_snapshot()``.

    Args:
        total_tasks_routed: Cumulative number of routing decisions made.
        total_cost_usd: Cumulative estimated cost in USD.
        avg_latency_ms: Rolling average latency per routing decision.
        cache_hits: Semantic cache hits since startup.
        cache_misses: Semantic cache misses since startup.
        active_tasks: Task IDs currently being coordinated.
        provider_health: Mapping of provider name → healthy bool.
        model_usage: Mapping of model key → number of times chosen.
        recent_decisions: Last 50 routing decision dicts for the log panel.
        cost_history: Last 60 per-request cost samples (sparkline data).
        latency_history: Last 60 per-request latency samples (sparkline data).
        paused: Whether the TUI is paused (no further updates rendered).
        uptime_s: Seconds since the TUI was started.
        decision_cache_hit_rate: Router decision cache hit rate (0–1).
    """

    total_tasks_routed: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    active_tasks: list[str] = field(default_factory=list)
    provider_health: dict[str, bool] = field(default_factory=dict)
    model_usage: dict[str, int] = field(default_factory=dict)
    recent_decisions: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=50)
    )
    cost_history: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    latency_history: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    paused: bool = False
    uptime_s: float = 0.0
    decision_cache_hit_rate: float = 0.0
    # Per-model percentile latencies from the rolling window (keyed by model_key)
    model_latency_p95: dict[str, float] = field(default_factory=dict)
    model_latency_p99: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sparkline helper
# ---------------------------------------------------------------------------

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: deque[float], width: int = 20) -> str:
    """Render a mini sparkline string from a deque of floats.

    Args:
        values: Recent samples (oldest first).
        width: Number of characters to render.

    Returns:
        A string of block characters representing the trend.
    """
    if not values:
        return " " * width
    tail = list(values)[-width:]
    lo, hi = min(tail), max(tail)
    span = hi - lo or 1.0
    bars = [_SPARK_CHARS[int((v - lo) / span * (len(_SPARK_CHARS) - 1))] for v in tail]
    # pad left if shorter than width
    padded = [" "] * (width - len(bars)) + bars
    return "".join(padded)


# ---------------------------------------------------------------------------
# KortexTUI
# ---------------------------------------------------------------------------


class KortexTUI:
    """Interactive Rich TUI for live Kortex runtime monitoring.

    Keyboard controls (case-insensitive):
    - ``q`` — quit the dashboard
    - ``p`` — pause/resume live updates
    - ``c`` — clear the log stream
    - ``r`` — force an immediate refresh
    - ``1``–``9`` — jump focus to panel N (reserved, no-op currently)

    Args:
        runtime: The KortexRuntime to observe.
        refresh_rate: Seconds between automatic screen refreshes.
        history_len: Number of sparkline samples to retain.
    """

    def __init__(
        self,
        runtime: KortexRuntime,
        refresh_rate: float = 1.0,
        history_len: int = 60,
    ) -> None:
        self._runtime = runtime
        self._refresh_rate = refresh_rate
        self._history_len = history_len
        self._metrics = DashboardMetrics(
            cost_history=deque(maxlen=history_len),
            latency_history=deque(maxlen=history_len),
        )
        self._running = False
        self._start_time = 0.0
        self._key_queue: asyncio.Queue[str] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @classmethod
    async def start(
        cls,
        runtime: KortexRuntime,
        refresh_rate: float = 1.0,
    ) -> None:
        """Launch the TUI and block until the user quits.

        Args:
            runtime: The KortexRuntime to observe.
            refresh_rate: Seconds between automatic refreshes.
        """
        tui = cls(runtime, refresh_rate=refresh_rate)
        await tui.run()

    async def run(self) -> None:
        """Run the TUI event loop until the user presses ``q``."""
        try:
            from rich.console import Console
            from rich.layout import Layout
            from rich.live import Live
        except ImportError as exc:
            raise ImportError(
                "Rich is required for the TUI dashboard. "
                "Install it with: pip install 'kortex-ai[tui]' or pip install rich>=13.7.0"
            ) from exc

        self._running = True
        self._start_time = time.monotonic()

        console = Console()

        # Keyboard listener runs in a background thread (cross-platform)
        stop_event = threading.Event()
        key_thread = threading.Thread(
            target=self._key_listener,
            args=(stop_event,),
            daemon=True,
        )
        key_thread.start()

        layout = self._build_layout()

        try:
            with Live(
                layout,
                console=console,
                refresh_per_second=max(1, int(1.0 / self._refresh_rate)),
                screen=True,
            ) as live:
                while self._running:
                    # Process queued keypresses
                    await self._handle_keys()

                    if not self._metrics.paused:
                        # Pull fresh snapshot from runtime
                        self._sync_from_runtime()

                    self._metrics.uptime_s = time.monotonic() - self._start_time
                    self._update_layout(layout)
                    live.refresh()

                    await asyncio.sleep(self._refresh_rate)
        finally:
            stop_event.set()
            key_thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> Any:
        from rich.layout import Layout

        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="active_tasks", size=8),
            Layout(name="provider_health"),
        )
        layout["right"].split_column(
            Layout(name="log_stream"),
            Layout(name="cost_tracker", size=10),
        )
        return layout

    def _update_layout(self, layout: Any) -> None:
        layout["header"].update(self._render_header())
        layout["active_tasks"].update(self._render_active_tasks())
        layout["provider_health"].update(self._render_provider_health())
        layout["log_stream"].update(self._render_log_stream())
        layout["cost_tracker"].update(self._render_cost_tracker())
        layout["footer"].update(self._render_footer())

    # ------------------------------------------------------------------
    # Panel renderers
    # ------------------------------------------------------------------

    def _render_header(self) -> Any:
        from rich.panel import Panel
        from rich.text import Text

        m = self._metrics
        status = "[yellow]PAUSED[/yellow]" if m.paused else "[green]LIVE[/green]"
        uptime = _format_uptime(m.uptime_s)
        text = Text.from_markup(
            f"[bold cyan]Kortex[/bold cyan] Dashboard  •  {status}  •  "
            f"Uptime: {uptime}  •  "
            f"Tasks: [bold]{m.total_tasks_routed}[/bold]  •  "
            f"Cost: [bold green]${m.total_cost_usd:.4f}[/bold green]  •  "
            f"Cache: [cyan]{m.cache_hits}H[/cyan]/[dim]{m.cache_misses}M[/dim]"
        )
        return Panel(text, style="bold")

    def _render_active_tasks(self) -> Any:
        from rich.panel import Panel
        from rich.text import Text

        m = self._metrics
        if not m.active_tasks:
            body = Text("[dim]No active tasks[/dim]", justify="center")
        else:
            lines = [Text.from_markup(f"[cyan]▶[/cyan] {tid}") for tid in m.active_tasks[-5:]]
            body = Text("\n").join(lines)
        return Panel(body, title="Active Tasks", border_style="blue")

    def _render_provider_health(self) -> Any:
        from rich.panel import Panel
        from rich.table import Table

        m = self._metrics
        table = Table(show_header=True, header_style="bold", box=None, expand=True)
        table.add_column("Provider")
        table.add_column("Status", justify="center")
        table.add_column("Avg ms", justify="right")
        table.add_column("P95 ms", justify="right")
        table.add_column("P99 ms", justify="right")
        table.add_column("Calls", justify="right")

        for provider, healthy in sorted(m.provider_health.items()):
            status_str = "[green]● OK[/green]" if healthy else "[red]● DOWN[/red]"
            calls = sum(
                count for key, count in m.model_usage.items()
                if key.startswith(f"{provider}::")
            )
            # Aggregate P95/P99 across all models for this provider
            provider_keys = [k for k in m.model_latency_p99 if k.startswith(f"{provider}::")]
            if provider_keys:
                p95_val = max(m.model_latency_p95.get(k, 0.0) for k in provider_keys)
                p99_val = max(m.model_latency_p99.get(k, 0.0) for k in provider_keys)
            else:
                p95_val = p99_val = 0.0

            # Avg from model_usage-weighted latency (best-effort via dashboard history)
            avg_val = m.avg_latency_ms if calls > 0 else 0.0

            # Color-code P99: green < 500ms, yellow < 1000ms, red >= 1000ms
            if p99_val == 0.0:
                p99_str = "[dim]—[/dim]"
                p95_str = "[dim]—[/dim]"
                avg_str = f"{avg_val:.0f}" if avg_val > 0 else "[dim]—[/dim]"
            elif p99_val < 500:
                p99_str = f"[green]{p99_val:.0f}[/green]"
                p95_str = f"[green]{p95_val:.0f}[/green]"
                avg_str = f"{avg_val:.0f}"
            elif p99_val < 1000:
                p99_str = f"[yellow]{p99_val:.0f}[/yellow]"
                p95_str = f"[yellow]{p95_val:.0f}[/yellow]"
                avg_str = f"{avg_val:.0f}"
            else:
                p99_str = f"[red]{p99_val:.0f}[/red]"
                p95_str = f"[yellow]{p95_val:.0f}[/yellow]"
                avg_str = f"{avg_val:.0f}"

            table.add_row(provider, status_str, avg_str, p95_str, p99_str, str(calls))

        if not m.provider_health:
            table.add_row(
                "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]",
                "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]",
            )

        return Panel(table, title="Provider Health (avg / P95 / P99)", border_style="green")

    def _render_log_stream(self) -> Any:
        from rich.panel import Panel
        from rich.text import Text

        m = self._metrics
        decisions = list(m.recent_decisions)[-20:]  # last 20 for display
        lines: list[Text] = []
        for d in reversed(decisions):
            model = d.get("chosen_model", "?")
            provider = d.get("chosen_provider", "?")
            cost = d.get("estimated_cost_usd", 0.0)
            agent = d.get("agent_id", "?")
            latency = d.get("estimated_latency_ms", 0.0)
            lines.append(Text.from_markup(
                f"[dim]{agent}[/dim] → [cyan]{provider}::{model}[/cyan]  "
                f"[green]${cost:.4f}[/green]  [yellow]{latency:.0f}ms[/yellow]"
            ))

        if not lines:
            body = Text("[dim]Waiting for routing decisions…[/dim]", justify="center")
        else:
            body = Text("\n").join(lines)

        return Panel(body, title="Routing Decisions", border_style="cyan")

    def _render_cost_tracker(self) -> Any:
        from rich.panel import Panel
        from rich.text import Text

        m = self._metrics

        cost_spark = _sparkline(m.cost_history, width=30)
        lat_spark = _sparkline(m.latency_history, width=30)

        avg_cost = (
            m.total_cost_usd / m.total_tasks_routed if m.total_tasks_routed > 0 else 0.0
        )
        lines = [
            Text.from_markup(
                f"Total cost:   [bold green]${m.total_cost_usd:.4f}[/bold green]  "
                f"avg ${avg_cost:.5f}/req"
            ),
            Text.from_markup(f"Cost trend:   [cyan]{cost_spark}[/cyan]"),
            Text.from_markup(
                f"Avg latency:  [bold yellow]{m.avg_latency_ms:.0f}ms[/bold yellow]"
            ),
            Text.from_markup(f"Latency trend:[yellow]{lat_spark}[/yellow]"),
            Text.from_markup(
                f"Cache rate:   [cyan]{m.decision_cache_hit_rate*100:.1f}%[/cyan] hit  "
                f"[dim]{m.total_tasks_routed} total[/dim]"
            ),
        ]
        body = Text("\n").join(lines)
        return Panel(body, title="Cost & Latency", border_style="magenta")

    def _render_footer(self) -> Any:
        from rich.text import Text

        return Text.from_markup(
            "[dim]  q[/dim] quit  "
            "[dim]p[/dim] pause  "
            "[dim]c[/dim] clear log  "
            "[dim]r[/dim] refresh  "
            "[dim]1-9[/dim] focus panel"
        )

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def _key_listener(self, stop_event: threading.Event) -> None:
        """Read single keypresses in a background thread and enqueue them."""
        try:
            import msvcrt  # Windows

            while not stop_event.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    try:
                        key = ch.decode("utf-8").lower()
                    except UnicodeDecodeError:
                        key = ""
                    if key:
                        # Thread-safe put via run_coroutine_threadsafe is complex;
                        # use a simple list and drain in async context
                        asyncio.run_coroutine_threadsafe(
                            self._key_queue.put(key),
                            asyncio.get_event_loop(),
                        )
                time.sleep(0.05)
        except Exception:
            # Unix fallback or unavailable — ignore
            try:
                import sys
                import tty
                import termios

                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    while not stop_event.is_set():
                        import select

                        r, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if r:
                            ch = sys.stdin.read(1).lower()
                            asyncio.run_coroutine_threadsafe(
                                self._key_queue.put(ch),
                                asyncio.get_event_loop(),
                            )
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    async def _handle_keys(self) -> None:
        """Drain the key queue and apply keyboard commands."""
        while not self._key_queue.empty():
            key = self._key_queue.get_nowait()
            if key == "q":
                self._running = False
            elif key == "p":
                self._metrics.paused = not self._metrics.paused
            elif key == "c":
                self._metrics.recent_decisions.clear()
            elif key == "r":
                self._sync_from_runtime()
            # 1-9: reserved for panel focus (no-op)

    # ------------------------------------------------------------------
    # Runtime sync
    # ------------------------------------------------------------------

    def _sync_from_runtime(self) -> None:
        """Pull a fresh DashboardMetrics snapshot from the runtime."""
        try:
            snapshot = self._runtime.get_dashboard_snapshot()
        except Exception:
            return

        # Merge snapshot into our local metrics (preserving history deques)
        self._metrics.total_tasks_routed = snapshot.total_tasks_routed
        self._metrics.total_cost_usd = snapshot.total_cost_usd
        self._metrics.avg_latency_ms = snapshot.avg_latency_ms
        self._metrics.cache_hits = snapshot.cache_hits
        self._metrics.cache_misses = snapshot.cache_misses
        self._metrics.active_tasks = list(snapshot.active_tasks)
        self._metrics.provider_health = dict(snapshot.provider_health)
        self._metrics.model_usage = dict(snapshot.model_usage)
        self._metrics.decision_cache_hit_rate = snapshot.decision_cache_hit_rate
        self._metrics.model_latency_p95 = dict(snapshot.model_latency_p95)
        self._metrics.model_latency_p99 = dict(snapshot.model_latency_p99)

        # Append new history points
        for v in snapshot.cost_history:
            self._metrics.cost_history.append(v)
        for v in snapshot.latency_history:
            self._metrics.latency_history.append(v)

        # Append new log entries (avoid duplicates by checking the queue)
        existing_len = len(self._metrics.recent_decisions)
        new_entries = list(snapshot.recent_decisions)
        for entry in new_entries[existing_len:]:
            self._metrics.recent_decisions.append(entry)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_uptime(seconds: float) -> str:
    s = int(seconds)
    h, remainder = divmod(s, 3600)
    m, sec = divmod(remainder, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"
