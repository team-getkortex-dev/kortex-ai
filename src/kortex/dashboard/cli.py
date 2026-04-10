"""Terminal CLI for Kortex.

Provides subcommands for inspecting providers, models, routing decisions,
checkpoint history, traces, replay, and policy management. Uses only
stdlib + structlog + kortex internals.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import structlog

from kortex.benchmark.harness import (
    BaselineStrategy,
    BenchmarkHarness,
    TaskDataset,
)
from kortex.security.redaction import scan_and_redact
from kortex.core.policy import RoutingPolicy
from kortex.core.replay import ReplayEngine
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.trace import TaskTrace, TraceStep
from kortex.core.trace_store import InMemoryTraceStore, SQLiteTraceStore, TraceStore
from kortex.core.types import TaskSpec
from kortex.dashboard.formatter import (
    colorize,
    format_cost,
    format_duration,
    format_table,
)
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore

logger = structlog.get_logger(component="cli")


# ---------------------------------------------------------------------------
# Demo models shown when no API keys are detected
# ---------------------------------------------------------------------------

_DEMO_MODELS: list[ProviderModel] = [
    ProviderModel(
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=200,
        capabilities=["reasoning", "content_generation"],
        tier="fast",
    ),
    ProviderModel(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        avg_latency_ms=800,
        capabilities=["reasoning", "code_generation", "content_generation"],
        tier="balanced",
    ),
    ProviderModel(
        provider="anthropic",
        model="claude-opus-4-20250514",
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.075,
        avg_latency_ms=2000,
        capabilities=["reasoning", "code_generation", "content_generation", "analysis"],
        tier="powerful",
    ),
    ProviderModel(
        provider="openai",
        model="gpt-4o",
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.015,
        avg_latency_ms=600,
        capabilities=["reasoning", "code_generation", "content_generation"],
        tier="balanced",
    ),
]

_DEFAULT_AGENTS: list[AgentDescriptor] = [
    AgentDescriptor(
        agent_id="researcher",
        name="Researcher",
        description="Gathers and analyzes information",
        capabilities=["reasoning", "content_generation"],
    ),
    AgentDescriptor(
        agent_id="writer",
        name="Writer",
        description="Drafts content and documentation",
        capabilities=["content_generation"],
    ),
    AgentDescriptor(
        agent_id="reviewer",
        name="Reviewer",
        description="Reviews and provides feedback",
        capabilities=["reasoning", "analysis"],
    ),
]


class KortexCLI:
    """Interactive CLI for Kortex operations.

    Args:
        runtime: An KortexRuntime instance.
        registry: A ProviderRegistry instance.
        demo_mode: Whether demo models are being used.
        trace_store: Optional trace store for trace/replay commands.
    """

    def __init__(
        self,
        runtime: KortexRuntime,
        registry: ProviderRegistry,
        demo_mode: bool = False,
        trace_store: TraceStore | None = None,
    ) -> None:
        self._runtime = runtime
        self._registry = registry
        self._demo_mode = demo_mode
        self._trace_store = trace_store

    def _demo_banner(self) -> str:
        """Return a demo mode notice if applicable."""
        if self._demo_mode:
            return colorize(
                "[demo mode - no API keys detected, showing example models]",
                "yellow",
            )
        return ""

    # -- status -------------------------------------------------------------

    async def cmd_status(self) -> str:
        """Show system status: providers, models, agents, state store."""
        lines: list[str] = []
        lines.append(colorize("Kortex Status", "bold"))

        banner = self._demo_banner()
        if banner:
            lines.append(banner)

        lines.append("")

        # Providers
        providers = self._registry.list_providers()
        lines.append(colorize(f"Providers ({len(providers)}):", "cyan"))
        for name in providers:
            connector = self._registry.get_provider(name)
            try:
                healthy = await connector.health_check()
            except Exception:
                healthy = False
            status = colorize("OK", "green") if healthy else colorize("FAIL", "red")
            connector_info = scan_and_redact(f"  {name}: {status}")
            lines.append(connector_info)

        # Models by tier
        all_models = self._registry.get_all_models()
        tiers: dict[str, int] = {"fast": 0, "balanced": 0, "powerful": 0}
        for m in all_models:
            tiers[m.tier] = tiers.get(m.tier, 0) + 1
        lines.append("")
        lines.append(colorize(f"Models ({len(all_models)}):", "cyan"))
        for tier, count in tiers.items():
            lines.append(f"  {tier}: {count}")

        # Agents
        agent_count = len(self._runtime._agents)
        lines.append("")
        lines.append(colorize(f"Agents: {agent_count}", "cyan"))
        for agent in self._runtime._agents.values():
            lines.append(f"  {agent.agent_id}: {agent.description}")

        # State store
        store = self._runtime._state._store
        store_type = type(store).__name__
        lines.append("")
        lines.append(colorize(f"State Store: {store_type}", "cyan"))

        return "\n".join(lines)

    # -- models -------------------------------------------------------------

    def cmd_models(self) -> str:
        """List all available models across all providers."""
        all_models = self._registry.get_all_models()

        # Sort by tier order, then by cost ascending
        tier_order = {"fast": 0, "balanced": 1, "powerful": 2}
        all_models.sort(key=lambda m: (tier_order.get(m.tier, 9), m.estimated_cost()))

        headers = ["PROVIDER", "MODEL", "TIER", "INPUT/1K", "OUTPUT/1K", "LATENCY", "CAPABILITIES"]
        rows: list[list[str]] = []
        for m in all_models:
            is_free = m.cost_per_1k_input_tokens == 0.0 and m.cost_per_1k_output_tokens == 0.0
            tag = " [local]" if is_free else ""
            rows.append([
                m.provider + tag,
                m.model,
                m.tier,
                format_cost(m.cost_per_1k_input_tokens),
                format_cost(m.cost_per_1k_output_tokens),
                format_duration(m.avg_latency_ms),
                ", ".join(m.capabilities),
            ])

        parts: list[str] = []
        banner = self._demo_banner()
        if banner:
            parts.append(banner)
            parts.append("")
        parts.append(scan_and_redact(format_table(headers, rows)))
        return "\n".join(parts)

    # -- dry-run ------------------------------------------------------------

    async def cmd_dry_run(
        self,
        task_content: str,
        complexity: str = "moderate",
        pipeline: list[str] | None = None,
    ) -> str:
        """Run a routing dry-run and display decisions."""
        if pipeline is None:
            pipeline = list(self._runtime._agents.keys())

        if not pipeline:
            return colorize("Error: No agents in pipeline.", "red")

        task = TaskSpec(
            content=task_content,
            complexity_hint=complexity,  # type: ignore[arg-type]
        )

        result = await self._runtime.coordinate(task, pipeline, execute=False)

        lines: list[str] = []
        lines.append(colorize("Dry Run Results", "bold"))

        banner = self._demo_banner()
        if banner:
            lines.append(banner)

        lines.append(f"Task: {task_content}")
        lines.append(f"Complexity: {complexity}")
        lines.append(f"Pipeline: {' -> '.join(pipeline)}")
        lines.append("")

        # Routing decisions
        lines.append(colorize("Routing Decisions:", "cyan"))
        for i, d in enumerate(result.routing_decisions, 1):
            lines.append(f"  Step {i}: {d.chosen_provider}/{d.chosen_model}")
            lines.append(f"    Reasoning: {d.reasoning}")
            lines.append(f"    Est. cost: {format_cost(d.estimated_cost_usd)}")
            lines.append(f"    Est. latency: {format_duration(d.estimated_latency_ms)}")
            if d.fallback_model:
                lines.append(f"    Fallback: {d.fallback_model}")

        # Cost summary
        lines.append("")
        lines.append(colorize("Cost Summary:", "cyan"))
        lines.append(f"  Total estimated: {format_cost(result.total_estimated_cost_usd)}")

        # Handoff chain
        lines.append("")
        lines.append(colorize("Handoff Chain:", "cyan"))
        for h in result.handoffs:
            lines.append(f"  {h.source_agent} -> {h.target_agent} [{h.checkpoint_id[:8]}...]")

        return "\n".join(lines)

    # -- history ------------------------------------------------------------

    async def cmd_history(
        self,
        agent_id: str | None = None,
        task_id: str | None = None,
        last: int | None = None,
    ) -> str:
        """Show checkpoint history."""
        store = self._runtime._state._store

        if agent_id:
            checkpoints = await store.list_checkpoints(agent_id=agent_id)
        elif task_id:
            checkpoints = await store.list_checkpoints(task_id=task_id)
        else:
            checkpoints = await store.list_checkpoints()

        # Sort by created_at descending
        checkpoints.sort(key=lambda c: c.created_at, reverse=True)

        if last is not None:
            checkpoints = checkpoints[:last]

        if not checkpoints:
            return "No checkpoints found."

        lines: list[str] = []
        lines.append(colorize(f"Checkpoints ({len(checkpoints)}):", "bold"))
        lines.append("")

        for c in checkpoints:
            ts = c.created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(c.created_at, "strftime") else str(c.created_at)
            lines.append(f"  [{ts}] {c.source_agent} -> {c.target_agent}")
            lines.append(f"    ID: {c.checkpoint_id[:12]}...")
            if c.parent_checkpoint_id:
                lines.append(f"    Parent: {c.parent_checkpoint_id[:12]}...")
            if c.compressed_summary:
                summary = c.compressed_summary
                if len(summary) > 80:
                    summary = summary[:80] + "..."
                lines.append(f"    Summary: {summary}")

        return "\n".join(lines)

    # -- config -------------------------------------------------------------

    def cmd_config(self) -> str:
        """Show current configuration."""
        lines: list[str] = []
        lines.append(colorize("Kortex Configuration", "bold"))
        lines.append("")

        # State store
        store = self._runtime._state._store
        store_type = type(store).__name__
        lines.append(f"State Store: {store_type}")

        # Providers
        providers = self._registry.list_providers()
        lines.append(f"Providers: {', '.join(providers) if providers else 'none'}")

        # Router strategy
        strategy_type = type(self._runtime._router._strategy).__name__
        lines.append(f"Routing Strategy: {strategy_type}")

        # Active policy
        policy = self._runtime._router.get_policy()
        if policy is not None:
            lines.append(f"Active Policy: {policy.name}")
        else:
            lines.append("Active Policy: none (using heuristic strategy)")

        # Log level
        log_level = os.environ.get("KORTEX_LOG_LEVEL", "INFO")
        lines.append(f"Log Level: {log_level}")

        # Env vars
        lines.append("")
        lines.append(colorize("Environment:", "cyan"))
        for key in sorted(os.environ):
            if key.startswith("KORTEX_"):
                lines.append(f"  {key}={os.environ[key]}")

        # Config file
        config_path = "kortex.toml"
        if os.path.exists(config_path):
            lines.append(f"\nConfig file: {config_path} (found)")
        else:
            lines.append(f"\nConfig file: {config_path} (not found)")

        return "\n".join(lines)

    # -- trace list ---------------------------------------------------------

    async def cmd_trace_list(
        self,
        limit: int = 20,
        task_id: str | None = None,
    ) -> str:
        """List recent traces from the trace store."""
        if self._trace_store is None:
            return colorize(
                "No trace store configured. Use KORTEX_TRACE_STORE=sqlite to enable.",
                "yellow",
            )

        traces = await self._trace_store.list_traces(limit=limit, task_id=task_id)

        if not traces:
            return "No traces found."

        headers = ["TRACE_ID", "TASK", "PIPELINE", "STEPS", "COST", "DURATION", "OK", "CREATED"]
        rows: list[list[str]] = []
        for t in traces:
            task_text = t.task_content[:50] + ("..." if len(t.task_content) > 50 else "")
            rows.append([
                t.trace_id[:12],
                task_text,
                " -> ".join(t.pipeline),
                str(len(t.steps)),
                format_cost(t.total_estimated_cost_usd),
                format_duration(t.total_duration_ms) if t.total_duration_ms > 0 else "-",
                "yes" if t.success else "NO",
                t.created_at[:19] if t.created_at else "-",
            ])

        return format_table(headers, rows)

    # -- trace show ---------------------------------------------------------

    async def cmd_trace_show(self, trace_id: str) -> str:
        """Show full detail for a specific trace."""
        if self._trace_store is None:
            return colorize(
                "No trace store configured. Use KORTEX_TRACE_STORE=sqlite to enable.",
                "yellow",
            )

        try:
            trace = await self._trace_store.get_trace(trace_id)
        except KeyError:
            return colorize(f"Error: Trace '{trace_id}' not found.", "red")

        lines: list[str] = []
        lines.append(colorize(f"Trace {trace.trace_id}", "bold"))
        lines.append(f"Task: {trace.task_content}")
        lines.append(f"Complexity: {trace.task_complexity}")
        lines.append(f"Pipeline: {' -> '.join(trace.pipeline)}")
        lines.append(f"Success: {'yes' if trace.success else 'NO'}")
        lines.append(f"Created: {trace.created_at}")
        lines.append("")

        # Steps
        lines.append(colorize(f"Steps ({len(trace.steps)}):", "cyan"))
        for step in trace.steps:
            rd = step.routing_decision
            model = rd.get("chosen_model", "?")
            provider = rd.get("chosen_provider", "?")
            cost = rd.get("estimated_cost_usd", 0.0)
            latency = rd.get("estimated_latency_ms", 0.0)
            reasoning = rd.get("reasoning", "")

            lines.append(f"  Step {step.step_index}: {colorize(step.agent_id, 'cyan')}")
            lines.append(f"    Model: {provider}/{model}")
            lines.append(f"    Est. cost: {format_cost(cost)}")
            lines.append(f"    Est. latency: {format_duration(latency)}")
            if reasoning:
                lines.append(f"    Reasoning: {reasoning}")
            if step.duration_ms > 0:
                lines.append(f"    Duration: {format_duration(step.duration_ms)}")

            if step.anomalies:
                lines.append(f"    Anomalies: {len(step.anomalies)}")
                for a in step.anomalies:
                    atype = a.get("anomaly_type", "unknown")
                    action = a.get("recommended_action", "unknown")
                    lines.append(f"      - {atype} -> {action}")

            if step.recovery_records:
                lines.append(f"    Recovery: {len(step.recovery_records)}")
                for r in step.recovery_records:
                    action = r.get("action_taken", "unknown")
                    lines.append(f"      - {action}")

        # Totals
        lines.append("")
        lines.append(colorize("Totals:", "cyan"))
        lines.append(f"  Estimated cost: {format_cost(trace.total_estimated_cost_usd)}")
        if trace.total_actual_cost_usd > 0:
            lines.append(f"  Actual cost: {format_cost(trace.total_actual_cost_usd)}")
        lines.append(f"  Duration: {format_duration(trace.total_duration_ms)}")

        return "\n".join(lines)

    # -- trace export -------------------------------------------------------

    async def cmd_trace_export(
        self, trace_id: str, output_file: str | None = None
    ) -> str:
        """Export a trace as JSON."""
        if self._trace_store is None:
            return colorize(
                "No trace store configured. Use KORTEX_TRACE_STORE=sqlite to enable.",
                "yellow",
            )

        try:
            trace = await self._trace_store.get_trace(trace_id)
        except KeyError:
            return colorize(f"Error: Trace '{trace_id}' not found.", "red")

        json_str = trace.to_json()

        if output_file:
            with open(output_file, "w") as f:
                f.write(json_str)
            return f"Trace exported to {output_file}"

        return json_str

    # -- replay -------------------------------------------------------------

    async def cmd_replay(
        self, trace_id: str, policy_file: str | None = None
    ) -> str:
        """Replay a trace under a different policy."""
        if self._trace_store is None:
            return colorize(
                "No trace store configured. Use KORTEX_TRACE_STORE=sqlite to enable.",
                "yellow",
            )

        try:
            trace = await self._trace_store.get_trace(trace_id)
        except KeyError:
            return colorize(f"Error: Trace '{trace_id}' not found.", "red")

        policy: RoutingPolicy | None = None
        if policy_file:
            try:
                policy = RoutingPolicy.from_toml(policy_file)
            except Exception as exc:
                return colorize(f"Error: Could not load policy file '{policy_file}': {exc}", "red")

        engine = ReplayEngine(self._runtime._router)
        result = await engine.replay(trace, policy=policy)

        lines: list[str] = []
        lines.append(colorize("Replay Results", "bold"))
        policy_name = result.policy_used.get("name", "original")
        lines.append(f"Policy: {policy_name}")
        lines.append("")

        # Per-step comparison
        headers = ["STEP", "AGENT", "ORIGINAL", "REPLAYED", "CHANGED", "COST DELTA"]
        rows: list[list[str]] = []
        for step in result.replayed_steps:
            orig = f"{step.original_provider}/{step.original_model}"
            replay = f"{step.replayed_provider}/{step.replayed_model}"
            changed = "YES" if step.model_changed else "-"
            delta = _format_delta(step.cost_delta)
            rows.append([
                str(step.step_index),
                step.agent_id,
                orig,
                replay,
                changed,
                delta,
            ])

        lines.append(format_table(headers, rows))
        lines.append("")
        lines.append(result.summary)

        return "\n".join(lines)

    # -- policy diff --------------------------------------------------------

    async def cmd_policy_diff(
        self, trace_id: str, policy_file: str
    ) -> str:
        """Show only steps that would change under a different policy."""
        if self._trace_store is None:
            return colorize(
                "No trace store configured. Use KORTEX_TRACE_STORE=sqlite to enable.",
                "yellow",
            )

        try:
            trace = await self._trace_store.get_trace(trace_id)
        except KeyError:
            return colorize(f"Error: Trace '{trace_id}' not found.", "red")

        try:
            policy = RoutingPolicy.from_toml(policy_file)
        except Exception as exc:
            return colorize(f"Error: Could not load policy file '{policy_file}': {exc}", "red")

        engine = ReplayEngine(self._runtime._router)
        result = await engine.policy_diff(trace, policy)

        changed_steps = [s for s in result.replayed_steps if s.model_changed]

        lines: list[str] = []
        lines.append(colorize("Policy Diff", "bold"))
        lines.append(f"Policy: {policy.name}")
        lines.append("")

        if not changed_steps:
            lines.append("No steps would change under this policy.")
        else:
            headers = ["STEP", "AGENT", "ORIGINAL", "NEW MODEL", "COST DELTA"]
            rows: list[list[str]] = []
            for step in changed_steps:
                orig = f"{step.original_provider}/{step.original_model}"
                new = f"{step.replayed_provider}/{step.replayed_model}"
                delta = _format_delta(step.cost_delta)
                rows.append([
                    str(step.step_index),
                    step.agent_id,
                    orig,
                    new,
                    delta,
                ])
            lines.append(format_table(headers, rows))

        lines.append("")
        lines.append(result.summary)

        return "\n".join(lines)

    # -- policy show --------------------------------------------------------

    def cmd_policy_show(self, policy_file: str | None = None) -> str:
        """Show a policy's details."""
        if policy_file:
            try:
                policy = RoutingPolicy.from_toml(policy_file)
            except Exception as exc:
                return colorize(f"Error: Could not load policy file '{policy_file}': {exc}", "red")
        else:
            policy = self._runtime._router.get_policy()
            if policy is None:
                policy = RoutingPolicy()

        lines: list[str] = []
        lines.append(colorize(f"Policy: {policy.name}", "bold"))
        if policy.description:
            lines.append(f"Description: {policy.description}")
        lines.append("")

        # Constraints
        lines.append(colorize("Constraints:", "cyan"))
        c = policy.constraints
        if c.max_cost_usd is not None:
            lines.append(f"  Max cost/request: {format_cost(c.max_cost_usd)}")
        else:
            lines.append("  Max cost/request: none")
        if c.max_latency_ms is not None:
            lines.append(f"  Max latency: {format_duration(c.max_latency_ms)}")
        else:
            lines.append("  Max latency: none")
        if c.required_capabilities:
            lines.append(f"  Required capabilities: {', '.join(c.required_capabilities)}")
        if c.allowed_providers is not None:
            lines.append(f"  Allowed providers: {', '.join(c.allowed_providers)}")
        if c.denied_providers:
            lines.append(f"  Denied providers: {', '.join(c.denied_providers)}")
        if c.allowed_models is not None:
            lines.append(f"  Allowed models: {', '.join(c.allowed_models)}")
        if c.denied_models:
            lines.append(f"  Denied models: {', '.join(c.denied_models)}")

        # Objective
        lines.append("")
        lines.append(colorize("Objective:", "cyan"))
        o = policy.objective
        lines.append(f"  Minimize: {o.minimize}")
        lines.append(f"  Prefer tier: {o.prefer_tier}")
        if o.prefer_provider:
            lines.append(f"  Prefer provider: {o.prefer_provider}")

        # Fallback
        lines.append("")
        lines.append(colorize("Fallback:", "cyan"))
        f = policy.fallback
        lines.append(f"  Strategy: {f.strategy}")
        if f.explicit_model_identity:
            lines.append(f"  Explicit model: {f.explicit_model_identity}")

        # Budget
        if policy.budget_ceiling_usd is not None:
            lines.append("")
            lines.append(f"Budget ceiling: {format_cost(policy.budget_ceiling_usd)}")

        return "\n".join(lines)

    # -- policy from-text ---------------------------------------------------

    async def cmd_policy_from_text(
        self,
        description: str,
        output_file: str | None = None,
        model: str | None = None,
    ) -> str:
        """Compile a RoutingPolicy from a natural-language description."""
        from kortex.core.nl_policy import NLPolicyCompiler

        providers = self._registry.list_providers()
        if not providers:
            return colorize(
                "Error: No provider configured. "
                "Set an API key (e.g. ANTHROPIC_API_KEY) to use NL policy compilation.",
                "red",
            )

        connector = self._registry.get_provider(providers[0])
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        compiler = NLPolicyCompiler(connector, **kwargs)

        try:
            policy = await compiler.compile(description)
        except ValueError as exc:
            return colorize(f"Error: {exc}", "red")

        lines: list[str] = [
            colorize(f"Compiled policy: {policy.name}", "bold"),
            f"Description: {policy.description}",
            "",
        ]

        # Render the compiled policy details via JSON
        import json as _json
        lines.append(_json.dumps(policy.to_dict(), indent=2))

        if output_file:
            try:
                policy.to_toml(output_file)
                lines.append(colorize(f"\nPolicy saved to: {output_file}", "green"))
            except Exception as exc:
                lines.append(colorize(f"\nWarning: Could not save policy: {exc}", "yellow"))

        return "\n".join(lines)

    async def cmd_policy_interactive(self) -> str:
        """Interactively build a policy from natural language (stdin)."""
        from kortex.core.nl_policy import NLPolicyCompiler

        providers = self._registry.list_providers()
        if not providers:
            return colorize(
                "Error: No provider configured. "
                "Set an API key to use interactive policy compilation.",
                "red",
            )

        lines: list[str] = [
            colorize("Interactive Policy Builder", "bold"),
            "Describe your routing requirements in plain English.",
            "(empty line = done, Ctrl-C = cancel)",
            "",
        ]

        description_parts: list[str] = []
        try:
            while True:
                line = input("> ").strip()
                if not line:
                    break
                description_parts.append(line)
        except (KeyboardInterrupt, EOFError):
            return colorize("\nCancelled.", "yellow")

        if not description_parts:
            return colorize("No description provided.", "yellow")

        description = " ".join(description_parts)
        connector = self._registry.get_provider(providers[0])
        compiler = NLPolicyCompiler(connector)

        try:
            policy = await compiler.compile(description)
        except ValueError as exc:
            return colorize(f"Error: {exc}", "red")

        import json as _json
        lines.append(colorize(f"Compiled policy: {policy.name}", "bold"))
        lines.append(_json.dumps(policy.to_dict(), indent=2))
        return "\n".join(lines)

    # -- benchmark run ------------------------------------------------------

    async def cmd_benchmark_run(
        self,
        dataset_name: str = "all",
        output_file: str | None = None,
    ) -> str:
        """Run the full benchmark with current models.

        Args:
            dataset_name: Which dataset(s) to use: mixed, cost_sensitive,
                latency_sensitive, or all.
            output_file: Optional JSON output path.

        Returns:
            Formatted benchmark results.
        """
        all_models = self._registry.get_all_models()
        if not all_models:
            return colorize("Error: No models available for benchmarking.", "red")

        harness = BenchmarkHarness(all_models)

        ds_map = {
            "mixed": TaskDataset.mixed_workload,
            "cost_sensitive": TaskDataset.cost_sensitive,
            "latency_sensitive": TaskDataset.latency_sensitive,
        }

        if dataset_name == "all":
            datasets = [fn() for fn in ds_map.values()]
        elif dataset_name in ds_map:
            datasets = [ds_map[dataset_name]()]
        else:
            return colorize(
                f"Error: Unknown dataset '{dataset_name}'. "
                "Use: mixed, cost_sensitive, latency_sensitive, all",
                "red",
            )

        report = await harness.full_benchmark(datasets=datasets)

        if output_file:
            import json as _json

            with open(output_file, "w") as f:
                _json.dump(report.to_dict(), f, indent=2)

        return report.to_markdown()

    # -- benchmark compare --------------------------------------------------

    async def cmd_benchmark_compare(
        self,
        policy_file: str,
        baseline: str = "strongest",
    ) -> str:
        """Run a single benchmark comparison.

        Args:
            policy_file: Path to a TOML policy file.
            baseline: Baseline strategy name (cheapest, strongest, random).

        Returns:
            Formatted comparison results.
        """
        all_models = self._registry.get_all_models()
        if not all_models:
            return colorize("Error: No models available for benchmarking.", "red")

        try:
            policy = RoutingPolicy.from_toml(policy_file)
        except Exception as exc:
            return colorize(
                f"Error: Could not load policy file '{policy_file}': {exc}",
                "red",
            )

        strategy_map = {
            "cheapest": BaselineStrategy.ALWAYS_CHEAPEST,
            "strongest": BaselineStrategy.ALWAYS_STRONGEST,
            "random": BaselineStrategy.RANDOM,
        }
        strategy = strategy_map.get(baseline)
        if strategy is None:
            return colorize(
                f"Error: Unknown baseline '{baseline}'. "
                "Use: cheapest, strongest, random",
                "red",
            )

        harness = BenchmarkHarness(all_models)
        dataset = TaskDataset.mixed_workload()
        comparison = await harness.compare(dataset, policy, strategy)

        lines: list[str] = []
        lines.append(colorize("Benchmark Comparison", "bold"))
        lines.append(f"Policy: {policy.name}")
        lines.append(f"Baseline: {baseline}")
        lines.append(f"Dataset: {dataset.name} ({len(dataset.tasks)} tasks)")
        lines.append("")

        # Side-by-side
        headers = ["METRIC", "BASELINE", "KORTEX", "DELTA"]
        rows: list[list[str]] = [
            [
                "Total cost",
                format_cost(comparison.baseline.total_estimated_cost_usd),
                format_cost(comparison.kortex.total_estimated_cost_usd),
                f"{comparison.cost_delta_pct:+.1f}%",
            ],
            [
                "Avg cost/task",
                format_cost(comparison.baseline.avg_cost_per_task),
                format_cost(comparison.kortex.avg_cost_per_task),
                "",
            ],
            [
                "Total latency",
                format_duration(comparison.baseline.total_estimated_latency_ms),
                format_duration(comparison.kortex.total_estimated_latency_ms),
                f"{comparison.latency_delta_pct:+.1f}%",
            ],
            [
                "Cap. mismatches",
                str(comparison.baseline.capability_mismatches),
                str(comparison.kortex.capability_mismatches),
                f"{comparison.capability_match_improvement:+d}",
            ],
            [
                "Routing failures",
                str(comparison.baseline.routing_failures),
                str(comparison.kortex.routing_failures),
                "",
            ],
        ]
        lines.append(format_table(headers, rows))
        lines.append("")
        lines.append(comparison.summary)

        return "\n".join(lines)


    # -- arbitrage ----------------------------------------------------------

    async def cmd_arbitrage(
        self,
        model: str,
        excluded_providers: list[str] | None = None,
    ) -> str:
        """Show cost arbitrage opportunities for a model."""
        from kortex.router.cost_arbitrage import CostArbitrage

        all_models = self._registry.get_all_models()
        if not all_models:
            return colorize("Error: No models registered.", "red")

        arbitrage = CostArbitrage()

        # Auto-register all models by their name as equivalents to each other
        model_names: list[str] = list({m.model for m in all_models})
        if model not in model_names:
            return colorize(
                f"Error: Model '{model}' not found in registry. "
                f"Available: {', '.join(model_names[:5])}",
                "red",
            )

        # Register prices from registered models
        for pm in all_models:
            arbitrage.update_price(
                pm.provider, pm.model,
                input_per_1k=pm.cost_per_1k_input_tokens,
                output_per_1k=pm.cost_per_1k_output_tokens,
            )

        # Register all models with same name across providers as equivalents
        by_name: dict[str, list[str]] = {}
        for pm in all_models:
            by_name.setdefault(pm.model, []).append(pm.provider)

        # Also register demo equivalents
        known_equivalents = [
            ("gpt-4o-mini", "claude-haiku-4-5"),
            ("gpt-4o", "claude-sonnet-4-20250514"),
        ]
        for m1, m2 in known_equivalents:
            if m1 in model_names and m2 in model_names:
                arbitrage.register_equivalent_models(m1, m2)

        # For the requested model, find cheapest equivalent (if any)
        decision = arbitrage.find_cheapest(
            model, excluded_providers=excluded_providers or []
        )

        lines: list[str] = [colorize(f"Cost Arbitrage for '{model}'", "bold"), ""]

        # Show all registered prices for this model and equivalents
        headers = ["PROVIDER", "MODEL", "IN/1K", "OUT/1K", "EST COST"]
        rows = []
        for m in all_models:
            if m.model == model:
                rows.append([
                    m.provider,
                    m.model,
                    format_cost(m.cost_per_1k_input_tokens),
                    format_cost(m.cost_per_1k_output_tokens),
                    format_cost(m.estimated_cost()),
                ])

        if rows:
            lines.append(format_table(headers, rows))

        if decision:
            lines.append("")
            lines.append(colorize("Arbitrage Result:", "cyan"))
            lines.append(f"  {decision.reason}")
            lines.append(
                f"  Savings: {format_cost(decision.savings_usd)} "
                f"({decision.savings_usd / decision.original_cost * 100:.1f}%)"
                if decision.original_cost > 0
                else "  No savings (already optimal)"
            )
        else:
            lines.append("")
            lines.append(colorize(
                f"No equivalent models found for '{model}'. "
                "Register equivalents with CostArbitrage.register_equivalent_models().",
                "yellow",
            ))

        return "\n".join(lines)

    # -- experiment ---------------------------------------------------------

    async def cmd_experiment_run(self, parsed: Any) -> str:
        """Run an A/B experiment over trace store data."""
        from kortex.core.ab_testing import ABTest, ExperimentConfig
        from kortex.core.replay import ReplayEngine

        if self._trace_store is None:
            return colorize("Error: No trace store configured.", "red")

        policy_a = None
        if parsed.policy_a:
            try:
                policy_a = RoutingPolicy.from_toml(parsed.policy_a)
            except Exception as exc:
                return colorize(f"Error loading policy A: {exc}", "red")
        else:
            policy_a = RoutingPolicy.cost_optimized()

        try:
            policy_b = RoutingPolicy.from_toml(parsed.policy_b)
        except Exception as exc:
            return colorize(f"Error loading policy B: {exc}", "red")

        traces = await self._runtime.list_traces(limit=parsed.limit)
        if not traces:
            return colorize("No traces available. Run some tasks first.", "yellow")

        config = ExperimentConfig(
            name=f"experiment_{policy_a.name}_vs_{policy_b.name}",
            control_policy=policy_a,
            treatment_policy=policy_b,
            traffic_split=parsed.traffic_split,
            min_samples=parsed.min_samples,
            metric=parsed.metric,
            auto_promote=False,  # CLI is informational
        )
        experiment = ABTest(config, rng_seed=42)
        engine = ReplayEngine(self._runtime._router)

        # Replay each trace under the assigned policy arm
        for trace in traces:
            arm_policy = experiment.split_traffic()
            try:
                result = await engine.replay(trace, policy=arm_policy)
                cost = sum(s.replayed_estimated_cost for s in result.replayed_steps)
                latency = sum(
                    s.replayed_estimated_cost * 1000
                    for s in result.replayed_steps
                )  # approximate
                experiment.record_result(arm_policy.name, cost=cost, latency_ms=latency)
            except Exception:
                pass

        exp_result = experiment.get_result()
        lines: list[str] = [
            colorize("A/B Experiment Results", "bold"),
            "",
            exp_result.summary(),
        ]
        if experiment.should_promote():
            lines.append("")
            lines.append(colorize(
                f"RECOMMENDATION: Promote '{policy_b.name}' — "
                f"significant {parsed.metric} improvement detected.",
                "green",
            ))
        return "\n".join(lines)

    # -- debug (time-travel) -----------------------------------------------

    async def cmd_debug_show(self, trace_id: str) -> str:
        """Show a step-by-step walkthrough of a trace."""
        from kortex.core.time_machine import TimeMachine

        if self._trace_store is None:
            return colorize("Error: No trace store configured.", "red")
        try:
            trace = await self._trace_store.get_trace(trace_id)
        except KeyError:
            return colorize(f"Error: Trace '{trace_id}' not found.", "red")

        tm = TimeMachine(trace)
        return tm.full_summary()

    async def cmd_debug_replay_from(
        self,
        trace_id: str,
        step: int,
        policy_file: str | None = None,
    ) -> str:
        """Replay a trace from a specific step."""
        from kortex.core.replay import ReplayEngine
        from kortex.core.time_machine import TimeMachine

        if self._trace_store is None:
            return colorize("Error: No trace store configured.", "red")
        try:
            trace = await self._trace_store.get_trace(trace_id)
        except KeyError:
            return colorize(f"Error: Trace '{trace_id}' not found.", "red")

        policy = None
        if policy_file:
            try:
                policy = RoutingPolicy.from_toml(policy_file)
            except Exception as exc:
                return colorize(f"Error loading policy: {exc}", "red")

        engine = ReplayEngine(self._runtime._router)
        try:
            result = await engine.replay_from_step(trace, from_step=step, policy=policy)
        except IndexError as exc:
            return colorize(f"Error: {exc}", "red")

        lines: list[str] = []
        lines.append(colorize(f"Replay from step {step} of trace {trace_id[:16]}...", "bold"))
        lines.append(result.summary)
        lines.append("")

        headers = ["STEP", "AGENT", "ORIGINAL", "REPLAYED", "CHANGED", "DELTA"]
        rows: list[list[str]] = []
        for s in result.replayed_steps:
            rows.append([
                str(s.step_index),
                s.agent_id,
                f"{s.original_provider}::{s.original_model}",
                f"{s.replayed_provider}::{s.replayed_model}",
                colorize("YES", "yellow") if s.model_changed else "no",
                f"{s.cost_delta:+.5f}",
            ])
        lines.append(format_table(headers, rows))
        return "\n".join(lines)

    async def cmd_debug_diff(
        self,
        trace_id: str,
        policy_a_file: str | None,
        policy_b_file: str,
    ) -> str:
        """Diff two policy replays of the same trace."""
        from kortex.core.replay import ReplayEngine

        if self._trace_store is None:
            return colorize("Error: No trace store configured.", "red")
        try:
            trace = await self._trace_store.get_trace(trace_id)
        except KeyError:
            return colorize(f"Error: Trace '{trace_id}' not found.", "red")

        policy_a: RoutingPolicy | None = None
        if policy_a_file:
            try:
                policy_a = RoutingPolicy.from_toml(policy_a_file)
            except Exception as exc:
                return colorize(f"Error loading policy A: {exc}", "red")

        try:
            policy_b = RoutingPolicy.from_toml(policy_b_file)
        except Exception as exc:
            return colorize(f"Error loading policy B: {exc}", "red")

        engine = ReplayEngine(self._runtime._router)
        result_a = await engine.replay(trace, policy=policy_a)
        result_b = await engine.replay(trace, policy=policy_b)
        diff = result_a.diff(result_b)

        lines: list[str] = []
        name_a = diff["policy_self"]
        name_b = diff["policy_other"]
        lines.append(colorize(f"Diff: '{name_a}' vs '{name_b}'", "bold"))
        lines.append(
            f"Cost A: ${diff['total_cost_self']:.5f}  "
            f"Cost B: ${diff['total_cost_other']:.5f}  "
            f"Delta: {diff['cost_delta']:+.5f}"
        )
        lines.append(f"Changed steps: {diff['changed_steps']}")
        lines.append("")

        headers = ["STEP", "MODEL A", "MODEL B", "CHANGED", "COST DELTA"]
        rows = []
        for sd in diff["step_diffs"]:
            rows.append([
                str(sd["step_index"]),
                sd["self_model"] or "—",
                sd["other_model"] or "—",
                colorize("YES", "yellow") if sd["model_differs"] else "no",
                f"{sd['cost_delta']:+.5f}",
            ])
        lines.append(format_table(headers, rows))
        return "\n".join(lines)


def _format_delta(delta: float) -> str:
    """Format a cost delta with sign."""
    if delta == 0:
        return "-"
    sign = "+" if delta > 0 else ""
    return f"{sign}{format_cost(delta)}" if delta >= 0 else f"-{format_cost(abs(delta))}"


def _build_demo_trace(router: Router) -> TaskTrace:
    """Build a demo trace so trace commands have data in demo mode."""
    policy = RoutingPolicy.cost_optimized()
    policy_dict = policy.to_dict()

    steps: list[TraceStep] = []
    agents = ["researcher", "writer", "reviewer"]
    models_cycle = [
        ("openai", "gpt-4o-mini", 0.0004, 200),
        ("anthropic", "claude-sonnet-4-20250514", 0.0105, 800),
        ("openai", "gpt-4o-mini", 0.0004, 200),
    ]

    for i, (agent_id, (prov, model, cost, latency)) in enumerate(zip(agents, models_cycle)):
        steps.append(TraceStep(
            step_index=i,
            agent_id=agent_id,
            input_payload={"content": "Write about AI coordination", "task_id": "demo-task"},
            routing_decision={
                "task_id": "demo-task",
                "chosen_provider": prov,
                "chosen_model": model,
                "estimated_cost_usd": cost,
                "estimated_latency_ms": latency,
                "reasoning": f"Cost-optimized: {model} selected for {agent_id}.",
            },
            policy_snapshot=policy_dict,
            started_at="2026-03-29T10:00:00+00:00",
            completed_at="2026-03-29T10:00:01+00:00",
            duration_ms=float(latency),
        ))

    total_cost = sum(m[2] for m in models_cycle)
    return TaskTrace(
        trace_id="demo-trace-001",
        task_id="demo-task",
        task_content="Write about AI coordination",
        task_complexity="moderate",
        pipeline=agents,
        steps=steps,
        policy_snapshot=policy_dict,
        total_estimated_cost_usd=total_cost,
        total_actual_cost_usd=0.0,
        total_duration_ms=1200.0,
        success=True,
        created_at="2026-03-29T10:00:00+00:00",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="kortex",
        description="Kortex CLI - Agent coordination runtime",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("status", help="Show system status")
    sub.add_parser("models", help="List available models")
    sub.add_parser("config", help="Show configuration")

    dr = sub.add_parser("dry-run", help="Run a routing dry-run")
    dr.add_argument("--task", required=True, help="Task description")
    dr.add_argument("--complexity", default="moderate", choices=["simple", "moderate", "complex"])
    dr.add_argument("--pipeline", default=None, help="Comma-separated agent IDs")

    hist = sub.add_parser("history", help="Show checkpoint history")
    hist.add_argument("--agent", default=None, help="Filter by agent ID")
    hist.add_argument("--task", dest="task_id", default=None, help="Filter by task ID")
    hist.add_argument("--last", type=int, default=None, help="Show N most recent checkpoints")

    # Trace commands
    trace_parser = sub.add_parser("trace", help="Trace management")
    trace_sub = trace_parser.add_subparsers(dest="trace_command", help="Trace subcommands")

    trace_list = trace_sub.add_parser("list", help="List recent traces")
    trace_list.add_argument("--limit", type=int, default=20, help="Max traces to show")
    trace_list.add_argument("--task-id", default=None, help="Filter by task ID")

    trace_show = trace_sub.add_parser("show", help="Show full trace detail")
    trace_show.add_argument("trace_id", help="Trace ID to show")

    trace_export = trace_sub.add_parser("export", help="Export trace as JSON")
    trace_export.add_argument("trace_id", help="Trace ID to export")
    trace_export.add_argument("--output", default=None, help="Output file path")

    # Replay command
    replay_parser = sub.add_parser("replay", help="Replay a trace under a different policy")
    replay_parser.add_argument("trace_id", help="Trace ID to replay")
    replay_parser.add_argument("--policy", default=None, help="TOML policy file")

    # Policy commands
    policy_parser = sub.add_parser("policy", help="Policy management")
    policy_sub = policy_parser.add_subparsers(dest="policy_command", help="Policy subcommands")

    policy_diff = policy_sub.add_parser("diff", help="Compare trace against a different policy")
    policy_diff.add_argument("trace_id", help="Trace ID to diff")
    policy_diff.add_argument("--policy", required=True, help="TOML policy file")

    policy_show = policy_sub.add_parser("show", help="Show policy details")
    policy_show.add_argument("--file", default=None, help="TOML policy file")

    policy_from_text = policy_sub.add_parser(
        "from-text", help="Compile a RoutingPolicy from natural language",
    )
    policy_from_text.add_argument("description", help="Natural language policy description")
    policy_from_text.add_argument(
        "--output", default=None, help="Save compiled policy to this TOML file",
    )
    policy_from_text.add_argument(
        "--model", default=None, help="Model to use for compilation",
    )

    policy_sub.add_parser(
        "interactive", help="Interactively build a policy from natural language",
    )

    # Debug (time-travel) command
    debug_parser = sub.add_parser(
        "debug", help="Time-travel debugging — inspect and replay trace steps",
    )
    debug_sub = debug_parser.add_subparsers(dest="debug_command", help="Debug subcommands")

    debug_show = debug_sub.add_parser("show", help="Show step-by-step trace walkthrough")
    debug_show.add_argument("trace_id", help="Trace ID to inspect")

    debug_from = debug_sub.add_parser("replay-from", help="Replay from a specific step")
    debug_from.add_argument("trace_id", help="Trace ID")
    debug_from.add_argument("--step", type=int, required=True, help="Step index to resume from")
    debug_from.add_argument("--policy", default=None, help="TOML policy file for replay")

    debug_diff = debug_sub.add_parser("diff", help="Diff two policy replays of the same trace")
    debug_diff.add_argument("trace_id", help="Trace ID")
    debug_diff.add_argument("--policy-a", default=None, help="TOML file for policy A")
    debug_diff.add_argument("--policy-b", required=True, help="TOML file for policy B")

    # Benchmark commands
    bench_parser = sub.add_parser("benchmark", help="Run routing benchmarks")
    bench_sub = bench_parser.add_subparsers(dest="bench_command", help="Benchmark subcommands")

    bench_run = bench_sub.add_parser("run", help="Run full benchmark suite")
    bench_run.add_argument(
        "--dataset", default="all",
        choices=["mixed", "cost_sensitive", "latency_sensitive", "all"],
        help="Dataset to benchmark",
    )
    bench_run.add_argument("--output", default=None, help="Export results to JSON file")

    bench_cmp = bench_sub.add_parser("compare", help="Compare policy vs baseline")
    bench_cmp.add_argument("--policy", required=True, help="TOML policy file")
    bench_cmp.add_argument(
        "--baseline", default="strongest",
        choices=["cheapest", "strongest", "random"],
        help="Baseline strategy",
    )

    # Experiment command
    exp_parser = sub.add_parser(
        "experiment", help="Manage A/B experiments for policy comparison",
    )
    exp_sub = exp_parser.add_subparsers(dest="exp_command", help="Experiment subcommands")

    exp_run = exp_sub.add_parser("run", help="Run an A/B experiment over traces in the store")
    exp_run.add_argument("--policy-a", default=None, help="TOML file for policy A (control)")
    exp_run.add_argument("--policy-b", required=True, help="TOML file for policy B (treatment)")
    exp_run.add_argument("--traffic-split", type=float, default=0.5, help="Fraction to treatment")
    exp_run.add_argument("--min-samples", type=int, default=10, help="Min samples per arm")
    exp_run.add_argument("--limit", type=int, default=100, help="Max traces to replay")
    exp_run.add_argument("--metric", default="cost", choices=["cost", "latency"])

    # Arbitrage command
    arb_parser = sub.add_parser(
        "arbitrage", help="Show provider cost arbitrage opportunities",
    )
    arb_parser.add_argument(
        "--model", required=True,
        help="Model name to find cheapest equivalent for",
    )
    arb_parser.add_argument(
        "--exclude", default=None,
        help="Comma-separated providers to exclude",
    )

    # Optimize command
    opt_parser = sub.add_parser(
        "optimize", help="Auto-optimize routing policy via grid search + Pareto frontier",
    )
    opt_parser.add_argument(
        "--limit", type=int, default=50,
        help="Max traces to use from the store (default: 50)",
    )
    opt_parser.add_argument(
        "--output", default=None,
        help="Write best-balanced policy to this TOML file",
    )

    # Dashboard command
    dash_parser = sub.add_parser(
        "dashboard", help="Launch interactive TUI dashboard",
    )
    dash_parser.add_argument(
        "--refresh", type=float, default=1.0,
        help="Refresh rate in seconds (default: 1.0)",
    )

    # Stream command
    stream_parser = sub.add_parser(
        "stream", help="Stream task execution through a pipeline in real time",
    )
    stream_parser.add_argument("--task", required=True, help="Task description")
    stream_parser.add_argument(
        "--complexity", default="moderate",
        choices=["simple", "moderate", "complex"],
    )
    stream_parser.add_argument(
        "--pipeline", default=None,
        help="Comma-separated agent IDs (default: researcher,writer)",
    )

    # Coordinate-batch command
    batch_parser = sub.add_parser(
        "coordinate-batch",
        help="Coordinate a batch of tasks from a JSON file",
    )
    batch_parser.add_argument(
        "--tasks-file", required=True,
        help="Path to JSON file containing an array of task objects",
    )
    batch_parser.add_argument(
        "--pipeline", default=None,
        help="Comma-separated agent IDs applied to every task (default: researcher,writer)",
    )
    batch_parser.add_argument(
        "--execute", action="store_true", default=False,
        help="Execute tasks against live providers (default: dry-run)",
    )

    # Estimate command
    estimate_parser = sub.add_parser(
        "estimate",
        help="Estimate cost for a batch of tasks without executing",
    )
    estimate_parser.add_argument(
        "--tasks-file", required=True,
        help="Path to JSON file containing an array of task objects",
    )
    estimate_parser.add_argument(
        "--pipeline", default=None,
        help="Comma-separated agent IDs applied to every task (default: researcher,writer)",
    )
    estimate_parser.add_argument(
        "--warn-threshold", type=float, default=None,
        help="Warn if estimated total cost exceeds this amount in USD",
    )

    gen_parser = sub.add_parser(
        "generate-tests",
        help="Auto-generate pytest regression tests from production traces",
    )
    gen_parser.add_argument(
        "--traces-db", default=None,
        help="Path to SQLite trace database (default: kortex_traces.db if exists, else in-memory)",
    )
    gen_parser.add_argument(
        "--output", default="tests/generated/test_routing_regressions.py",
        help="Output path for generated test file",
    )
    gen_parser.add_argument(
        "--sample-rate", type=float, default=1.0,
        help="Fraction of traces to include (0–1, default: 1.0)",
    )
    gen_parser.add_argument(
        "--min-cost", type=float, default=0.0,
        help="Minimum trace cost (USD) to include (default: 0.0)",
    )
    gen_parser.add_argument(
        "--cost-tolerance", type=float, default=0.20,
        help="Allowed cost deviation fraction (default: 0.20)",
    )
    gen_parser.add_argument(
        "--latency-tolerance", type=float, default=0.50,
        help="Allowed latency deviation fraction (default: 0.50)",
    )
    gen_parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of traces to load (default: all)",
    )

    return parser


def _build_default_runtime() -> tuple[KortexRuntime, ProviderRegistry, bool]:
    """Build a runtime from environment configuration.

    Returns:
        Tuple of (runtime, registry, demo_mode).
        The trace store is accessible via ``runtime._trace_store``.
    """
    registry = ProviderRegistry()
    registry.auto_discover()

    demo_mode = len(registry.list_providers()) == 0

    router = Router()

    if demo_mode:
        # No API keys found -- register demo models so CLI output is useful
        # Group demo models by provider and register as OpenAI-compatible
        by_provider: dict[str, list[ProviderModel]] = {}
        for model in _DEMO_MODELS:
            by_provider.setdefault(model.provider, []).append(model)
        for provider_name, models in by_provider.items():
            registry.register_openai_compatible(
                name=provider_name,
                base_url=f"https://{provider_name}.example.com/v1",
                api_key=None,
                models=models,
            )
        for model in _DEMO_MODELS:
            router.register_model(model)
    else:
        for model in registry.get_all_models():
            router.register_model(model)

    backend = os.environ.get("KORTEX_STATE_BACKEND", "memory")
    state_manager = StateManager.create(backend=backend)  # type: ignore[arg-type]

    # Trace store
    trace_store_type = os.environ.get("KORTEX_TRACE_STORE", "memory" if demo_mode else "")
    trace_store: TraceStore | None = None
    if trace_store_type == "memory":
        trace_store = InMemoryTraceStore()
    elif trace_store_type == "sqlite":
        db_path = os.environ.get("KORTEX_TRACE_DB", "kortex_traces.db")
        trace_store = SQLiteTraceStore(db_path)

    runtime = KortexRuntime(
        router=router,
        state_manager=state_manager,
        registry=registry,
        trace_store=trace_store,
    )

    # Register default agents so dry-run works out of the box
    if not runtime._agents:
        for agent in _DEFAULT_AGENTS:
            runtime.register_agent(agent)

    return runtime, registry, demo_mode


async def _run_command(parsed: argparse.Namespace) -> str:
    """Execute a CLI command inside an async context with proper lifecycle."""
    runtime, registry, demo_mode = _build_default_runtime()
    trace_store = runtime._trace_store
    cli = KortexCLI(runtime, registry, demo_mode=demo_mode, trace_store=trace_store)

    # In demo mode, seed the trace store with a demo trace
    if demo_mode and trace_store is not None:
        demo_trace = _build_demo_trace(runtime._router)
        await trace_store.save_trace(demo_trace)

    async with runtime:
        if parsed.command == "status":
            return await cli.cmd_status()
        elif parsed.command == "models":
            return cli.cmd_models()
        elif parsed.command == "config":
            return cli.cmd_config()
        elif parsed.command == "dry-run":
            pipeline = parsed.pipeline.split(",") if parsed.pipeline else None
            return await cli.cmd_dry_run(
                task_content=parsed.task,
                complexity=parsed.complexity,
                pipeline=pipeline,
            )
        elif parsed.command == "history":
            return await cli.cmd_history(
                agent_id=parsed.agent,
                task_id=parsed.task_id,
                last=parsed.last,
            )
        elif parsed.command == "trace":
            tc = parsed.trace_command
            if tc == "list":
                return await cli.cmd_trace_list(
                    limit=parsed.limit,
                    task_id=parsed.task_id,
                )
            elif tc == "show":
                return await cli.cmd_trace_show(parsed.trace_id)
            elif tc == "export":
                return await cli.cmd_trace_export(
                    parsed.trace_id, output_file=parsed.output,
                )
            else:
                return colorize("Error: Unknown trace subcommand. Use: list, show, export", "red")
        elif parsed.command == "replay":
            return await cli.cmd_replay(
                parsed.trace_id, policy_file=parsed.policy,
            )
        elif parsed.command == "policy":
            pc = parsed.policy_command
            if pc == "diff":
                return await cli.cmd_policy_diff(
                    parsed.trace_id, policy_file=parsed.policy,
                )
            elif pc == "show":
                return cli.cmd_policy_show(policy_file=parsed.file)
            elif pc == "from-text":
                return await cli.cmd_policy_from_text(
                    parsed.description,
                    output_file=parsed.output,
                    model=parsed.model,
                )
            elif pc == "interactive":
                return await cli.cmd_policy_interactive()
            else:
                return colorize(
                    "Error: Unknown policy subcommand. Use: diff, show, from-text, interactive",
                    "red",
                )
        elif parsed.command == "debug":
            dc = parsed.debug_command
            if dc == "show":
                return await cli.cmd_debug_show(parsed.trace_id)
            elif dc == "replay-from":
                return await cli.cmd_debug_replay_from(
                    parsed.trace_id, step=parsed.step, policy_file=parsed.policy,
                )
            elif dc == "diff":
                return await cli.cmd_debug_diff(
                    parsed.trace_id,
                    policy_a_file=parsed.policy_a,
                    policy_b_file=parsed.policy_b,
                )
            else:
                return colorize("Error: Unknown debug subcommand. Use: show, replay-from, diff", "red")
        elif parsed.command == "benchmark":
            bc = parsed.bench_command
            if bc == "run":
                return await cli.cmd_benchmark_run(
                    dataset_name=parsed.dataset,
                    output_file=parsed.output,
                )
            elif bc == "compare":
                return await cli.cmd_benchmark_compare(
                    policy_file=parsed.policy,
                    baseline=parsed.baseline,
                )
            else:
                return colorize("Error: Unknown benchmark subcommand. Use: run, compare", "red")
        elif parsed.command == "arbitrage":
            excluded = [p.strip() for p in parsed.exclude.split(",")] if parsed.exclude else []
            return await cli.cmd_arbitrage(parsed.model, excluded_providers=excluded)
        elif parsed.command == "experiment":
            ec = parsed.exp_command
            if ec == "run":
                return await cli.cmd_experiment_run(parsed)
            else:
                return colorize("Error: Unknown experiment subcommand. Use: run", "red")
        else:
            raise ValueError(f"Unknown command: {parsed.command}")


async def _run_stream_command(parsed: argparse.Namespace) -> None:
    """Execute the stream command, printing tokens to stdout as they arrive."""
    runtime, registry, demo_mode = _build_default_runtime()
    pipeline_str = parsed.pipeline or "researcher,writer"
    pipeline = [p.strip() for p in pipeline_str.split(",")]

    task = TaskSpec(
        content=parsed.task,
        complexity_hint=parsed.complexity,
    )

    async with runtime:
        print(colorize(f"Streaming pipeline: {' → '.join(pipeline)}", "cyan"))
        print(colorize(f"Task: {parsed.task[:80]}", "bold"))
        if demo_mode:
            print(colorize("[demo mode — no live provider; showing routing events only]", "yellow"))
        print()

        async for event_type, data in runtime.stream_coordinate(task, pipeline):
            if event_type == "routing_decision":
                print(
                    colorize(
                        f"[step {data['step']}] {data['agent_id']} → "
                        f"{data['chosen_provider']}::{data['chosen_model']} "
                        f"(~${data['estimated_cost_usd']:.4f})",
                        "cyan",
                    )
                )
            elif event_type == "handoff":
                if data["step"] >= 0:
                    print(
                        colorize(
                            f"  ✓ handoff {data['source']} → {data['target']} "
                            f"[{data['checkpoint_id'][:8]}...]",
                            "green",
                        )
                    )
            elif event_type == "token":
                print(data["token"], end="", flush=True)
            elif event_type == "error":
                print(
                    colorize(f"\n[error step {data['step']}] {data['error']}", "red"),
                )
            elif event_type == "completion":
                print()
                print()
                print(
                    colorize(
                        f"Completed: {data['agents_routed']}/{data['total_agents']} agents "
                        f"in {data['duration_ms']:.0f}ms",
                        "green" if data["success"] else "red",
                    )
                )


async def _run_batch_command(parsed: argparse.Namespace) -> None:
    """Execute the coordinate-batch command."""
    with open(parsed.tasks_file) as f:
        task_data = json.load(f)

    pipeline_str = parsed.pipeline or "researcher,writer"
    pipeline = [p.strip() for p in pipeline_str.split(",")]

    tasks = [
        TaskSpec(
            content=t.get("content", ""),
            complexity_hint=t.get("complexity_hint", "moderate"),
            task_id=t.get("task_id"),
        )
        for t in task_data
    ]
    pipelines = [pipeline] * len(tasks)

    runtime, _, demo_mode = _build_default_runtime()
    if demo_mode:
        print(colorize("[demo mode — routing only, no live providers]", "yellow"))

    async with runtime:
        results = await runtime.coordinate_batch(tasks, pipelines, execute=parsed.execute)

    print(colorize(f"Completed {len(results)} task(s):", "bold"))
    for i, result in enumerate(results):
        task_id = result.task_id
        models = " → ".join(d.chosen_model for d in result.routing_decisions)
        cost = result.total_estimated_cost_usd
        status = colorize("OK", "green") if result.success else colorize("FAIL", "red")
        print(f"  [{i}] {status} {task_id}  models: {models}  est. cost: ${cost:.4f}")


async def _run_estimate_command(parsed: argparse.Namespace) -> None:
    """Execute the estimate command."""
    with open(parsed.tasks_file) as f:
        task_data = json.load(f)

    pipeline_str = parsed.pipeline or "researcher,writer"
    pipeline = [p.strip() for p in pipeline_str.split(",")]

    tasks = [
        TaskSpec(
            content=t.get("content", ""),
            complexity_hint=t.get("complexity_hint", "moderate"),
            task_id=t.get("task_id"),
        )
        for t in task_data
    ]
    pipelines = [pipeline] * len(tasks)

    runtime, _, demo_mode = _build_default_runtime()
    if demo_mode:
        print(colorize("[demo mode — using demo model costs]", "yellow"))

    async with runtime:
        estimate = await runtime.estimate_cost(tasks, pipelines)

    print(colorize("Cost Estimate", "bold"))
    print(estimate.summary())

    if parsed.warn_threshold is not None and estimate.total_usd > parsed.warn_threshold:
        print(
            colorize(
                f"\nWARNING: estimated total ${estimate.total_usd:.4f} exceeds "
                f"threshold ${parsed.warn_threshold:.4f}",
                "red",
            )
        )

    print()
    print(colorize("Per-task breakdown:", "cyan"))
    for i, (task, cost) in enumerate(zip(tasks, estimate.per_task)):
        print(f"  [{i}] {task.task_id}: ${cost:.4f}")


async def _run_optimize_command(parsed: argparse.Namespace) -> str:
    """Run the auto-optimization playground."""
    from kortex.core.optimization import OptimizationPlayground

    runtime, registry, demo_mode = _build_default_runtime()

    if runtime._trace_store is None:
        return colorize(
            "Error: No trace store configured. "
            "Set KORTEX_TRACE_STORE=memory or KORTEX_TRACE_STORE=sqlite.",
            "red",
        )

    async with runtime:
        traces = await runtime.list_traces(limit=parsed.limit)
        if not traces:
            if demo_mode:
                demo_trace = _build_demo_trace(runtime._router)
                traces = [demo_trace]
            else:
                return colorize("No traces available. Run some tasks first.", "yellow")

        playground = OptimizationPlayground(runtime._router)
        result = await playground.optimize(traces)

        lines: list[str] = [colorize("Optimization Results", "bold"), ""]
        lines.append(result.summary())
        lines.append("")
        lines.append(colorize("Pareto Frontier:", "cyan"))

        headers = ["POLICY", "AVG COST", "AVG LATENCY", "PARETO"]
        rows = []
        for e in result.pareto_frontier:
            rows.append([
                e.policy_name,
                format_cost(e.avg_cost_usd),
                format_duration(e.avg_latency_ms),
                colorize("YES", "green"),
            ])
        if rows:
            lines.append(format_table(headers, rows))

        if result.best_balanced and parsed.output:
            policy = RoutingPolicy.from_dict(result.best_balanced.policy_dict)
            try:
                policy.to_toml(parsed.output)
                lines.append(colorize(
                    f"\nBest-balanced policy written to: {parsed.output}", "green",
                ))
            except Exception as exc:
                lines.append(colorize(f"\nWarning: Could not write policy: {exc}", "yellow"))

        return "\n".join(lines)


async def _run_generate_tests_command(parsed: argparse.Namespace) -> str:
    """Generate pytest regression tests from production traces."""
    from kortex.testing.trace_to_test import ConversionConfig, TraceToTestConverter

    # Determine trace store
    traces_db = parsed.traces_db
    if traces_db is None and os.path.exists("kortex_traces.db"):
        traces_db = "kortex_traces.db"

    if traces_db:
        store: TraceStore = SQLiteTraceStore(traces_db)
    else:
        store = InMemoryTraceStore()

    traces = await store.list_traces(limit=parsed.limit)

    if not traces:
        return colorize(
            "No traces found. Run some tasks first, or specify --traces-db.",
            "yellow",
        )

    cfg = ConversionConfig(
        cost_tolerance_pct=parsed.cost_tolerance,
        latency_tolerance_pct=parsed.latency_tolerance,
        min_cost_threshold=parsed.min_cost,
        sample_rate=parsed.sample_rate,
    )
    converter = TraceToTestConverter(cfg)
    code = converter.generate_tests(traces, output_path=parsed.output)

    lines = [
        colorize("Test generation complete", "green"),
        f"  Traces loaded : {len(traces)}",
        f"  Tests written : {code.count('async def test_')}",
        f"  Output file   : {parsed.output}",
    ]
    return "\n".join(lines)


async def _run_dashboard_command(parsed: argparse.Namespace) -> None:
    """Launch the interactive TUI dashboard."""
    from kortex.dashboard.tui import KortexTUI

    runtime, registry, demo_mode = _build_default_runtime()

    if demo_mode:
        print(colorize(
            "[demo mode — no live providers; routing telemetry from dry-run pipeline]",
            "yellow",
        ))

    async with runtime:
        await KortexTUI.start(runtime, refresh_rate=parsed.refresh)


def main(args: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        args: Command-line arguments. Uses sys.argv if None.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = _build_parser()
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 1

    try:
        if parsed.command == "stream":
            asyncio.run(_run_stream_command(parsed))
        elif parsed.command == "coordinate-batch":
            asyncio.run(_run_batch_command(parsed))
        elif parsed.command == "estimate":
            asyncio.run(_run_estimate_command(parsed))
        elif parsed.command == "generate-tests":
            output = asyncio.run(_run_generate_tests_command(parsed))
            print(output)
        elif parsed.command == "optimize":
            output = asyncio.run(_run_optimize_command(parsed))
            print(output)
        elif parsed.command == "dashboard":
            asyncio.run(_run_dashboard_command(parsed))
        else:
            output = asyncio.run(_run_command(parsed))
            print(output)
        return 0

    except Exception as exc:
        print(colorize(f"Error: {exc}", "red"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
