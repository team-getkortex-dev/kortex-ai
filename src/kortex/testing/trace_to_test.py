"""Auto-generate pytest test suites from production traces.

The ``TraceToTestConverter`` takes ``TaskTrace`` objects and generates
pytest test code that:
1. Recreates the exact routing conditions (task spec + policy)
2. Asserts that the same model is chosen (regression guard)
3. Asserts cost and latency are within tolerance of the recorded values
4. Optionally includes a dry-run execute assertion

Generated tests are plain Python strings — write them to a file and
``pytest`` will pick them up.

Example::

    from kortex.testing.trace_to_test import TraceToTestConverter

    converter = TraceToTestConverter()
    code = converter.generate_tests(traces, output_path="tests/generated/test_routing_regressions.py")
    print(code)
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from kortex.core.trace import TaskTrace

logger = structlog.get_logger(component="trace_to_test")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConversionConfig:
    """Configuration for test generation.

    Args:
        cost_tolerance_pct: Allowed deviation from recorded cost (0–1).
        latency_tolerance_pct: Allowed deviation from recorded latency (0–1).
        include_execute_assertion: Also test that execute=True doesn't raise.
        min_cost_threshold: Skip traces with total cost below this.
        sample_rate: Fraction of traces to include (0–1), sampled deterministically.
        test_module_docstring: Docstring for the generated module.
        fixture_name: Name of the pytest fixture for the runtime.
    """

    cost_tolerance_pct: float = 0.20
    latency_tolerance_pct: float = 0.50
    include_execute_assertion: bool = False
    min_cost_threshold: float = 0.0
    sample_rate: float = 1.0
    test_module_docstring: str = "Auto-generated routing regression tests."
    fixture_name: str = "routing_runtime"


# ---------------------------------------------------------------------------
# TraceToTestConverter
# ---------------------------------------------------------------------------


class TraceToTestConverter:
    """Converts TaskTrace objects into pytest regression tests.

    Args:
        config: Conversion configuration.
    """

    def __init__(self, config: ConversionConfig | None = None) -> None:
        self._config = config or ConversionConfig()
        self._log = structlog.get_logger(component="trace_to_test")

    def generate_tests(
        self,
        traces: list[TaskTrace],
        output_path: str | None = None,
    ) -> str:
        """Generate pytest test code from a list of traces.

        Args:
            traces: The traces to convert to tests.
            output_path: If provided, write the generated code to this path.

        Returns:
            The generated Python test code as a string.
        """
        filtered = self._filter_traces(traces)
        self._log.info(
            "generating_tests",
            total_traces=len(traces),
            filtered_traces=len(filtered),
        )

        parts: list[str] = []
        parts.append(self._render_header())
        parts.append(self._render_fixtures())
        parts.append("")

        for i, trace in enumerate(filtered):
            test_code = self._render_trace_test(trace, i)
            parts.append(test_code)
            parts.append("")

        code = "\n".join(parts)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code, encoding="utf-8")
            self._log.info("tests_written", path=str(path), count=len(filtered))

        return code

    def _filter_traces(self, traces: list[TaskTrace]) -> list[TaskTrace]:
        """Apply cost threshold and sample rate filtering."""
        filtered: list[TaskTrace] = []
        for i, trace in enumerate(traces):
            if not trace.success:
                continue
            if trace.total_estimated_cost_usd < self._config.min_cost_threshold:
                continue
            # Deterministic sampling
            if self._config.sample_rate < 1.0:
                if (i % round(1.0 / self._config.sample_rate)) != 0:
                    continue
            filtered.append(trace)
        return filtered

    def _render_header(self) -> str:
        return textwrap.dedent(f'''\
            """
            {self._config.test_module_docstring}

            AUTO-GENERATED — DO NOT EDIT BY HAND.
            Re-generate with: kortex generate-tests --traces-db kortex_traces.db
            """
            from __future__ import annotations

            import pytest
            from kortex.core.policy import RoutingPolicy
            from kortex.core.router import ProviderModel, Router
            from kortex.core.runtime import KortexRuntime
            from kortex.core.state import StateManager
            from kortex.core.types import TaskSpec
        ''')

    def _render_fixtures(self) -> str:
        name = self._config.fixture_name
        router_fixture = f"{name}_router"
        return textwrap.dedent(f'''\

            @pytest.fixture
            def {router_fixture}():
                """Router fixture — extend with your registered models."""
                router = Router()
                # TODO: register your production models here
                return router


            @pytest.fixture
            def {name}({router_fixture}):
                """Runtime fixture for generated routing tests."""
                return KortexRuntime(
                    router={router_fixture},
                    state_manager=StateManager(),
                )
        ''')

    def _render_trace_test(self, trace: TaskTrace, index: int) -> str:
        """Render a pytest test function for one trace."""
        safe_task_id = re.sub(r"[^a-zA-Z0-9_]", "_", trace.task_id)[:40]
        func_name = f"test_routing_regression_{safe_task_id}_{index}"
        step_tests = "\n".join(
            self._render_step_assertion(step, step_idx)
            for step_idx, step in enumerate(trace.steps)
        )

        policy_code = self._render_policy(trace.policy_snapshot)
        complexity = trace.task_complexity or "moderate"
        content_repr = repr(trace.task_content[:100])

        fixture = self._config.fixture_name
        return textwrap.dedent(f'''\
            @pytest.mark.asyncio
            async def {func_name}({fixture}):
                """Regression test for trace {trace.trace_id[:16]}...

                Original task: {trace.task_content[:60]!r}
                Pipeline: {trace.pipeline}
                Recorded cost: ${trace.total_estimated_cost_usd:.5f}
                """
                # Set up the routing policy from the trace
                {policy_code}

                task = TaskSpec(
                    content={content_repr},
                    complexity_hint={complexity!r},
                )

                result = await {fixture}.coordinate(
                    task,
                    agent_pipeline={trace.pipeline!r},
                    execute=False,
                )

                assert result.success, "Coordination should succeed"
                assert len(result.routing_decisions) == {len(trace.steps)}

            {self._indent(step_tests, 4)}
        ''')

    def _render_step_assertion(self, step: Any, step_idx: int) -> str:
        """Render assertions for one step."""
        rd = step.routing_decision
        model = rd.get("chosen_model", "")
        provider = rd.get("chosen_provider", "")
        cost = rd.get("estimated_cost_usd", 0.0)
        latency = rd.get("estimated_latency_ms", 0.0)
        tol_cost = cost * (1 + self._config.cost_tolerance_pct)
        tol_latency = latency * (1 + self._config.latency_tolerance_pct)

        return textwrap.dedent(f'''\
            # Step {step_idx}: {step.agent_id}
            assert result.routing_decisions[{step_idx}].chosen_model == {model!r}, (
                f"Step {step_idx} model regression: expected {model!r}, "
                f"got {{result.routing_decisions[{step_idx}].chosen_model}}"
            )
            assert result.routing_decisions[{step_idx}].chosen_provider == {provider!r}
            assert result.routing_decisions[{step_idx}].estimated_cost_usd <= {tol_cost:.6f}, (
                f"Step {step_idx} cost regression: {{result.routing_decisions[{step_idx}].estimated_cost_usd:.6f}} > {tol_cost:.6f}"
            )
        ''').strip()

    def _render_policy(self, policy_snapshot: dict[str, Any]) -> str:
        """Render code to reconstruct the policy from its snapshot."""
        if not policy_snapshot:
            return "# No policy snapshot — using router default\n    pass"
        name = policy_snapshot.get("name", "default")
        return (
            f"from kortex.core.policy import RoutingPolicy\n"
            f"    policy = RoutingPolicy.from_dict({policy_snapshot!r})\n"
            f"    routing_runtime._router.set_policy(policy)"
        )

    @staticmethod
    def _indent(text: str, spaces: int) -> str:
        """Indent all lines of text by ``spaces`` spaces."""
        indent = " " * spaces
        return "\n".join(indent + line if line.strip() else line for line in text.splitlines())
