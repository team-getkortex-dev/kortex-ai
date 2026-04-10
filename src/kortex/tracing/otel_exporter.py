"""OpenTelemetry exporter for Kortex TaskTrace objects.

Converts a ``TaskTrace`` into an OTEL span tree:

- One root span per ``TaskTrace`` (named ``kortex.coordination``)
- One child span per ``TraceStep`` (named ``kortex.step.<agent_id>``)
- Attributes mirror the trace fields (task_id, model, cost, latency, etc.)

Installation
------------
This module requires the ``opentelemetry`` extras::

    pip install kortex-ai[otel]

Usage
-----
::

    from kortex.tracing import OTELExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    exporter = OTELExporter(tracer_provider=provider)
    await exporter.export_trace(task_trace)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kortex.core.trace import TaskTrace


def _require_otel() -> Any:
    """Import opentelemetry.trace or raise a clear InstallationError."""
    try:
        import opentelemetry.trace as otel_trace  # type: ignore[import]
        return otel_trace
    except ImportError as exc:
        raise ImportError(
            "OpenTelemetry is not installed. "
            "Install it with: pip install kortex-ai[otel]"
        ) from exc


class OTELExporter:
    """Exports Kortex TaskTrace objects as OpenTelemetry spans.

    Each coordination becomes a root span named ``kortex.coordination``
    with one child span per pipeline step (``kortex.step.<agent_id>``).

    Args:
        tracer_provider: An OTEL ``TracerProvider``. When ``None``, the
            globally-configured provider is used (``opentelemetry.trace.get_tracer_provider()``).
        service_name: The OTEL service name attribute. Defaults to ``"kortex"``.

    Raises:
        ImportError: If the ``opentelemetry`` packages are not installed.
    """

    _TRACER_NAME = "kortex.runtime"

    def __init__(
        self,
        tracer_provider: Any | None = None,
        service_name: str = "kortex",
    ) -> None:
        otel_trace = _require_otel()
        self._otel_trace = otel_trace
        self._service_name = service_name

        if tracer_provider is not None:
            self._tracer = tracer_provider.get_tracer(self._TRACER_NAME)
        else:
            self._tracer = otel_trace.get_tracer(self._TRACER_NAME)

    def export_trace(self, task_trace: TaskTrace) -> None:
        """Export a TaskTrace as an OTEL span tree.

        Creates a root span for the coordination and one child span per
        pipeline step. All span timings are derived from the trace's
        ``total_duration_ms`` and per-step ``duration_ms`` fields.

        Args:
            task_trace: The TaskTrace to export.
        """
        StatusCode = self._otel_trace.StatusCode

        with self._tracer.start_as_current_span("kortex.coordination") as root_span:
            # Root span attributes
            root_span.set_attribute("kortex.task_id", task_trace.task_id)
            root_span.set_attribute("kortex.service", self._service_name)
            root_span.set_attribute("kortex.task_complexity", task_trace.task_complexity)
            root_span.set_attribute("kortex.pipeline", ",".join(task_trace.pipeline))
            root_span.set_attribute(
                "kortex.total_estimated_cost_usd",
                task_trace.total_estimated_cost_usd,
            )
            root_span.set_attribute(
                "kortex.total_actual_cost_usd",
                task_trace.total_actual_cost_usd,
            )
            root_span.set_attribute("kortex.total_duration_ms", task_trace.total_duration_ms)
            root_span.set_attribute("kortex.success", task_trace.success)
            root_span.set_attribute("kortex.step_count", len(task_trace.steps))

            if task_trace.policy_snapshot:
                policy_name = task_trace.policy_snapshot.get("name", "")
                if policy_name:
                    root_span.set_attribute("kortex.policy_name", policy_name)

            if not task_trace.success:
                root_span.set_status(StatusCode.ERROR, "Coordination failed")
            else:
                root_span.set_status(StatusCode.OK)

            # Child spans — one per step
            for step in task_trace.steps:
                span_name = f"kortex.step.{step.agent_id}"
                with self._tracer.start_as_current_span(span_name) as step_span:
                    step_span.set_attribute("kortex.step.index", step.step_index)
                    step_span.set_attribute("kortex.step.agent_id", step.agent_id)
                    step_span.set_attribute("kortex.step.duration_ms", step.duration_ms)

                    # Routing decision attributes
                    rd = step.routing_decision
                    if rd:
                        step_span.set_attribute(
                            "kortex.step.chosen_model",
                            rd.get("chosen_model", ""),
                        )
                        step_span.set_attribute(
                            "kortex.step.chosen_provider",
                            rd.get("chosen_provider", ""),
                        )
                        step_span.set_attribute(
                            "kortex.step.estimated_cost_usd",
                            rd.get("estimated_cost_usd", 0.0),
                        )
                        step_span.set_attribute(
                            "kortex.step.estimated_latency_ms",
                            rd.get("estimated_latency_ms", 0.0),
                        )
                        reasoning = rd.get("reasoning", "")
                        if reasoning:
                            step_span.set_attribute("kortex.step.routing_reasoning", reasoning)

                    # Provider response attributes (only in execute mode)
                    pr = step.provider_response
                    if pr:
                        step_span.set_attribute(
                            "kortex.step.actual_cost_usd", pr.get("cost_usd", 0.0)
                        )
                        step_span.set_attribute(
                            "kortex.step.actual_latency_ms", pr.get("latency_ms", 0.0)
                        )
                        step_span.set_attribute(
                            "kortex.step.input_tokens", pr.get("input_tokens", 0)
                        )
                        step_span.set_attribute(
                            "kortex.step.output_tokens", pr.get("output_tokens", 0)
                        )

                    # Anomaly count
                    anomaly_count = len(step.anomalies) if step.anomalies else 0
                    step_span.set_attribute("kortex.step.anomaly_count", anomaly_count)

                    if anomaly_count > 0:
                        step_span.set_status(StatusCode.ERROR, f"{anomaly_count} anomalies detected")
                    else:
                        step_span.set_status(StatusCode.OK)
