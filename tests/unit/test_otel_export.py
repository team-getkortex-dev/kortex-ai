"""Tests for OTELExporter — skipped when opentelemetry is not installed."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

try:
    import opentelemetry  # noqa: F401
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

pytestmark = pytest.mark.skipif(not HAS_OTEL, reason="opentelemetry not installed")


# ---------------------------------------------------------------------------
# Helpers — build a minimal TaskTrace without running a full coordination
# ---------------------------------------------------------------------------


def _make_trace(
    success: bool = True,
    step_count: int = 1,
    has_provider_response: bool = False,
    anomalies_per_step: int = 0,
):
    from kortex.core.trace import TaskTrace, TraceStep

    steps = []
    for i in range(step_count):
        routing_decision = {
            "chosen_model": f"model-{i}",
            "chosen_provider": "test-provider",
            "estimated_cost_usd": 0.001 * (i + 1),
            "estimated_latency_ms": 100.0 * (i + 1),
            "reasoning": f"step {i} reasoning",
        }
        provider_response = None
        if has_provider_response:
            provider_response = {
                "content": "hello",
                "model": f"model-{i}",
                "provider": "test-provider",
                "input_tokens": 50,
                "output_tokens": 20,
                "cost_usd": 0.0005,
                "latency_ms": 80.0,
            }
        anomalies = [{"type": "cost_overrun"}] * anomalies_per_step
        steps.append(TraceStep(
            step_index=i,
            agent_id=f"agent-{i}",
            input_payload={"content": "test"},
            routing_decision=routing_decision,
            policy_snapshot={"name": "test-policy"},
            provider_response=provider_response,
            handoff_checkpoint_id=None,
            anomalies=anomalies,
            recovery_records=[],
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=50.0 * (i + 1),
        ))

    return TaskTrace(
        task_id="trace-test-001",
        task_content="test content",
        task_complexity="simple",
        pipeline=[f"agent-{i}" for i in range(step_count)],
        steps=steps,
        policy_snapshot={"name": "test-policy"},
        total_estimated_cost_usd=0.001 * step_count,
        total_actual_cost_usd=0.0005 * step_count if has_provider_response else 0.0,
        total_duration_ms=150.0 * step_count,
        success=success,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOTELExporterInstantiation:
    def test_import_succeeds(self) -> None:
        from kortex.tracing import OTELExporter
        assert OTELExporter is not None

    def test_instantiate_with_sdk_provider(self) -> None:
        from opentelemetry.sdk.trace import TracerProvider

        from kortex.tracing import OTELExporter

        provider = TracerProvider()
        exporter = OTELExporter(tracer_provider=provider)
        assert exporter is not None

    def test_instantiate_without_provider_uses_global(self) -> None:
        from kortex.tracing import OTELExporter

        # Should not raise — falls back to global provider
        exporter = OTELExporter()
        assert exporter is not None

    def test_custom_service_name(self) -> None:
        from opentelemetry.sdk.trace import TracerProvider

        from kortex.tracing import OTELExporter

        provider = TracerProvider()
        exporter = OTELExporter(tracer_provider=provider, service_name="my-service")
        assert exporter._service_name == "my-service"


class TestOTELExporterSpanAttributes:
    @pytest.fixture()
    def in_memory_exporter(self):
        """Returns (OTELExporter, InMemorySpanExporter) pair."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from kortex.tracing import OTELExporter

        mem_exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(mem_exporter))

        otel_exporter = OTELExporter(tracer_provider=provider, service_name="test")
        return otel_exporter, mem_exporter

    def test_root_span_task_id_attribute(self, in_memory_exporter) -> None:
        otel_exp, mem_exp = in_memory_exporter
        trace = _make_trace(step_count=1)
        otel_exp.export_trace(trace)

        spans = mem_exp.get_finished_spans()
        root = next(s for s in spans if s.name == "kortex.coordination")
        assert root.attributes["kortex.task_id"] == "trace-test-001"

    def test_root_span_has_pipeline_attribute(self, in_memory_exporter) -> None:
        otel_exp, mem_exp = in_memory_exporter
        trace = _make_trace(step_count=2)
        otel_exp.export_trace(trace)

        spans = mem_exp.get_finished_spans()
        root = next(s for s in spans if s.name == "kortex.coordination")
        assert "agent-0,agent-1" == root.attributes["kortex.pipeline"]

    def test_root_span_success_attribute(self, in_memory_exporter) -> None:
        otel_exp, mem_exp = in_memory_exporter
        trace = _make_trace(success=True)
        otel_exp.export_trace(trace)

        spans = mem_exp.get_finished_spans()
        root = next(s for s in spans if s.name == "kortex.coordination")
        assert root.attributes["kortex.success"] is True

    def test_root_span_failure_status_when_not_successful(self, in_memory_exporter) -> None:
        from opentelemetry.trace import StatusCode

        otel_exp, mem_exp = in_memory_exporter
        trace = _make_trace(success=False)
        otel_exp.export_trace(trace)

        spans = mem_exp.get_finished_spans()
        root = next(s for s in spans if s.name == "kortex.coordination")
        assert root.status.status_code == StatusCode.ERROR

    def test_step_spans_created(self, in_memory_exporter) -> None:
        otel_exp, mem_exp = in_memory_exporter
        trace = _make_trace(step_count=3)
        otel_exp.export_trace(trace)

        spans = mem_exp.get_finished_spans()
        step_spans = [s for s in spans if s.name.startswith("kortex.step.")]
        assert len(step_spans) == 3

    def test_step_span_model_attribute(self, in_memory_exporter) -> None:
        otel_exp, mem_exp = in_memory_exporter
        trace = _make_trace(step_count=1)
        otel_exp.export_trace(trace)

        spans = mem_exp.get_finished_spans()
        step_span = next(s for s in spans if s.name == "kortex.step.agent-0")
        assert step_span.attributes["kortex.step.chosen_model"] == "model-0"
        assert step_span.attributes["kortex.step.chosen_provider"] == "test-provider"

    def test_step_span_provider_response_attributes(self, in_memory_exporter) -> None:
        otel_exp, mem_exp = in_memory_exporter
        trace = _make_trace(step_count=1, has_provider_response=True)
        otel_exp.export_trace(trace)

        spans = mem_exp.get_finished_spans()
        step_span = next(s for s in spans if s.name == "kortex.step.agent-0")
        assert step_span.attributes["kortex.step.actual_cost_usd"] == 0.0005
        assert step_span.attributes["kortex.step.input_tokens"] == 50
        assert step_span.attributes["kortex.step.output_tokens"] == 20

    def test_step_span_anomaly_count(self, in_memory_exporter) -> None:
        otel_exp, mem_exp = in_memory_exporter
        trace = _make_trace(step_count=1, anomalies_per_step=2)
        otel_exp.export_trace(trace)

        spans = mem_exp.get_finished_spans()
        step_span = next(s for s in spans if s.name == "kortex.step.agent-0")
        assert step_span.attributes["kortex.step.anomaly_count"] == 2

    def test_total_span_count(self, in_memory_exporter) -> None:
        """1 root + N step spans."""
        otel_exp, mem_exp = in_memory_exporter
        trace = _make_trace(step_count=2)
        otel_exp.export_trace(trace)

        spans = mem_exp.get_finished_spans()
        assert len(spans) == 3  # 1 root + 2 steps


class TestOTELExporterWithoutOtel:
    def test_import_error_without_otel(self, monkeypatch) -> None:
        """OTELExporter raises ImportError when opentelemetry is not installed."""
        import sys
        import unittest.mock

        # Simulate opentelemetry not being available
        with unittest.mock.patch.dict(sys.modules, {"opentelemetry.trace": None}):
            from kortex.tracing.otel_exporter import _require_otel
            with pytest.raises((ImportError, TypeError)):
                _require_otel()
