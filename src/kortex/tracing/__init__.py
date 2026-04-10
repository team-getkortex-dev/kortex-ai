"""Kortex tracing integrations.

Optional exporters for sending TaskTrace data to external observability
platforms. Install the relevant extras for the exporter you need:

    pip install kortex-ai[otel]   # OpenTelemetry
"""

from kortex.tracing.otel_exporter import OTELExporter

__all__ = ["OTELExporter"]
