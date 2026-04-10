# Observability

Kortex ships a built-in trace system (`TaskTrace`) and an optional OpenTelemetry exporter that bridges Kortex traces to any OTEL-compatible backend (Jaeger, Tempo, Honeycomb, Datadog, etc.).

---

## Built-in Tracing

Every call to `KortexRuntime.coordinate()` automatically produces a `TaskTrace` attached to the result. No configuration required.

```python
result = await runtime.coordinate(task, ["researcher", "writer"])

# Access the trace
trace_dict = result.trace   # dict representation
```

To persist traces, configure a `TraceStore`:

```python
from kortex.core.trace_store import SQLiteTraceStore

runtime = KortexRuntime(
    router=router,
    state_manager=state_manager,
    trace_store=SQLiteTraceStore("kortex_traces.db"),
)
```

Then use the CLI to inspect them:

```bash
kortex trace list
kortex trace show <trace-id>
kortex trace export <trace-id> --output trace.json
kortex replay <trace-id> --policy examples/policies/cost_optimized.toml
```

---

## OpenTelemetry Export

### Installation

```bash
pip install kortex-ai[otel]
```

### Basic Usage

```python
import asyncio
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from kortex import KortexRuntime, Router, StateManager
from kortex.tracing import OTELExporter

# Set up OTEL provider (sends spans to your collector)
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
)

exporter = OTELExporter(tracer_provider=provider, service_name="my-agent-system")

async def main():
    async with KortexRuntime(router=Router(), state_manager=StateManager.create("memory")) as runtime:
        result = await runtime.coordinate(task, pipeline)

        # Export the trace to OTEL
        if result.trace:
            from kortex.core.trace import TaskTrace
            task_trace = TaskTrace(**result.trace)  # reconstruct from dict
            exporter.export_trace(task_trace)

asyncio.run(main())
```

### Console Output (for development)

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from kortex.tracing import OTELExporter

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

exporter = OTELExporter(tracer_provider=provider)
exporter.export_trace(task_trace)
```

---

## Span Structure

Each `export_trace()` call produces:

```
kortex.coordination          # root span
├── kortex.step.researcher   # one span per agent step
└── kortex.step.writer
```

### Root span attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `kortex.task_id` | string | Unique task identifier |
| `kortex.task_complexity` | string | simple / moderate / complex |
| `kortex.pipeline` | string | Comma-separated agent IDs |
| `kortex.total_estimated_cost_usd` | float | Estimated total cost |
| `kortex.total_actual_cost_usd` | float | Actual cost (if executed) |
| `kortex.total_duration_ms` | float | End-to-end wall time |
| `kortex.success` | bool | Whether the pipeline succeeded |
| `kortex.step_count` | int | Number of pipeline steps |
| `kortex.policy_name` | string | Active routing policy name |

### Step span attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `kortex.step.index` | int | Step position (0-based) |
| `kortex.step.agent_id` | string | Agent that ran this step |
| `kortex.step.chosen_model` | string | Model selected by router |
| `kortex.step.chosen_provider` | string | Provider selected by router |
| `kortex.step.estimated_cost_usd` | float | Router's cost estimate |
| `kortex.step.estimated_latency_ms` | float | Router's latency estimate |
| `kortex.step.routing_reasoning` | string | Why this model was chosen |
| `kortex.step.actual_cost_usd` | float | Real cost (execute mode only) |
| `kortex.step.actual_latency_ms` | float | Real latency (execute mode only) |
| `kortex.step.input_tokens` | int | Input token count |
| `kortex.step.output_tokens` | int | Output token count |
| `kortex.step.anomaly_count` | int | Anomalies detected this step |
| `kortex.step.duration_ms` | float | Step wall time |

---

## Using the Global Tracer Provider

If you have already configured the OTEL global provider elsewhere in your application, just omit `tracer_provider`:

```python
from kortex.tracing import OTELExporter

# Uses whatever global TracerProvider was configured by your framework
exporter = OTELExporter(service_name="my-service")
exporter.export_trace(task_trace)
```
