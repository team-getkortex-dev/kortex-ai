# Changelog

All notable changes to Kortex are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions.

---

## [Unreleased]

### Added

#### Core runtime
- `KortexRuntime` — central async coordinator for multi-agent pipelines
- `KortexRuntime.coordinate()` — routes and optionally executes tasks across an ordered agent pipeline with full trace and anomaly support
- `KortexRuntime.stream_coordinate()` — async generator variant that yields `(event_type, payload)` tuples in real time (routing decisions, streamed tokens, handoffs, completion)
- Public adapter API: `route_task()`, `persist_handoff()`, `record_event()`, `get_router()`, `get_policy()` — no private attribute access required from adapters

#### Routing
- `Router` — heuristic routing engine with cost, latency, capability, and complexity-based model selection
- `RoutingPolicy` — composable policy engine with constraints, objectives, and fallback rules
- `PolicyRouter` — evaluates `RoutingPolicy` against a candidate set with full score breakdown
- `ObservedMetrics` — EWMA feedback loop that converges router estimates toward observed latency/cost
- Canonical `Capability` enum with alias normalization

#### State management
- `StateManager` — unified facade over three backends: `InMemoryStateStore`, `SQLiteStateStore` (WAL mode), `RedisStateStore`
- Checkpoint chains with rollback support
- Idempotent batch operations for bulk handoff creation

#### Anomaly detection & recovery
- `FailureDetector` with configurable `DetectionPolicy` — monitors cost overruns, latency spikes, output quality drops
- `RecoveryExecutor` — executes recovery actions: retry, fallback, rollback, escalate
- Recovery records attached to `CoordinationResult` for full auditability

#### Tracing & replay
- `TaskTrace` / `TraceStep` — structured per-step trace with policy snapshots and provider responses
- `InMemoryTraceStore` and `SQLiteTraceStore` — trace persistence
- `ReplayEngine` — re-runs a saved trace under a different policy and computes step-level diffs

#### Providers
- `GenericOpenAIConnector` — universal connector for any OpenAI-compatible API
- `AnthropicConnector` — Anthropic Messages API with streaming
- `OpenAIConnector` — OpenAI Chat Completions with model catalog
- `OpenRouterConnector` — OpenRouter multi-provider gateway
- `ProviderRegistry` — auto-discovery from environment variables, model aggregation

#### Provider resilience
- `RetryPolicy` — exponential backoff with per-status-code retryability rules
- `CircuitBreaker` — CLOSED / OPEN / HALF_OPEN state machine with configurable thresholds
- `ResilientClient` — httpx wrapper combining retry + circuit breaker + timeouts
- Exception taxonomy: `ProviderTimeoutError`, `ProviderRateLimitError`, `ProviderOverloadError`, `ProviderAuthError`, `CircuitOpenError`

#### Framework adapters
- `KortexLangGraphAdapter` — wrap any LangGraph node with `@adapter.wrap_node()`
- `KortexCrewAIAdapter` — wrap any CrewAI task with `@adapter.wrap_task()`, with automatic capability inference from crew role descriptions
- Adapter boundary: all adapter ↔ runtime communication goes through the public API only

#### Benchmark harness
- `BenchmarkHarness` — runs Kortex routing against static baselines (cheapest, strongest, random)
- `TaskDataset` — three pre-built workload datasets: mixed, cost-sensitive, latency-sensitive
- `BenchmarkReport.to_markdown()` — tabular comparison with cost savings % and latency percentiles

#### Configuration
- `KortexConfig` (pydantic-settings) — all settings readable from `KORTEX_*` environment variables
- `get_config()` / `reset_config()` — singleton accessor with test-friendly reset
- Fields: `log_level`, `default_backend`, `trace_enabled`, `trace_store`, `trace_db`, `default_timeout_ms`, `max_retries`, `circuit_breaker_threshold`, `circuit_breaker_recovery_s`

#### CLI
- `kortex status` — provider health, model tiers, agent registry, state store type
- `kortex models` — full model catalog sorted by tier and cost
- `kortex dry-run` — route a task without executing it
- `kortex history` — checkpoint history with agent/task filters
- `kortex trace list/show/export` — trace management
- `kortex replay` — replay a trace under a different policy
- `kortex policy diff/show` — policy comparison and inspection
- `kortex benchmark run/compare` — benchmark routing quality
- `kortex stream` — stream a task pipeline in real time, printing tokens as they arrive

#### OpenTelemetry (optional)
- `OTELExporter` — exports `TaskTrace` objects as OTEL spans with full attribute mapping
- Install with: `pip install kortex-ai[otel]`

#### Developer tooling
- `scripts/kortex_system_load_test.py` — concurrency load harness (dry-run and mock modes)
- Stress test suite with `ChaosProvider` for fault injection testing

### Fixed

- Handoff `state_snapshot` now contains only the boundary-crossing payload; execution metadata is stored separately in `CoordinationResult.steps`
- Store lifecycle: `SQLiteStateStore` correctly requires `start()` before use; `InMemoryStateStore` is startless
- Detector anomaly types are enum values, not free-form strings
- Model identity uses composite `provider::model_name` keys to prevent cross-provider collisions
- Capability strings are validated against the canonical `Capability` enum at registration time — free-form strings are rejected
- Adapter boundary: removed all `._router` and `._state` private accesses from adapters

---

[Unreleased]: https://github.com/kortex-ai/kortex/compare/HEAD...HEAD
