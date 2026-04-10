# Kortex — Agent Coordination Runtime

## What This Project Is
Kortex is a middleware runtime for multi-agent AI systems. It sits between agent frameworks (LangGraph, CrewAI) and LLM providers to handle:
1. **Heuristic task routing** — rule-based model selection per sub-task using cost, latency, capability, and complexity policies
2. **Stateful handoff management** — context preservation across agent boundaries with checkpoint chains and rollback
3. **Threshold-based anomaly detection** — monitors cost overruns, latency spikes, and output quality drops with configurable recovery actions (retry, fallback, rollback, escalate)

## Tech Stack
- **Language:** Python 3.11+
- **Package manager:** uv (preferred) or pip
- **Async:** asyncio throughout — all public APIs must be async
- **State store:** Redis (hot state), SQLite (durable checkpoints, local dev), in-memory (testing)
- **Testing:** pytest + pytest-asyncio
- **Linting:** ruff
- **Type checking:** pyright in strict mode
- **Docs:** mkdocs-material

## Project Structure
```
kortex/
├── src/kortex/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py       # Public exports for core module
│   │   ├── capabilities.py   # Canonical capability enum, validation, normalization
│   │   ├── detector.py       # Threshold-based anomaly detection
│   │   ├── exceptions.py     # KortexError hierarchy
│   │   ├── recovery.py       # Recovery executor (retry, fallback, rollback, escalate)
│   │   ├── router.py         # Heuristic routing engine
│   │   ├── runtime.py        # Main runtime orchestrator (KortexRuntime)
│   │   ├── state.py          # State/checkpoint management
│   │   └── types.py          # Pydantic models (TaskSpec, RoutingDecision, etc.)
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── crewai.py         # CrewAI adapter with capability inference
│   │   └── langgraph.py      # LangGraph adapter with node wrapping
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py           # ProviderConnector protocol + GenericOpenAIConnector
│   │   ├── anthropic.py      # Anthropic Messages API connector
│   │   ├── openai.py         # OpenAI connector (extends GenericOpenAIConnector)
│   │   ├── openrouter.py     # OpenRouter connector
│   │   └── registry.py       # ProviderRegistry with auto-discovery
│   ├── store/
│   │   ├── __init__.py
│   │   ├── base.py           # StateStore protocol
│   │   ├── memory.py         # In-memory store (testing)
│   │   ├── redis.py          # Redis store (production)
│   │   └── sqlite.py         # SQLite store (local dev)
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── cli.py            # Terminal CLI (status, models, dry-run, history, config)
│   │   └── formatter.py      # ANSI color + table formatting
│   └── config.py             # Configuration management
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── stress/               # Chaos provider stress tests
│   └── fixtures/
├── examples/                  # Working examples (run without API keys)
├── docs/
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## Commands
- `uv run pytest` — run all tests
- `uv run pytest tests/unit` — unit tests only
- `uv run pytest tests/stress` — stress tests with chaos providers
- `uv run ruff check src/` — lint
- `uv run ruff format src/` — format
- `uv run pyright src/` — type check
- `uv run mkdocs serve` — local docs

## Conventions
- All public functions and classes have docstrings (Google style)
- Use `typing` annotations everywhere — no `Any` unless truly necessary
- Error handling: custom exception hierarchy rooted in `KortexError`
- Logging via `structlog` — structured JSON logs, no print statements
- Config via pydantic-settings with env var support (KORTEX_ prefix)
- All state operations are idempotent and safe under concurrent access
- Router decisions are logged with full context as structured events
- Capabilities use a canonical enum (`Capability` in `core/capabilities.py`) — free-form strings are rejected at registration boundaries
- Models are identified by composite keys (provider::model_name) to prevent cross-provider collisions
- Test coverage target: 80%+ on core/, 60%+ on adapters/

## Architecture Principles
- **Framework-agnostic:** Kortex wraps existing frameworks, never replaces them
- **Minimal integration surface:** Adding Kortex to an existing project should require <20 lines of code
- **Fail-open by default:** If Kortex is unavailable, agents fall back to direct execution
- **Linear pipelines:** Agents are specified as an ordered list; no DAG support currently
- **Async throughout:** All I/O is non-blocking via asyncio

## What NOT To Do
- Don't import framework-specific code (langgraph, crewai) in core/ — adapters only
- Don't use synchronous blocking calls in async code paths
- Don't store secrets in code — use env vars or config files
- Don't add dependencies without checking: is there a lighter alternative?
- Don't use free-form capability strings — always use values from the Capability enum
