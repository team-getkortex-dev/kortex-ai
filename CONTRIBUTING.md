# Contributing to Kortex

## Dev Environment Setup

```bash
git clone https://github.com/team-getkortex-dev/kortex-ai.git
cd kortex-ai
uv sync --extra dev
```

## Running Tests

```bash
uv run pytest                  # all tests
uv run pytest tests/unit       # unit tests only
uv run pytest tests/integration # integration tests only
uv run pytest -x               # stop on first failure
```

## Linting and Formatting

```bash
uv run ruff check src/         # lint
uv run ruff format src/        # format
uv run pyright src/            # type check
```

## Code Style

Follow the conventions in [CLAUDE.md](CLAUDE.md):

- All public functions have Google-style docstrings
- Type annotations everywhere — no `Any` unless truly necessary
- Async throughout — all public APIs are async
- Custom exceptions inherit from `KortexError`
- Structured logging via `structlog` — no print statements
- Config via pydantic-settings with `KORTEX_` prefix

## PR Process

1. Fork the repo and create a feature branch from `main`
2. Make your changes
3. Run `uv run pytest` — all tests must pass
4. Run `uv run ruff check src/` and `uv run ruff format src/`
5. Open a PR against `main` with a clear description of what and why

## Architecture Rules

- Don't import framework-specific code (langgraph, crewai) in `core/` — adapters only
- Don't use synchronous blocking calls in async code paths
- Don't store secrets in code — use env vars
- Don't add dependencies without checking for lighter alternatives
