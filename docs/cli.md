# CLI Reference

Kortex includes a terminal CLI for inspecting providers, models, routing decisions, and checkpoint history.

## Installation

The CLI is installed automatically with the package:

```bash
pip install kortex-ai
```

Run with:

```bash
kortex <command>
# or
python -m kortex.dashboard.cli <command>
```

## Demo Mode

When no API keys are detected, the CLI runs in **demo mode** with example models. This lets you explore all commands without any configuration.

## Commands

### status

Show system status — providers, models by tier, registered agents, and state store.

```bash
$ kortex status
Kortex Status
[demo mode - no API keys detected, showing example models]

Providers (2):
  openai: OK
  anthropic: OK

Models (4):
  fast: 1
  balanced: 2
  powerful: 1

Agents: 3
  researcher: Gathers and analyzes information
  writer: Drafts content and documentation
  reviewer: Reviews and provides feedback

State Store: InMemoryStateStore
```

### models

List all available models across all providers, sorted by tier and cost.

```bash
$ kortex models
PROVIDER   MODEL                     TIER      INPUT/1K  OUTPUT/1K  LATENCY  CAPABILITIES
---------  ------------------------  --------  --------  ---------  -------  -------------------------
openai     gpt-4o-mini               fast      $0.0001   $0.0006    200ms    reasoning, content_generation
anthropic  claude-sonnet-4-20250514  balanced  $0.0030   $0.01      800ms    reasoning, code_generation, content_generation
openai     gpt-4o                    balanced  $0.0050   $0.01      600ms    reasoning, code_generation, content_generation
anthropic  claude-opus-4-20250514    powerful  $0.01     $0.07      2.0s     reasoning, code_generation, content_generation, analysis
```

Local models (zero cost) are tagged with `[local]`.

### dry-run

Run a routing dry-run to see what models would be selected and at what cost.

```bash
$ kortex dry-run --task "Analyze data" --complexity complex --pipeline researcher,writer,reviewer
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | *(required)* | Task description |
| `--complexity` | `moderate` | `simple`, `moderate`, or `complex` |
| `--pipeline` | all agents | Comma-separated agent IDs |

Output includes routing decisions (model, reasoning, cost, latency, fallback), total cost estimate, and handoff chain with checkpoint IDs.

### history

Show checkpoint history from the state store.

```bash
$ kortex history
$ kortex history --agent researcher
$ kortex history --task <task-id>
$ kortex history --last 5
```

Options:

| Flag | Description |
|------|-------------|
| `--agent` | Filter by agent ID |
| `--task` | Filter by task ID |
| `--last N` | Show only the N most recent checkpoints |

### config

Show current configuration — state store, providers, routing strategy, environment variables.

```bash
$ kortex config
Kortex Configuration

State Store: InMemoryStateStore
Providers: openai, anthropic
Routing Strategy: HeuristicRoutingStrategy
Log Level: INFO

Environment:
  KORTEX_LOG_LEVEL=DEBUG
  KORTEX_STATE_BACKEND=sqlite
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `KORTEX_STATE_BACKEND` | State store backend: `memory`, `redis`, or `sqlite` |
| `KORTEX_LOG_LEVEL` | Log level for structlog output |
| `OPENAI_API_KEY` | Enables OpenAI provider auto-discovery |
| `ANTHROPIC_API_KEY` | Enables Anthropic provider auto-discovery |
| `OPENROUTER_API_KEY` | Enables OpenRouter provider auto-discovery |
| `NO_COLOR` | Disables ANSI color output when set |
