# Framework Adapters

Kortex integrates with existing agent frameworks through adapters. Adapters intercept agent transitions to route tasks through the Kortex router and persist state at each boundary.

Framework packages (LangGraph, CrewAI) are **not required dependencies** — all imports are lazy. Users who don't use a particular framework are never affected.

## LangGraph

### wrap_graph

Wrap an entire LangGraph graph so each node is routed through Kortex:

```python
from kortex.adapters.langgraph import KortexLangGraphAdapter

adapter = KortexLangGraphAdapter(runtime)

wrapped = adapter.wrap_graph(graph, agent_mapping={
    "research_node": "researcher",
    "write_node": "writer",
    "review_node": "reviewer",
})

# Returns (graph_output, CoordinationResult)
output, coordination = await wrapped("Analyze this data")
```

The `agent_mapping` maps LangGraph node names to Kortex agent IDs. The pipeline order is derived from the graph's node ordering.

### wrap_node

Decorator for individual node functions:

```python
@adapter.wrap_node("research_node", "researcher")
async def research(state: dict) -> dict:
    # Your node logic here
    return {"findings": "..."}
```

What happens:

- **Before**: routes via the router, logs the decision
- **After**: creates a handoff checkpoint with the output
- **On error**: logs a warning, falls back to direct execution

Works with both sync and async node functions.

### Middleware

For LangGraph's event system:

```python
from kortex.adapters.langgraph import kortex_middleware

callback = kortex_middleware(runtime)
# Register as a LangGraph event handler
# On each "node_end" event, logs a handoff through Kortex
```

## CrewAI

### wrap_crew

Wrap a CrewAI crew for Kortex coordination:

```python
from kortex.adapters.crewai import KortexCrewAIAdapter

adapter = KortexCrewAIAdapter(runtime)

wrapped = adapter.wrap_crew(crew, agent_mapping={
    "Research Analyst": "researcher",
    "Content Writer": "writer",
    "Editor": "reviewer",
})

# Dry-run (default)
crew_output, coordination = await wrapped("Research AI trends")

# Live execution
crew_output, coordination = await wrapped("Research AI trends", execute=True)
```

The `agent_mapping` maps CrewAI agent role names to Kortex agent IDs. Pipeline order is derived from the crew's task order.

### wrap_task

Decorator for individual task functions:

```python
@adapter.wrap_task("Research Analyst", "researcher")
async def do_research(topic: str) -> dict:
    return {"findings": f"Research on {topic}"}
```

Same behavior as LangGraph's `wrap_node` — routes before, checkpoints after, falls back on error.

### create_agents_from_crew

Auto-generate AgentDescriptors from a CrewAI crew definition:

```python
descriptors = adapter.create_agents_from_crew(crew)
for desc in descriptors:
    runtime.register_agent(desc)
```

Capabilities are inferred from role keywords and normalized to the canonical vocabulary:

| Keyword in role | Inferred capabilities |
|----------------|----------------------|
| `research` | `research`, `analysis` |
| `writ` | `content_generation` |
| `review` | `analysis`, `quality_assurance` |
| `code` | `code_generation` |
| `design` | `planning` |
| `test` | `testing`, `quality_assurance` |
| `manage` | `planning` |
| `analy` | `analysis`, `data_processing` |

## Building a Custom Adapter

To integrate Kortex with a framework not listed above, follow this pattern:

```python
from kortex.core.exceptions import KortexError
from kortex.core.runtime import KortexRuntime
from kortex.core.types import CoordinationResult, TaskSpec

class MyFrameworkAdapter:
    def __init__(self, runtime: KortexRuntime) -> None:
        self._runtime = runtime

    async def run_pipeline(self, task_description: str, agents: list[str]) -> CoordinationResult:
        task = TaskSpec(content=task_description)
        try:
            return await self._runtime.coordinate(task, agents)
        except KortexError:
            # Fall back to direct execution
            return CoordinationResult(task_id=task.task_id, success=False)
```

Key principles:

1. **Lazy imports** — don't import the framework at module level. Use `TYPE_CHECKING` guards.
2. **Fail-open** — catch `KortexError` and fall back to direct execution.
3. **Map identifiers** — provide an `agent_mapping` that translates framework-specific names to Kortex agent IDs.
4. **Checkpoint after execution** — use `runtime._state.handoff()` to create checkpoints at each agent boundary.
