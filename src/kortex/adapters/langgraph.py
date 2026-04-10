"""LangGraph integration adapter for Kortex.

Provides wrappers that intercept LangGraph node transitions to route tasks
through Kortex's router and persist state at each node boundary.

LangGraph is NOT a required dependency — all imports are lazy so that
Kortex users who don't use LangGraph are never affected.
"""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from kortex.core.exceptions import KortexError
from kortex.core.runtime import KortexRuntime
from kortex.core.types import CoordinationResult, TaskSpec

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(component="langgraph_adapter")


class KortexLangGraphAdapter:
    """Adapter that integrates Kortex with LangGraph workflows.

    Intercepts node transitions to route tasks and persist handoff state.
    Falls back to direct execution if Kortex is unavailable.

    Args:
        runtime: The Kortex runtime instance.
    """

    def __init__(self, runtime: KortexRuntime) -> None:
        self._runtime = runtime
        self._log = structlog.get_logger(component="langgraph_adapter")
        self._last_checkpoint_id: str | None = None

    def wrap_graph(
        self,
        graph: Any,
        agent_mapping: dict[str, str],
    ) -> Callable[[str], Coroutine[Any, Any, tuple[Any, CoordinationResult]]]:
        """Wrap a LangGraph graph so each node is routed through Kortex.

        Args:
            graph: A LangGraph-compatible graph object. Must have a ``nodes``
                attribute (dict of node_name → callable) and an ``invoke``
                method.
            agent_mapping: Maps LangGraph node names to Kortex agent_ids.

        Returns:
            An async callable that accepts a task description string and
            returns ``(graph_output, CoordinationResult)``.
        """
        adapter = self

        async def wrapped(task_description: str, **kwargs: Any) -> tuple[Any, CoordinationResult]:
            task = TaskSpec(content=task_description)

            # Derive pipeline order from the graph's node ordering,
            # filtered to only nodes present in the agent_mapping.
            node_names: list[str] = []
            if hasattr(graph, "nodes"):
                node_names = [n for n in graph.nodes if n in agent_mapping]
            else:
                node_names = list(agent_mapping.keys())

            pipeline = [agent_mapping[n] for n in node_names]

            try:
                coordination = await adapter._runtime.coordinate(task, pipeline)
            except KortexError as exc:
                adapter._log.warning(
                    "coordination_failed_fallback",
                    error=str(exc),
                )
                coordination = CoordinationResult(
                    task_id=task.task_id,
                    success=False,
                )

            # Invoke the original graph
            if asyncio.iscoroutinefunction(getattr(graph, "invoke", None)):
                graph_output = await graph.invoke(task_description, **kwargs)
            else:
                graph_output = graph.invoke(task_description, **kwargs)

            return graph_output, coordination

        return wrapped

    def wrap_node(
        self,
        node_name: str,
        agent_id: str,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that wraps a LangGraph node function with Kortex routing.

        Before the node executes: routes via the router and logs the decision.
        After the node executes: creates a handoff checkpoint with the output.
        On any KortexError: logs a warning and falls back to direct execution.

        Args:
            node_name: The LangGraph node name.
            agent_id: The Kortex agent_id for this node.

        Returns:
            A decorator that wraps the node function.
        """
        adapter = self

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Pre-execution: route
                task = TaskSpec(
                    content=f"Node '{node_name}' execution",
                    metadata={"node_name": node_name, "agent_id": agent_id},
                )
                decision = None
                try:
                    decision = await adapter._runtime.route_task(task)
                    adapter._log.info(
                        "node_routed",
                        node=node_name,
                        agent_id=agent_id,
                        model=decision.chosen_model,
                    )
                except KortexError as exc:
                    adapter._log.warning(
                        "node_routing_failed_fallback",
                        node=node_name,
                        agent_id=agent_id,
                        error=str(exc),
                    )

                # Execute the original node
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)

                # Post-execution: checkpoint
                try:
                    state_snapshot: dict[str, Any] = {
                        "node_name": node_name,
                        "output": result if isinstance(result, dict) else {"value": result},
                    }
                    if decision is not None:
                        state_snapshot["chosen_model"] = decision.chosen_model

                    ctx = await adapter._runtime.persist_handoff(
                        source_agent=node_name,
                        target_agent=agent_id,
                        state_snapshot=state_snapshot,
                        parent_checkpoint_id=adapter._last_checkpoint_id,
                    )
                    adapter._last_checkpoint_id = ctx.checkpoint_id
                except KortexError as exc:
                    adapter._log.warning(
                        "node_checkpoint_failed",
                        node=node_name,
                        error=str(exc),
                    )

                return result

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return asyncio.get_event_loop().run_until_complete(
                    async_wrapper(*args, **kwargs)
                )

            if asyncio.iscoroutinefunction(fn):
                return async_wrapper
            return sync_wrapper

        return decorator


def kortex_middleware(
    runtime: KortexRuntime,
) -> Callable[[str, str, dict[str, Any]], Coroutine[Any, Any, None]]:
    """Create a callback compatible with LangGraph's event system.

    Returns an async callback that can be registered as a LangGraph event
    handler. On each node transition it logs a handoff through Kortex.

    Args:
        runtime: The Kortex runtime instance.

    Returns:
        An async callback ``(event_type, node_name, data) -> None``.
    """
    log = structlog.get_logger(component="langgraph_middleware")
    last_checkpoint_id: dict[str, str | None] = {"value": None}

    async def callback(event_type: str, node_name: str, data: dict[str, Any]) -> None:
        if event_type != "node_end":
            return
        try:
            ctx = await runtime.persist_handoff(
                source_agent=node_name,
                target_agent=data.get("next_node", "__end__"),
                state_snapshot=data,
                parent_checkpoint_id=last_checkpoint_id["value"],
            )
            last_checkpoint_id["value"] = ctx.checkpoint_id
            log.info("middleware_handoff", node=node_name, checkpoint=ctx.checkpoint_id)
        except KortexError as exc:
            log.warning("middleware_handoff_failed", node=node_name, error=str(exc))

    return callback
