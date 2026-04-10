"""CrewAI integration adapter for Kortex.

Provides wrappers that intercept CrewAI crew execution to route tasks
through Kortex's router and persist state at each agent boundary.

CrewAI is NOT a required dependency -- all imports are lazy so that
Kortex users who don't use CrewAI are never affected.
"""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from kortex.core.capabilities import normalize_capabilities
from kortex.core.exceptions import KortexError
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.types import CoordinationResult, TaskSpec

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(component="crewai_adapter")

# Keyword-to-capabilities mapping for auto-inferring agent capabilities.
# All values MUST be canonical Capability enum values.
_CAPABILITY_KEYWORDS: dict[str, list[str]] = {
    "research": ["research", "analysis"],
    "writ": ["content_generation"],
    "review": ["analysis", "quality_assurance"],
    "code": ["code_generation"],
    "design": ["planning"],
    "test": ["testing", "quality_assurance"],
    "manage": ["planning"],
    "analy": ["analysis", "data_processing"],
}


def _infer_capabilities(role: str) -> list[str]:
    """Infer capabilities from a CrewAI agent role string.

    All returned values are canonical Capability enum values. The result
    is passed through ``normalize_capabilities`` to resolve any aliases
    and validate against the canonical vocabulary.

    Args:
        role: The agent's role string (e.g. "Research Analyst").

    Returns:
        Deduplicated list of canonical capability tags.
    """
    role_lower = role.lower()
    caps: list[str] = []
    seen: set[str] = set()
    for keyword, capabilities in _CAPABILITY_KEYWORDS.items():
        if keyword in role_lower:
            for cap in capabilities:
                if cap not in seen:
                    seen.add(cap)
                    caps.append(cap)
    return normalize_capabilities(caps) if caps else caps


class WrappedCrew:
    """A wrapped CrewAI crew that routes through Kortex on execution.

    Args:
        crew: The original CrewAI crew object.
        adapter: The KortexCrewAIAdapter instance.
        agent_mapping: Maps CrewAI agent role names to Kortex agent_ids.
    """

    def __init__(
        self,
        crew: Any,
        adapter: KortexCrewAIAdapter,
        agent_mapping: dict[str, str],
    ) -> None:
        self._crew = crew
        self._adapter = adapter
        self._agent_mapping = agent_mapping

    async def __call__(
        self,
        task_description: str | None = None,
        execute: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, CoordinationResult]:
        """Execute the crew with Kortex coordination.

        Args:
            task_description: Optional task description override. If None,
                extracts from the crew's tasks.
            execute: If True, call LLM providers. Default False (dry-run).
            **kwargs: Additional keyword arguments passed to crew.kickoff().

        Returns:
            Tuple of (crew_output, CoordinationResult).
        """
        # Build task description from crew if not provided
        if task_description is None:
            task_description = self._extract_task_description()

        task = TaskSpec(content=task_description)

        # Build pipeline from agent mapping (preserving task order)
        pipeline = self._build_pipeline()

        # Run Kortex coordination
        try:
            coordination = await self._adapter._runtime.coordinate(
                task, pipeline, execute=execute,
            )
        except KortexError as exc:
            logger.warning(
                "crew_coordination_failed_fallback",
                error=str(exc),
            )
            coordination = CoordinationResult(
                task_id=task.task_id,
                success=False,
            )

        # Execute the original crew
        crew_output = None
        try:
            kickoff = getattr(self._crew, "kickoff", None)
            if kickoff is not None:
                if asyncio.iscoroutinefunction(kickoff):
                    crew_output = await kickoff(**kwargs)
                else:
                    crew_output = kickoff(**kwargs)
        except Exception as exc:
            logger.warning("crew_kickoff_failed", error=str(exc))

        return crew_output, coordination

    def _extract_task_description(self) -> str:
        """Extract a task description from the crew's tasks."""
        tasks = getattr(self._crew, "tasks", [])
        if tasks:
            first_task = tasks[0]
            desc = getattr(first_task, "description", None)
            if desc:
                return str(desc)
        return "CrewAI crew execution"

    def _build_pipeline(self) -> list[str]:
        """Build the agent pipeline from the crew's task order."""
        # Try to derive order from crew tasks
        tasks = getattr(self._crew, "tasks", [])
        pipeline: list[str] = []
        seen: set[str] = set()

        for task in tasks:
            agent = getattr(task, "agent", None)
            if agent is not None:
                role = getattr(agent, "role", "")
                agent_id = self._agent_mapping.get(role)
                if agent_id and agent_id not in seen:
                    seen.add(agent_id)
                    pipeline.append(agent_id)

        # Fall back to mapping order if no tasks with agents found
        if not pipeline:
            pipeline = list(self._agent_mapping.values())

        return pipeline


class KortexCrewAIAdapter:
    """Adapter that integrates Kortex with CrewAI workflows.

    Intercepts crew execution to route tasks and persist handoff state.
    Falls back to direct execution if Kortex is unavailable.

    Args:
        runtime: The Kortex runtime instance.
    """

    def __init__(self, runtime: KortexRuntime) -> None:
        self._runtime = runtime
        self._log = structlog.get_logger(component="crewai_adapter")
        self._last_checkpoint_id: str | None = None

    def wrap_crew(
        self,
        crew: Any,
        agent_mapping: dict[str, str],
    ) -> WrappedCrew:
        """Wrap a CrewAI crew so execution is routed through Kortex.

        Args:
            crew: A CrewAI Crew object. Must have ``agents`` and ``tasks``
                attributes and a ``kickoff()`` method.
            agent_mapping: Maps CrewAI agent role names to Kortex agent_ids.

        Returns:
            A WrappedCrew that when called returns
            ``(crew_output, CoordinationResult)``.
        """
        return WrappedCrew(crew=crew, adapter=self, agent_mapping=agent_mapping)

    def wrap_task(
        self,
        task_role: str,
        agent_id: str,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that wraps a CrewAI task function with Kortex routing.

        Before the task executes: routes via the router and logs the decision.
        After the task executes: creates a handoff checkpoint with the output.
        On any KortexError: logs a warning and falls back to direct execution.

        Args:
            task_role: The CrewAI agent role name for this task.
            agent_id: The Kortex agent_id for this task.

        Returns:
            A decorator that wraps the task function.
        """
        adapter = self

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Pre-execution: route
                task = TaskSpec(
                    content=f"Task for role '{task_role}'",
                    metadata={"task_role": task_role, "agent_id": agent_id},
                )
                decision = None
                try:
                    decision = await adapter._runtime.route_task(task)
                    adapter._log.info(
                        "task_routed",
                        role=task_role,
                        agent_id=agent_id,
                        model=decision.chosen_model,
                    )
                except KortexError as exc:
                    adapter._log.warning(
                        "task_routing_failed_fallback",
                        role=task_role,
                        agent_id=agent_id,
                        error=str(exc),
                    )

                # Execute the original task function
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)

                # Post-execution: checkpoint
                try:
                    state_snapshot: dict[str, Any] = {
                        "task_role": task_role,
                        "output": result if isinstance(result, dict) else {"value": result},
                    }
                    if decision is not None:
                        state_snapshot["chosen_model"] = decision.chosen_model

                    ctx = await adapter._runtime.persist_handoff(
                        source_agent=task_role,
                        target_agent=agent_id,
                        state_snapshot=state_snapshot,
                        parent_checkpoint_id=adapter._last_checkpoint_id,
                    )
                    adapter._last_checkpoint_id = ctx.checkpoint_id
                except KortexError as exc:
                    adapter._log.warning(
                        "task_checkpoint_failed",
                        role=task_role,
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

    def create_agents_from_crew(self, crew: Any) -> list[AgentDescriptor]:
        """Auto-generate AgentDescriptors from a CrewAI crew definition.

        Extracts agent roles from the crew and infers capabilities based on
        role keywords.

        Args:
            crew: A CrewAI Crew object with an ``agents`` attribute.

        Returns:
            List of AgentDescriptor instances.
        """
        agents_attr = getattr(crew, "agents", [])
        descriptors: list[AgentDescriptor] = []

        for agent in agents_attr:
            role = getattr(agent, "role", "unknown")
            goal = getattr(agent, "goal", "")
            backstory = getattr(agent, "backstory", "")

            # Build agent_id from role: lowercase, replace spaces with hyphens
            agent_id = role.lower().replace(" ", "-")

            # Infer capabilities from role
            capabilities = _infer_capabilities(role)

            description = goal if goal else backstory if backstory else f"Agent with role: {role}"

            descriptors.append(AgentDescriptor(
                agent_id=agent_id,
                name=role,
                description=description,
                capabilities=capabilities,
            ))

        return descriptors
