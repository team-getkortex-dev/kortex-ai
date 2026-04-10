"""Main Kortex runtime orchestrator.

Coordinates multi-agent workflows by routing tasks, managing handoffs,
and emitting structured events for every decision. Optionally executes
tasks against real LLM providers when ``execute=True``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import TracebackType
from typing import TYPE_CHECKING, Any, AsyncGenerator
from uuid import uuid4

import structlog

from kortex.core.capabilities import normalize_capabilities
from kortex.core.exceptions import ProviderError, RoutingFailedError
from kortex.core.metrics import ObservedMetrics
from kortex.core.recovery import (
    RecoveryAction,
    RecoveryContext,
    RecoveryExecutor,
    RecoveryPolicy,
    RecoveryRecord,
    recovery_event,
)
from kortex.core.router import Router
from kortex.core.state import StateManager
from kortex.core.types import (
    CoordinationResult,
    ExecutionEvent,
    HandoffContext,
    RoutingDecision,
    StepExecutionRecord,
    StepInput,
    StepOutput,
    TaskSpec,
)

if TYPE_CHECKING:
    from kortex.cache.semantic_cache import SemanticCache
    from kortex.config import KortexConfig
    from kortex.core.detector import FailureDetector
    from kortex.core.trace import TaskTrace
    from kortex.core.trace_store import TraceStore
    from kortex.providers.base import ProviderResponse
    from kortex.providers.registry import ProviderRegistry
    from kortex.router.cost_estimate import CostEstimate


@dataclass
class AgentDescriptor:
    """Describes an agent registered with the runtime.

    Args:
        agent_id: Unique identifier for the agent.
        name: Human-readable name.
        description: What this agent does.
        capabilities: List of capability tags.
        preferred_model: Optional model override for this agent.
    """

    agent_id: str
    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    preferred_model: str | None = None


class KortexRuntime:
    """Central coordinator for multi-agent workflows.

    Args:
        router: The routing engine.
        state_manager: The state/checkpoint manager.
        registry: Optional provider registry for executing tasks against LLMs.
        detector: Optional failure detector for anomaly monitoring.
        recovery_policy: Optional recovery policy. When provided together with
            a detector, anomaly recommendations are actually executed (retry,
            fallback, rollback, escalate) instead of just logged.
        enable_tracing: When True (default), every ``coordinate()`` call
            produces a full ``TaskTrace`` attached to the result.
        trace_store: Optional trace store. When provided, traces are
            automatically persisted after each ``coordinate()`` call.
    """

    def __init__(
        self,
        router: Router,
        state_manager: StateManager,
        registry: ProviderRegistry | None = None,
        detector: FailureDetector | None = None,
        recovery_policy: RecoveryPolicy | None = None,
        enable_tracing: bool = True,
        trace_store: TraceStore | None = None,
        config: KortexConfig | None = None,
    ) -> None:
        self._config = config
        self._router = router
        self._state = state_manager
        self._registry: ProviderRegistry | None = registry
        self._detector: FailureDetector | None = detector
        self._recovery_policy = recovery_policy
        self._recovery_executor: RecoveryExecutor | None = None
        if detector is not None and recovery_policy is not None:
            self._recovery_executor = RecoveryExecutor(recovery_policy)
        self._agents: dict[str, AgentDescriptor] = {}
        self._enable_tracing = enable_tracing
        self._trace_store: TraceStore | None = trace_store
        self._metrics = ObservedMetrics()
        # Attach metrics to router so it uses EWMA values during routing
        self._router.set_metrics(self._metrics)
        self._cache: SemanticCache | None = None
        self._log = structlog.get_logger(component="runtime")

        # Dashboard telemetry ------------------------------------------------
        from collections import deque

        self._dashboard_total_tasks = 0
        self._dashboard_total_cost_usd = 0.0
        self._dashboard_total_latency_ms = 0.0
        self._dashboard_latency_count = 0
        self._dashboard_active_tasks: set[str] = set()
        self._dashboard_model_usage: dict[str, int] = {}
        self._dashboard_recent_decisions: deque[dict[str, Any]] = deque(maxlen=50)
        self._dashboard_cost_history: deque[float] = deque(maxlen=60)
        self._dashboard_latency_history: deque[float] = deque(maxlen=60)
        self._event_stream: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)

    def register_agent(self, descriptor: AgentDescriptor) -> None:
        """Add an agent to the registry.

        Normalizes the agent's capabilities (resolving aliases) so they
        match the canonical vocabulary used by providers and the router.

        Args:
            descriptor: The agent descriptor to register.

        Raises:
            ValueError: If any capability is not canonical and not a known alias.
        """
        if descriptor.capabilities:
            descriptor.capabilities = normalize_capabilities(descriptor.capabilities)
        self._agents[descriptor.agent_id] = descriptor

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry.

        Args:
            agent_id: The agent to remove.
        """
        self._agents.pop(agent_id, None)

    async def route_task(self, task: TaskSpec) -> RoutingDecision:
        """Route a single task through the router.

        Public entry point for adapters and external callers. Delegates to the
        internal router without exposing private attributes.

        Args:
            task: The task to route.

        Returns:
            The routing decision.
        """
        return await self._router.route(task)

    async def persist_handoff(
        self,
        source_agent: str,
        target_agent: str,
        state_snapshot: dict[str, Any],
        parent_checkpoint_id: str | None = None,
    ) -> HandoffContext:
        """Create a handoff checkpoint between two agents.

        Public entry point for adapters and external callers. Delegates to the
        internal state manager without exposing private attributes.

        Args:
            source_agent: The agent handing off.
            target_agent: The agent receiving the handoff.
            state_snapshot: The state payload to persist at this boundary.
            parent_checkpoint_id: Optional ID of the previous checkpoint.

        Returns:
            The created HandoffContext.
        """
        return await self._state.handoff(
            source_agent=source_agent,
            target_agent=target_agent,
            state_snapshot=state_snapshot,
            parent_checkpoint_id=parent_checkpoint_id,
        )

    async def stream_coordinate(
        self,
        task: TaskSpec,
        agent_pipeline: list[str],
    ) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        """Coordinate a task pipeline and stream events as they occur.

        Yields structured events for every significant step: routing decisions,
        streamed tokens (when the provider supports streaming), handoff
        checkpoints, and a final completion summary.

        Event types and payload shapes:

        - ``"routing_decision"`` — ``{step, agent_id, chosen_model, chosen_provider,
          reasoning, estimated_cost_usd, estimated_latency_ms}``
        - ``"handoff"`` — ``{checkpoint_id, source, target, step}``
        - ``"token"`` — ``{token, agent_id, step}``  *(provider-dependent)*
        - ``"completion"`` — ``{task_id, agents_routed, duration_ms, success}``
        - ``"error"`` — ``{agent_id, step, error}``

        The method is fail-open: if routing or streaming fails for an agent,
        an ``"error"`` event is yielded and the pipeline continues.

        Args:
            task: The task specification to coordinate.
            agent_pipeline: Ordered list of agent_ids to run the task through.

        Yields:
            Tuples of ``(event_type, payload_dict)``.
        """
        import time as _time

        start = _time.monotonic()
        agents_routed = 0

        initial_handoff = await self._state.handoff(
            source_agent="__input__",
            target_agent=agent_pipeline[0] if agent_pipeline else "__none__",
            state_snapshot={"task_id": task.task_id, "content": task.content},
        )
        yield ("handoff", {
            "checkpoint_id": initial_handoff.checkpoint_id,
            "source": "__input__",
            "target": initial_handoff.target_agent,
            "step": -1,
        })

        last_checkpoint_id: str | None = initial_handoff.checkpoint_id
        prev_agent_id = "__input__"

        for i, agent_id in enumerate(agent_pipeline):
            # Apply agent capabilities if registered
            step_task = task
            agent_desc = self._agents.get(agent_id)
            if agent_desc is not None and agent_desc.capabilities and not step_task.required_capabilities:
                step_task = step_task.model_copy(update={
                    "required_capabilities": list(agent_desc.capabilities),
                })

            # Route
            try:
                decision = await self._router.route(step_task)
            except Exception as exc:
                yield ("error", {"agent_id": agent_id, "step": i, "error": str(exc)})
                prev_agent_id = agent_id
                continue

            agents_routed += 1
            yield ("routing_decision", {
                "step": i,
                "agent_id": agent_id,
                "chosen_model": decision.chosen_model,
                "chosen_provider": decision.chosen_provider,
                "reasoning": decision.reasoning,
                "estimated_cost_usd": decision.estimated_cost_usd,
                "estimated_latency_ms": decision.estimated_latency_ms,
            })

            # Handoff checkpoint (for non-first agents)
            if prev_agent_id != "__input__":
                handoff = await self._state.handoff(
                    source_agent=prev_agent_id,
                    target_agent=agent_id,
                    state_snapshot={"task_id": task.task_id, "step": i},
                    parent_checkpoint_id=last_checkpoint_id,
                )
                last_checkpoint_id = handoff.checkpoint_id
                yield ("handoff", {
                    "checkpoint_id": handoff.checkpoint_id,
                    "source": prev_agent_id,
                    "target": agent_id,
                    "step": i,
                })

            # Stream tokens if provider is available and supports streaming
            if self._registry is not None:
                try:
                    connector = self._registry.get_provider(decision.chosen_provider)
                    if hasattr(connector, "stream"):
                        async for token in connector.stream(
                            prompt=task.content,
                            model=decision.chosen_model,
                        ):
                            yield ("token", {
                                "token": token,
                                "agent_id": agent_id,
                                "step": i,
                            })
                except Exception as exc:
                    yield ("error", {"agent_id": agent_id, "step": i, "error": str(exc)})

            prev_agent_id = agent_id

        duration_ms = (_time.monotonic() - start) * 1000
        yield ("completion", {
            "task_id": task.task_id,
            "agents_routed": agents_routed,
            "total_agents": len(agent_pipeline),
            "duration_ms": round(duration_ms, 2),
            "success": agents_routed > 0,
        })

    async def record_event(self, event: ExecutionEvent) -> None:
        """Record an execution event.

        No-op stub for adapters that emit events. Events are held in-memory
        within CoordinationResult; this method exists for adapter API symmetry.

        Args:
            event: The event to record.
        """

    def set_cache(self, cache: "SemanticCache") -> None:
        """Attach a SemanticCache to this runtime.

        Once set, ``coordinate()`` checks the cache before routing and
        stores successful results after completion.

        Args:
            cache: The SemanticCache instance to use.
        """
        self._cache = cache
        self._log.info("semantic_cache_attached")

    def get_dashboard_snapshot(self) -> "DashboardMetrics":
        """Return a ``DashboardMetrics`` snapshot of current runtime telemetry.

        The snapshot is built from counters updated during ``coordinate()`` calls.
        Intended for the TUI and any monitoring consumers.

        Returns:
            A ``DashboardMetrics`` instance populated with current values.
        """
        from collections import deque

        from kortex.dashboard.tui import DashboardMetrics

        # Provider health from registry
        provider_health: dict[str, bool] = {}
        if self._registry is not None:
            for name in self._registry.list_providers():
                provider_health[name] = True  # assume healthy; TUI can probe async

        avg_latency = (
            self._dashboard_total_latency_ms / self._dashboard_latency_count
            if self._dashboard_latency_count > 0
            else 0.0
        )

        # Router decision cache hit rate
        dc_hit_rate = 0.0
        if hasattr(self._router, "_decision_cache") and self._router._decision_cache is not None:
            dc_hit_rate = self._router._decision_cache.hit_rate

        # Collect per-model percentile latencies from the ObservedMetrics tracker
        model_latency_p95: dict[str, float] = {}
        model_latency_p99: dict[str, float] = {}
        if self._metrics is not None:
            for model_key in self._metrics.known_models():
                p95 = self._metrics.get_latency_p95(model_key)
                p99 = self._metrics.get_latency_p99(model_key)
                if p95 > 0.0 or p99 > 0.0:
                    model_latency_p95[model_key] = p95
                    model_latency_p99[model_key] = p99

        # Snapshot recent decisions and histories
        return DashboardMetrics(
            total_tasks_routed=self._dashboard_total_tasks,
            total_cost_usd=self._dashboard_total_cost_usd,
            avg_latency_ms=avg_latency,
            cache_hits=self._dashboard_total_tasks - len(self._dashboard_active_tasks),
            cache_misses=0,
            active_tasks=list(self._dashboard_active_tasks),
            provider_health=provider_health,
            model_usage=dict(self._dashboard_model_usage),
            recent_decisions=deque(self._dashboard_recent_decisions, maxlen=50),
            cost_history=deque(self._dashboard_cost_history, maxlen=60),
            latency_history=deque(self._dashboard_latency_history, maxlen=60),
            decision_cache_hit_rate=dc_hit_rate,
            model_latency_p95=model_latency_p95,
            model_latency_p99=model_latency_p99,
        )

    def get_router(self) -> Router:
        """Return the router for read-only inspection.

        Returns:
            The Router instance.
        """
        return self._router

    def get_policy(self) -> Any:
        """Return the current routing policy, if one is set.

        Returns:
            The active RoutingPolicy, or None.
        """
        return self._router.get_policy()

    async def start(self) -> None:
        """Start the runtime by connecting the state manager's store.

        Call this before ``coordinate()`` when using non-memory backends,
        or use the runtime as an async context manager instead.
        """
        await self._state.start()
        self._log.info("runtime_started")

    async def stop(self) -> None:
        """Stop the runtime, close provider clients, and disconnect the state store."""
        if self._registry is not None:
            await self._registry.close_all()
        await self._state.stop()
        # Release all pooled HTTP connections
        from kortex.providers.http_pool import ConnectionPool

        await ConnectionPool.get_instance().close_all()
        self._log.info("runtime_stopped")

    async def __aenter__(self) -> KortexRuntime:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()

    async def execute_routed_task(
        self, task: TaskSpec, decision: RoutingDecision
    ) -> ProviderResponse:
        """Execute a routed task against the chosen provider.

        Looks up the provider from the decision, calls ``complete()``, and
        falls back to the fallback model on failure.

        Args:
            task: The task specification.
            decision: The routing decision with chosen provider/model.

        Returns:
            The ProviderResponse from the provider.

        Raises:
            ProviderError: If both primary and fallback calls fail, or if
                no registry is available.
        """
        if self._registry is None:
            raise ProviderError("No ProviderRegistry configured on this runtime")

        try:
            connector = self._registry.get_provider(decision.chosen_provider)
        except KeyError as exc:
            raise ProviderError(
                f"Provider '{decision.chosen_provider}' not found in registry"
            ) from exc

        # Try primary model
        try:
            response = await connector.complete(
                prompt=task.content,
                model=decision.chosen_model,
            )
            return response
        except Exception as primary_exc:
            self._log.warning(
                "primary_model_failed",
                provider=decision.chosen_provider,
                model=decision.chosen_model,
                error=str(primary_exc),
            )

            # Try fallback
            if decision.fallback_model:
                self._log.info(
                    "failover_to_fallback",
                    primary=decision.chosen_model,
                    fallback=decision.fallback_model,
                )
                try:
                    response = await connector.complete(
                        prompt=task.content,
                        model=decision.fallback_model,
                    )
                    return response
                except Exception as fallback_exc:
                    raise ProviderError(
                        f"Both primary model '{decision.chosen_model}' and fallback "
                        f"'{decision.fallback_model}' failed. "
                        f"Primary: {primary_exc}. Fallback: {fallback_exc}"
                    ) from fallback_exc

            raise ProviderError(
                f"Primary model '{decision.chosen_model}' failed and no fallback "
                f"is available: {primary_exc}"
            ) from primary_exc

    async def coordinate(
        self,
        task: TaskSpec,
        agent_pipeline: list[str],
        execute: bool = False,
        export_trace: bool = False,
    ) -> CoordinationResult:
        """Coordinate a task across an ordered pipeline of agents.

        For each agent in the pipeline:
        1. Build a StepInput from the previous step's output (or task content)
        2. Create a handoff checkpoint with ONLY the boundary-crossing payload
        3. Route the task
        4. If execute=True, call the provider
        5. Build a StepExecutionRecord with routing/response/anomalies/timing
        6. Build a StepOutput from the execution result

        When a detector AND recovery_policy are configured, anomaly
        recommendations are honored: retry re-routes and re-executes,
        fallback uses the fallback model, rollback restores the previous
        checkpoint and skips the agent, and escalate stops the pipeline.

        HandoffContext.state_snapshot contains only the input payload crossing
        the agent boundary. Execution metadata is stored in
        CoordinationResult.steps.

        Args:
            task: The task specification to coordinate.
            agent_pipeline: Ordered list of agent_ids to run the task through.
            execute: If True and a ProviderRegistry is available, actually
                call the LLM providers. Default False (dry-run).
            export_trace: If True, serialize the ``TaskTrace`` and attach it
                to ``result.trace``.  Default False — skips serialization for
                ~2-3ms savings when the caller does not need the trace dict.
                Traces are still persisted to the trace store regardless of
                this flag.

        Returns:
            CoordinationResult with decisions, handoffs, events, steps, and metrics.

        Raises:
            RoutingFailedError: If no agent in the pipeline can be routed.
        """
        # -- Semantic cache check -------------------------------------------------
        if self._cache is not None:
            cached = await self._cache.get(task, agent_pipeline, execute)
            if cached is not None:
                self._log.info(
                    "cache_hit",
                    task_id=task.task_id,
                    pipeline=agent_pipeline,
                )
                result = CoordinationResult(**{
                    k: v for k, v in cached.items()
                    if k in CoordinationResult.model_fields
                })
                if self._enable_tracing and result.trace:
                    result.trace["cache_hit"] = True
                return result

        start = time.monotonic()
        should_execute = execute and self._registry is not None

        # Reset recovery counters for this coordination
        if self._recovery_executor is not None:
            self._recovery_executor.reset()

        self._log.info(
            "coordination_start",
            task_id=task.task_id,
            pipeline=agent_pipeline,
            execute=should_execute,
        )

        routing_decisions: list[RoutingDecision] = []
        handoffs: list[HandoffContext] = []
        pending_handoffs: list[HandoffContext] = []  # for parallel save at end
        events: list[ExecutionEvent] = []
        responses: list[dict[str, Any]] = []
        step_records: list[dict[str, Any]] = []
        anomaly_dicts: list[dict[str, Any]] = []
        recovery_record_dicts: list[dict[str, Any]] = []
        errors: list[str] = []
        last_checkpoint_id: str | None = None
        escalated = False
        retry_counts_per_step: dict[int, int] = {}

        # Dashboard: mark task active
        self._dashboard_active_tasks.add(task.task_id)

        # Determine whether to use parallel handoff saves
        _parallel_handoffs = getattr(self._config, "parallel_handoffs_enabled", True)

        # The initial input payload is the task content
        current_payload: dict[str, Any] = {
            "task_id": task.task_id,
            "content": task.content,
            "metadata": task.metadata,
        }

        # Pre-create initial checkpoint (no DB write yet when parallel mode is on)
        initial_handoff = self._state._create_handoff_context(
            source_agent="__input__",
            target_agent=agent_pipeline[0] if agent_pipeline else "__none__",
            state_snapshot=current_payload,
        )
        pending_handoffs.append(initial_handoff)
        handoffs.append(initial_handoff)
        events.append(self._make_event("handoff", task.task_id, initial_handoff.target_agent, {
            "checkpoint_id": initial_handoff.checkpoint_id,
            "source": "__input__",
            "target": initial_handoff.target_agent,
        }))
        last_checkpoint_id = initial_handoff.checkpoint_id

        prev_agent_id = "__input__"
        prev_handoff: HandoffContext | None = initial_handoff

        for i, agent_id in enumerate(agent_pipeline):
            if escalated:
                break

            step_start = time.monotonic()
            step_started_at = datetime.now(timezone.utc)
            step_anomalies: list[dict[str, Any]] = []

            # Build the task for this step
            step_task = task
            if i > 0:
                step_task = task.model_copy(update={"complexity_hint": "moderate"})

            # If executing and we have output from a previous agent, feed it forward
            if should_execute and "output_content" in current_payload:
                step_task = step_task.model_copy(update={
                    "content": f"{task.content}\n\nPrevious agent output:\n{current_payload['output_content']}",
                })

            # Agent capabilities: if the agent has capabilities and the task
            # doesn't specify required_capabilities, use the agent's.
            agent_desc = self._agents.get(agent_id)
            if agent_desc is not None and agent_desc.capabilities and not step_task.required_capabilities:
                step_task = step_task.model_copy(update={
                    "required_capabilities": list(agent_desc.capabilities),
                })
                self._log.info(
                    "agent_capabilities_applied",
                    agent_id=agent_id,
                    capabilities=agent_desc.capabilities,
                )

            # Attempt routing
            try:
                decision = await self._router.route(step_task)
                routing_decisions.append(decision)
                events.append(self._make_route_event(decision))

                # Dashboard telemetry
                self._dashboard_total_tasks += 1
                self._dashboard_total_cost_usd += decision.estimated_cost_usd or 0.0
                _lat = decision.estimated_latency_ms or 0.0
                self._dashboard_total_latency_ms += _lat
                self._dashboard_latency_count += 1
                _model_key = f"{decision.chosen_provider}::{decision.chosen_model}"
                self._dashboard_model_usage[_model_key] = (
                    self._dashboard_model_usage.get(_model_key, 0) + 1
                )
                _dec_dict: dict[str, Any] = {
                    "agent_id": agent_id,
                    "chosen_model": decision.chosen_model,
                    "chosen_provider": decision.chosen_provider,
                    "estimated_cost_usd": decision.estimated_cost_usd or 0.0,
                    "estimated_latency_ms": _lat,
                }
                self._dashboard_recent_decisions.append(_dec_dict)
                self._dashboard_cost_history.append(decision.estimated_cost_usd or 0.0)
                self._dashboard_latency_history.append(_lat)
                # Non-blocking event push
                try:
                    self._event_stream.put_nowait(_dec_dict)
                except asyncio.QueueFull:
                    pass
            except RoutingFailedError as exc:
                errors.append(f"Agent '{agent_id}': {exc}")
                events.append(self._make_event("failure", task.task_id, agent_id, {
                    "error": str(exc),
                }))
                prev_agent_id = agent_id
                continue

            # Detector: check routing
            skip_step = False
            if self._detector is not None:
                anomaly = await self._detector.check_routing(decision, step_task)
                if anomaly is not None:
                    anomaly_dict = anomaly.to_dict()
                    anomaly_dicts.append(anomaly_dict)
                    step_anomalies.append(anomaly_dict)
                    events.append(self._make_event("failure", task.task_id, agent_id, {
                        "anomaly": anomaly.anomaly_type.value,
                        "action": anomaly.recommended_action,
                    }))

                    # Recovery executor handles the anomaly
                    if self._recovery_executor is not None:
                        rec_ctx = RecoveryContext(
                            task=step_task,
                            current_step_index=i,
                            agent_id=agent_id,
                            router=self._router,
                            state_manager=self._state,
                            registry=self._registry,
                            last_checkpoint_id=last_checkpoint_id,
                            execute_mode=should_execute,
                            current_routing_decision=decision,
                            retry_count_this_step=retry_counts_per_step.get(i, 0),
                            total_retry_count=self._recovery_executor.total_retries_used,
                        )
                        record = await self._recovery_executor.execute(anomaly, rec_ctx)
                        recovery_record_dicts.append(record.to_dict())
                        events.append(recovery_event(record, task.task_id, agent_id))

                        if record.action_taken == RecoveryAction.ESCALATED:
                            escalated = True
                            break
                        elif record.action_taken == RecoveryAction.ROLLED_BACK:
                            skip_step = True
                        elif record.action_taken == RecoveryAction.RETRIED:
                            retry_counts_per_step[i] = retry_counts_per_step.get(i, 0) + 1
                        elif record.action_taken == RecoveryAction.FELL_BACK:
                            pass  # proceed with next step
                        # CONTINUED: proceed normally
                    else:
                        # No recovery executor — legacy behavior
                        if anomaly.recommended_action == "escalate":
                            escalated = True
                        elif anomaly.recommended_action == "rollback":
                            skip_step = True

            if skip_step:
                prev_agent_id = agent_id
                continue

            # Execute against the provider if requested
            provider_response_dict: dict[str, Any] | None = None
            if should_execute:
                try:
                    resp = await self.execute_routed_task(step_task, decision)
                    provider_response_dict = {
                        "content": resp.content,
                        "model": resp.model,
                        "provider": resp.provider,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                        "latency_ms": resp.latency_ms,
                    }
                    responses.append(provider_response_dict)

                    # Feed observed latency/cost back into the EWMA metrics
                    # so the router self-calibrates over successive calls.
                    # Use the task's complexity_class for stratified tracking.
                    model_key = f"{decision.chosen_provider}::{decision.chosen_model}"
                    self._metrics.update(
                        model_key,
                        latency_ms=resp.latency_ms,
                        cost_usd=resp.cost_usd,
                        complexity_class=step_task.complexity_class,
                    )

                    # Detector: check execution
                    if self._detector is not None:
                        exec_anomaly = await self._detector.check_execution(
                            provider_response_dict, decision,
                        )
                        if exec_anomaly is not None:
                            exec_anomaly_dict = exec_anomaly.to_dict()
                            anomaly_dicts.append(exec_anomaly_dict)
                            step_anomalies.append(exec_anomaly_dict)
                            events.append(self._make_event("failure", task.task_id, agent_id, {
                                "anomaly": exec_anomaly.anomaly_type.value,
                                "action": exec_anomaly.recommended_action,
                            }))

                            # Recovery executor handles execution anomaly
                            if self._recovery_executor is not None:
                                rec_ctx = RecoveryContext(
                                    task=step_task,
                                    current_step_index=i,
                                    agent_id=agent_id,
                                    router=self._router,
                                    state_manager=self._state,
                                    registry=self._registry,
                                    last_checkpoint_id=last_checkpoint_id,
                                    execute_mode=should_execute,
                                    current_routing_decision=decision,
                                    retry_count_this_step=retry_counts_per_step.get(i, 0),
                                    total_retry_count=self._recovery_executor.total_retries_used,
                                )
                                exec_record = await self._recovery_executor.execute(
                                    exec_anomaly, rec_ctx,
                                )
                                recovery_record_dicts.append(exec_record.to_dict())
                                events.append(recovery_event(exec_record, task.task_id, agent_id))

                                if exec_record.action_taken == RecoveryAction.ESCALATED:
                                    escalated = True
                                    break
                                elif exec_record.action_taken == RecoveryAction.RETRIED:
                                    retry_counts_per_step[i] = retry_counts_per_step.get(i, 0) + 1
                            else:
                                # No recovery executor — legacy behavior
                                if exec_anomaly.recommended_action == "escalate":
                                    escalated = True

                except ProviderError as exc:
                    self._log.warning(
                        "execution_failed",
                        agent_id=agent_id,
                        error=str(exc),
                    )
                    events.append(self._make_event("failure", task.task_id, agent_id, {
                        "error": str(exc),
                        "phase": "execution",
                    }))

            # Build the step execution record (metadata separate from handoff)
            step_completed_at = datetime.now(timezone.utc)
            step_duration_ms = (time.monotonic() - step_start) * 1000

            step_record = StepExecutionRecord(
                step_index=i,
                agent_id=agent_id,
                routing_decision=decision,
                provider_response=provider_response_dict,
                anomalies=step_anomalies,
                started_at=step_started_at,
                completed_at=step_completed_at,
                duration_ms=step_duration_ms,
            )
            step_records.append(step_record.model_dump(mode="json"))

            # Build the output payload for this step
            if provider_response_dict is not None:
                output_payload: dict[str, Any] = {
                    "task_id": task.task_id,
                    "output_content": provider_response_dict["content"],
                    "step": i,
                    "agent_id": agent_id,
                }
            else:
                # Dry-run: pass through the task content as the output
                output_payload = {
                    "task_id": task.task_id,
                    "step": i,
                    "agent_id": agent_id,
                }

            # Create handoff from previous agent to this one
            # state_snapshot = ONLY the boundary-crossing payload
            if prev_agent_id != "__input__":
                handoff = self._state._create_handoff_context(
                    source_agent=prev_agent_id,
                    target_agent=agent_id,
                    state_snapshot=current_payload,
                    parent_checkpoint_id=last_checkpoint_id,
                )
                pending_handoffs.append(handoff)
                handoffs.append(handoff)
                events.append(self._make_event("handoff", task.task_id, agent_id, {
                    "checkpoint_id": handoff.checkpoint_id,
                    "source": prev_agent_id,
                    "target": agent_id,
                }))

                # Detector: check handoff
                if self._detector is not None:
                    handoff_anomaly = await self._detector.check_handoff(
                        handoff, prev_handoff,
                    )
                    if handoff_anomaly is not None:
                        anomaly_dicts.append(handoff_anomaly.to_dict())

                        if self._recovery_executor is not None:
                            rec_ctx = RecoveryContext(
                                task=step_task,
                                current_step_index=i,
                                agent_id=agent_id,
                                router=self._router,
                                state_manager=self._state,
                                registry=self._registry,
                                last_checkpoint_id=last_checkpoint_id,
                                execute_mode=should_execute,
                                current_routing_decision=decision,
                                retry_count_this_step=retry_counts_per_step.get(i, 0),
                                total_retry_count=self._recovery_executor.total_retries_used,
                            )
                            handoff_record = await self._recovery_executor.execute(
                                handoff_anomaly, rec_ctx,
                            )
                            recovery_record_dicts.append(handoff_record.to_dict())
                            events.append(recovery_event(handoff_record, task.task_id, agent_id))
                            if handoff_record.action_taken == RecoveryAction.ESCALATED:
                                escalated = True
                        else:
                            if handoff_anomaly.recommended_action == "escalate":
                                escalated = True

                last_checkpoint_id = handoff.checkpoint_id
                prev_handoff = handoff

            # Advance: next step's input = this step's output
            current_payload = output_payload
            prev_agent_id = agent_id

        # Persist all handoff contexts: parallel when enabled, sequential fallback
        if _parallel_handoffs and len(pending_handoffs) > 1:
            await self._state.execute_handoffs_parallel(pending_handoffs)
        else:
            for ctx in pending_handoffs:
                await self._state._store.save_checkpoint(ctx)

        duration_ms = (time.monotonic() - start) * 1000

        if not routing_decisions:
            raise RoutingFailedError(
                f"All agents in pipeline failed routing for task {task.task_id}. "
                f"Errors: {'; '.join(errors)}"
            )

        total_estimated = sum(d.estimated_cost_usd for d in routing_decisions)
        actual_cost = sum(r.get("cost_usd", 0.0) for r in responses)

        # Completion event
        events.append(self._make_event("completion", task.task_id, None, {
            "agents_routed": len(routing_decisions),
            "agents_failed": len(errors),
            "total_estimated_cost_usd": total_estimated,
            "actual_cost_usd": actual_cost,
            "duration_ms": duration_ms,
        }))

        result = CoordinationResult(
            task_id=task.task_id,
            routing_decisions=routing_decisions,
            handoffs=handoffs,
            events=events,
            total_estimated_cost_usd=total_estimated,
            responses=responses,
            actual_cost_usd=actual_cost,
            anomalies=anomaly_dicts,
            recovery_records=recovery_record_dicts,
            steps=step_records,
            duration_ms=duration_ms,
            success=not escalated,
        )

        # Detector: check full coordination
        if self._detector is not None:
            coord_anomalies = await self._detector.check_coordination(result)
            for a in coord_anomalies:
                result.anomalies.append(a.to_dict())
                if self._recovery_executor is not None:
                    rec_ctx = RecoveryContext(
                        task=task,
                        current_step_index=len(agent_pipeline) - 1,
                        agent_id=agent_pipeline[-1] if agent_pipeline else "",
                        router=self._router,
                        state_manager=self._state,
                        registry=self._registry,
                        last_checkpoint_id=last_checkpoint_id,
                        execute_mode=should_execute,
                    )
                    coord_record = await self._recovery_executor.execute(a, rec_ctx)
                    result.recovery_records.append(coord_record.to_dict())
                    events.append(recovery_event(coord_record, task.task_id))
                    if coord_record.action_taken == RecoveryAction.ESCALATED:
                        result.success = False
                else:
                    if a.recommended_action == "escalate":
                        result.success = False

        # Build and attach trace if tracing is enabled
        if self._enable_tracing:
            trace = self._build_trace(task, agent_pipeline, result, step_records)

            # Only serialize (expensive) when the caller explicitly requests it.
            # Trace store persistence always happens so auditing is unaffected.
            if export_trace:
                result.trace = trace.to_dict()

            # Persist trace if a store is configured (uses lazy to_dict() internally)
            if self._trace_store is not None:
                try:
                    await self._trace_store.save_trace(trace)
                except Exception as exc:
                    self._log.warning(
                        "trace_persist_failed",
                        trace_id=trace.trace_id,
                        error=str(exc),
                    )

        summary = self.get_coordination_summary(result)
        self._log.info("coordination_complete", task_id=task.task_id, summary=summary)

        # Dashboard: task no longer active
        self._dashboard_active_tasks.discard(task.task_id)

        # -- Store successful result in cache ------------------------------------
        if self._cache is not None and result.success:
            try:
                await self._cache.set(
                    task,
                    agent_pipeline,
                    result.model_dump(mode="json"),
                    execute=execute,
                )
            except Exception as exc:
                self._log.warning("cache_store_failed", error=str(exc))

        return result

    async def coordinate_batch(
        self,
        tasks: list[TaskSpec],
        agent_pipelines: list[list[str]],
        execute: bool = False,
    ) -> list[CoordinationResult]:
        """Coordinate multiple tasks concurrently.

        Uses ``asyncio.gather`` to run all ``coordinate()`` calls in parallel.
        The ``tasks`` and ``agent_pipelines`` lists must have the same length;
        ``agent_pipelines[i]`` is used for ``tasks[i]``.

        Args:
            tasks: The tasks to coordinate.
            agent_pipelines: One pipeline list per task.
            execute: Whether to execute against providers. Default False.

        Returns:
            A list of ``CoordinationResult`` objects in task order.

        Raises:
            ValueError: If ``tasks`` and ``agent_pipelines`` differ in length.
            RoutingFailedError: If any task cannot be routed (propagated from
                ``asyncio.gather``).
        """
        import asyncio

        if len(tasks) != len(agent_pipelines):
            raise ValueError(
                f"tasks ({len(tasks)}) and agent_pipelines ({len(agent_pipelines)}) "
                "must have the same length"
            )
        return list(
            await asyncio.gather(
                *[
                    self.coordinate(t, p, execute=execute)
                    for t, p in zip(tasks, agent_pipelines)
                ]
            )
        )

    async def estimate_cost(
        self,
        tasks: list[TaskSpec],
        agent_pipelines: list[list[str]],
    ) -> "CostEstimate":
        """Estimate the cost of coordinating a batch of tasks without executing.

        Dry-routes each task through its pipeline and aggregates the
        estimated costs. Tasks that fail routing contribute $0 and increment
        ``routing_failures``.

        Args:
            tasks: The tasks to estimate.
            agent_pipelines: One pipeline list per task.

        Returns:
            A ``CostEstimate`` with total, per-model, and per-task breakdown.

        Raises:
            ValueError: If ``tasks`` and ``agent_pipelines`` differ in length.
        """
        from kortex.router.cost_estimate import CostEstimate

        if len(tasks) != len(agent_pipelines):
            raise ValueError(
                f"tasks ({len(tasks)}) and agent_pipelines ({len(agent_pipelines)}) "
                "must have the same length"
            )

        per_task: list[float] = []
        per_model: dict[str, float] = {}
        routing_failures = 0

        for task, pipeline in zip(tasks, agent_pipelines):
            task_cost = 0.0
            try:
                for agent_id in pipeline:
                    step_task = task
                    agent_desc = self._agents.get(agent_id)
                    if (
                        agent_desc is not None
                        and agent_desc.capabilities
                        and not step_task.required_capabilities
                    ):
                        step_task = step_task.model_copy(
                            update={"required_capabilities": list(agent_desc.capabilities)}
                        )
                    decision = await self._router.route(step_task)
                    task_cost += decision.estimated_cost_usd
                    key = f"{decision.chosen_provider}::{decision.chosen_model}"
                    per_model[key] = per_model.get(key, 0.0) + decision.estimated_cost_usd
            except RoutingFailedError:
                routing_failures += 1
            per_task.append(task_cost)

        total = sum(per_task)
        return CostEstimate(
            total_usd=total,
            per_model=per_model,
            per_task=per_task,
            task_count=len(tasks),
            routing_failures=routing_failures,
        )

    async def rollback_to(self, checkpoint_id: str) -> HandoffContext:
        """Roll back to a specific checkpoint.

        Args:
            checkpoint_id: The checkpoint to restore.

        Returns:
            The HandoffContext at that checkpoint.

        Raises:
            CheckpointNotFoundError: If the checkpoint does not exist.
        """
        context = await self._state.rollback(checkpoint_id)

        self._log.info(
            "runtime_rollback",
            checkpoint_id=checkpoint_id,
            source=context.source_agent,
            target=context.target_agent,
        )

        return context

    def get_coordination_summary(self, result: CoordinationResult) -> str:
        """Produce a human-readable summary of a coordination result.

        Args:
            result: The coordination result to summarize.

        Returns:
            A summary string.
        """
        models = [d.chosen_model for d in result.routing_decisions]
        model_chain = " -> ".join(models) if models else "none"
        agent_count = len(result.routing_decisions)
        est_cost = f"${result.total_estimated_cost_usd:.4f}"
        duration = f"{result.duration_ms:.0f}ms"

        handoff_status = "All handoffs successful." if result.success else "Some handoffs failed."

        parts = [
            f"Task {result.task_id} coordinated across {agent_count} agent(s) "
            f"in {duration}. Models: {model_chain}. "
            f"Est. cost: {est_cost}.",
        ]

        if result.responses:
            actual = f"${result.actual_cost_usd:.4f}"
            parts.append(f" Actual cost: {actual}.")
            if result.total_estimated_cost_usd > 0:
                savings = (
                    (result.total_estimated_cost_usd - result.actual_cost_usd)
                    / result.total_estimated_cost_usd
                    * 100
                )
                parts.append(f" Saved: {savings:.0f}%.")

        if result.anomalies:
            parts.append(f" Anomalies: {len(result.anomalies)}.")

        if result.steps:
            parts.append(f" Steps: {len(result.steps)}.")

        # Recovery summary
        if result.recovery_records:
            recovery_parts: list[str] = []
            for rec in result.recovery_records:
                action = rec.get("action_taken", "unknown")
                success = rec.get("success", False)
                detail = rec.get("detail", "")

                if action == "escalated":
                    # Extract anomaly type for context
                    atype = rec.get("anomaly_type", "unknown")
                    recovery_parts.append(
                        f"Pipeline failed: escalated after {atype}"
                        + (f" ({detail})" if detail else "")
                    )
                elif action == "retried":
                    status = "succeeded" if success else "failed"
                    recovery_parts.append(f"1 retry ({status})")
                elif action == "fell_back":
                    status = "succeeded" if success else "failed"
                    # Try to extract fallback model from detail
                    recovery_parts.append(f"1 fallback ({status})")
                elif action == "rolled_back":
                    recovery_parts.append("1 rollback")
                elif action == "continued":
                    pass  # Don't clutter summary with continue actions

            if recovery_parts:
                parts.append(f" Recovery: {', '.join(recovery_parts)}.")

        parts.append(f" {handoff_status}")

        return "".join(parts)

    async def get_trace(self, trace_id: str) -> TaskTrace:
        """Retrieve a trace by ID from the trace store.

        Args:
            trace_id: The trace identifier.

        Returns:
            The TaskTrace.

        Raises:
            KeyError: If the trace is not found.
            RuntimeError: If no trace store is configured.
        """
        if self._trace_store is None:
            raise RuntimeError("No trace store configured on this runtime")
        from kortex.core.trace import TaskTrace
        return await self._trace_store.get_trace(trace_id)

    async def list_traces(
        self,
        limit: int = 50,
        task_id: str | None = None,
    ) -> list[TaskTrace]:
        """List traces from the trace store.

        Args:
            limit: Maximum number of traces to return.
            task_id: If set, filter to traces for this task.

        Returns:
            List of TaskTrace objects, most recent first.

        Raises:
            RuntimeError: If no trace store is configured.
        """
        if self._trace_store is None:
            raise RuntimeError("No trace store configured on this runtime")
        from kortex.core.trace import TaskTrace
        return await self._trace_store.list_traces(limit=limit, task_id=task_id)

    def _build_trace(
        self,
        task: TaskSpec,
        agent_pipeline: list[str],
        result: CoordinationResult,
        step_records: list[dict[str, Any]],
    ) -> TaskTrace:
        """Build a TaskTrace from the completed coordination result."""
        from kortex.core.trace import TaskTrace, TraceStep

        # Get the active policy snapshot
        policy_snapshot: dict[str, Any] = {}
        active_policy = self._router.get_policy()
        if active_policy is not None:
            policy_snapshot = active_policy.to_dict()

        # Build trace steps from step_records and routing decisions
        trace_steps: list[TraceStep] = []
        for record_dict in step_records:
            step_idx = record_dict.get("step_index", 0)
            agent_id = record_dict.get("agent_id", "")

            # Find the matching routing decision
            routing_dict = record_dict.get("routing_decision", {})

            # Find the handoff checkpoint for this step
            checkpoint_id: str | None = None
            for h in result.handoffs:
                if h.target_agent == agent_id and h.source_agent != "__input__":
                    checkpoint_id = h.checkpoint_id

            # Build input payload from handoff snapshots
            input_payload: dict[str, Any] = {}
            for h in result.handoffs:
                if h.target_agent == agent_id:
                    input_payload = h.state_snapshot
                    break

            trace_steps.append(TraceStep(
                step_index=step_idx,
                agent_id=agent_id,
                input_payload=input_payload,
                routing_decision=routing_dict,
                policy_snapshot=policy_snapshot,
                provider_response=record_dict.get("provider_response"),
                handoff_checkpoint_id=checkpoint_id,
                anomalies=record_dict.get("anomalies", []),
                recovery_records=[],
                started_at=record_dict.get("started_at", ""),
                completed_at=record_dict.get("completed_at", ""),
                duration_ms=record_dict.get("duration_ms", 0.0),
            ))

        return TaskTrace(
            task_id=task.task_id,
            task_content=task.content,
            task_complexity=task.complexity_hint,
            pipeline=list(agent_pipeline),
            steps=trace_steps,
            policy_snapshot=policy_snapshot,
            total_estimated_cost_usd=result.total_estimated_cost_usd,
            total_actual_cost_usd=result.actual_cost_usd,
            total_duration_ms=result.duration_ms,
            success=result.success,
        )

    def _make_event(
        self,
        event_type: str,
        task_id: str,
        agent_id: str | None,
        payload: dict[str, Any],
    ) -> ExecutionEvent:
        return ExecutionEvent(
            event_id=str(uuid4()),
            event_type=event_type,  # type: ignore[arg-type]
            task_id=task_id,
            agent_id=agent_id,
            payload=payload,
        )

    def _make_route_event(self, decision: RoutingDecision) -> ExecutionEvent:
        """Create an ExecutionEvent from a routing decision."""
        return ExecutionEvent(
            event_id=str(uuid4()),
            event_type="route",
            task_id=decision.task_id,
            payload={
                "chosen_provider": decision.chosen_provider,
                "chosen_model": decision.chosen_model,
                "reasoning": decision.reasoning,
                "estimated_cost_usd": decision.estimated_cost_usd,
                "estimated_latency_ms": decision.estimated_latency_ms,
                "fallback_model": decision.fallback_model,
            },
            timestamp=datetime.now(timezone.utc),
        )
