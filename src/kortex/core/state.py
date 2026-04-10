"""Stateful handoff manager for Kortex.

Manages context preservation across agent boundaries with checkpointing,
compressed summaries, and rollback support.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Literal
from uuid import uuid4

import structlog

from kortex.core.exceptions import StateError
from kortex.core.types import ExecutionEvent, HandoffContext
from kortex.store.base import StateStore
from kortex.store.memory import InMemoryStateStore

_TRUNCATE_THRESHOLD = 500
_TRUNCATE_KEEP = 200


def _compress_snapshot(snapshot: dict[str, Any]) -> str:
    """Build a token-efficient summary of a state snapshot.

    Extracts keys and truncates any value whose string representation
    exceeds 500 characters.
    """
    parts: list[str] = []
    for key, value in snapshot.items():
        val_str = str(value)
        if len(val_str) > _TRUNCATE_THRESHOLD:
            val_str = val_str[:_TRUNCATE_KEEP] + "... [truncated]"
        parts.append(f"{key}={val_str}")
    return "; ".join(parts)


class StateManager:
    """High-level state and checkpoint manager.

    Supports async lifecycle management. For backends that require
    connection setup (Redis, SQLite), call ``start()`` before use
    and ``stop()`` when done, or use as an async context manager::

        async with StateManager(store=SQLiteStateStore()) as mgr:
            await mgr.handoff(...)

    The in-memory backend works without calling ``start()``.

    Args:
        store: The storage backend. Defaults to InMemoryStateStore.
    """

    def __init__(self, store: StateStore | None = None) -> None:
        self._store: StateStore = store or InMemoryStateStore()
        self._log = structlog.get_logger(component="state_manager")
        self._started = False
        self._stopped = False

    async def start(self) -> None:
        """Connect the underlying store if it requires initialization.

        Safe to call multiple times (idempotent). For InMemoryStateStore this
        is a no-op. For Redis/SQLite it calls ``connect()``.
        """
        if self._started:
            return
        if hasattr(self._store, "connect"):
            await self._store.connect()  # type: ignore[attr-defined]
        self._started = True
        self._stopped = False
        self._log.info("state_manager_started", store=type(self._store).__name__)

    async def stop(self) -> None:
        """Disconnect the underlying store.

        Safe to call multiple times (idempotent). After calling ``stop()``,
        further operations will raise ``StateError``.
        """
        if self._stopped:
            return
        if hasattr(self._store, "disconnect"):
            await self._store.disconnect()  # type: ignore[attr-defined]
        self._stopped = True
        self._started = False
        self._log.info("state_manager_stopped", store=type(self._store).__name__)

    async def __aenter__(self) -> StateManager:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()

    def _ensure_running(self) -> None:
        """Raise StateError if the manager has been stopped."""
        if self._stopped:
            raise StateError(
                "StateManager has been stopped. Create a new instance or call start() again."
            )

    async def handoff(
        self,
        source_agent: str,
        target_agent: str,
        state_snapshot: dict[str, Any],
        parent_checkpoint_id: str | None = None,
    ) -> HandoffContext:
        """Create a handoff between agents, persisting the checkpoint.

        Args:
            source_agent: The agent handing off.
            target_agent: The agent receiving the handoff.
            state_snapshot: The context being passed.
            parent_checkpoint_id: Previous checkpoint for rollback chains.

        Returns:
            The persisted HandoffContext.

        Raises:
            StateError: If the manager has been stopped.
        """
        self._ensure_running()
        context = HandoffContext(
            handoff_id=str(uuid4()),
            source_agent=source_agent,
            target_agent=target_agent,
            state_snapshot=state_snapshot,
            compressed_summary=_compress_snapshot(state_snapshot),
            checkpoint_id=str(uuid4()),
            parent_checkpoint_id=parent_checkpoint_id,
            created_at=datetime.now(timezone.utc),
        )

        await self._store.save_checkpoint(context)

        self._log.info(
            "handoff",
            handoff_id=context.handoff_id,
            source=source_agent,
            target=target_agent,
            checkpoint_id=context.checkpoint_id,
            parent_checkpoint_id=parent_checkpoint_id,
        )

        return context

    async def batch_handoff(
        self,
        handoffs: list[tuple[str, str, dict[str, Any], str | None]],
    ) -> list[HandoffContext]:
        """Create multiple handoffs in a single batch operation.

        Each tuple contains (source_agent, target_agent, state_snapshot,
        parent_checkpoint_id). All HandoffContext objects are constructed
        first, then saved in a single batch call to the store.

        Args:
            handoffs: List of (source, target, snapshot, parent_id) tuples.

        Returns:
            List of persisted HandoffContext objects.

        Raises:
            StateError: If the manager has been stopped.
        """
        self._ensure_running()
        contexts: list[HandoffContext] = []
        for source_agent, target_agent, state_snapshot, parent_id in handoffs:
            ctx = HandoffContext(
                handoff_id=str(uuid4()),
                source_agent=source_agent,
                target_agent=target_agent,
                state_snapshot=state_snapshot,
                compressed_summary=_compress_snapshot(state_snapshot),
                checkpoint_id=str(uuid4()),
                parent_checkpoint_id=parent_id,
                created_at=datetime.now(timezone.utc),
            )
            contexts.append(ctx)

        await self._store.save_checkpoints_batch(contexts)

        self._log.info(
            "batch_handoff",
            count=len(contexts),
            first_checkpoint=contexts[0].checkpoint_id if contexts else None,
        )

        return contexts

    def _create_handoff_context(
        self,
        source_agent: str,
        target_agent: str,
        state_snapshot: dict[str, Any],
        parent_checkpoint_id: str | None = None,
    ) -> HandoffContext:
        """Create a HandoffContext with pre-assigned IDs without persisting it.

        Use this when you want to pre-assign checkpoint IDs before saving, so
        that dependent contexts can reference each other's IDs before any
        database writes occur.  Pass the resulting contexts to
        ``execute_handoffs_parallel()`` to persist them efficiently.

        Args:
            source_agent: The agent handing off.
            target_agent: The agent receiving the handoff.
            state_snapshot: The context being passed.
            parent_checkpoint_id: Previous checkpoint for rollback chains.

        Returns:
            An unpersisted HandoffContext with a fresh UUID for checkpoint_id.
        """
        return HandoffContext(
            handoff_id=str(uuid4()),
            source_agent=source_agent,
            target_agent=target_agent,
            state_snapshot=state_snapshot,
            compressed_summary=_compress_snapshot(state_snapshot),
            checkpoint_id=str(uuid4()),
            parent_checkpoint_id=parent_checkpoint_id,
            created_at=datetime.now(timezone.utc),
        )

    async def execute_handoffs_parallel(
        self, handoffs: list[HandoffContext]
    ) -> list[HandoffContext]:
        """Persist a batch of handoffs, running independent ones in parallel.

        Uses ``DAGAnalyzer`` to identify which handoffs have no intra-batch
        dependencies.  Independent handoffs within the same group are saved
        concurrently via ``asyncio.gather``; dependent groups run sequentially.

        Args:
            handoffs: Pre-created ``HandoffContext`` objects to persist.

        Returns:
            The same list of ``HandoffContext`` objects (now persisted).

        Raises:
            StateError: If the manager has been stopped.
        """
        from kortex.state.dag_analyzer import DAGAnalyzer

        self._ensure_running()
        if not handoffs:
            return []

        analyzer = DAGAnalyzer()
        groups = analyzer.get_execution_groups(handoffs)

        for group in groups:
            await asyncio.gather(
                *[self._store.save_checkpoint(ctx) for ctx in group]
            )

        self._log.info(
            "parallel_handoffs_saved",
            total=len(handoffs),
            groups=len(groups),
        )
        return handoffs

    def _emit_event(
        self,
        event_type: str,
        context: HandoffContext,
    ) -> ExecutionEvent:
        """Build an ExecutionEvent from a handoff context."""
        return ExecutionEvent(
            event_id=str(uuid4()),
            event_type=event_type,  # type: ignore[arg-type]
            task_id=context.state_snapshot.get("task_id", ""),
            agent_id=context.target_agent,
            payload={
                "checkpoint_id": context.checkpoint_id,
                "source_agent": context.source_agent,
                "target_agent": context.target_agent,
            },
            timestamp=datetime.now(timezone.utc),
        )

    async def rollback(self, checkpoint_id: str) -> HandoffContext:
        """Retrieve a checkpoint for state restoration.

        Args:
            checkpoint_id: The checkpoint to roll back to.

        Returns:
            The HandoffContext at that checkpoint.

        Raises:
            CheckpointNotFoundError: If the checkpoint does not exist.
            StateError: If the manager has been stopped.
        """
        self._ensure_running()
        context = await self._store.get_checkpoint(checkpoint_id)

        self._log.info(
            "rollback",
            checkpoint_id=checkpoint_id,
            source=context.source_agent,
            target=context.target_agent,
        )

        return context

    async def get_history(self, checkpoint_id: str) -> list[HandoffContext]:
        """Return the full checkpoint chain from root to the given checkpoint.

        Args:
            checkpoint_id: The leaf checkpoint to trace back from.

        Returns:
            List of HandoffContexts ordered root-first.

        Raises:
            CheckpointNotFoundError: If any checkpoint in the chain is missing.
            StateError: If the manager has been stopped.
        """
        self._ensure_running()
        return await self._store.get_checkpoint_chain(checkpoint_id)

    async def get_latest(self, agent_id: str) -> HandoffContext | None:
        """Return the most recent checkpoint involving the given agent.

        Args:
            agent_id: The agent to look up.

        Returns:
            The most recent HandoffContext, or None if no checkpoints exist.

        Raises:
            StateError: If the manager has been stopped.
        """
        self._ensure_running()
        checkpoints = await self._store.list_checkpoints(agent_id=agent_id)
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda c: c.created_at)

    @classmethod
    def create(
        cls,
        backend: Literal["memory", "redis", "sqlite"] = "memory",
        **kwargs: Any,
    ) -> StateManager:
        """Factory method to create a StateManager with the specified backend.

        Args:
            backend: Storage backend to use.
            **kwargs: Passed to the backend constructor.
                - redis: redis_url, key_prefix, ttl_seconds
                - sqlite: db_path
                - memory: (none)

        Returns:
            A configured StateManager instance.
        """
        if backend == "memory":
            return cls(store=InMemoryStateStore())
        if backend == "redis":
            from kortex.store.redis import RedisStateStore

            return cls(store=RedisStateStore(**kwargs))  # type: ignore[arg-type]
        if backend == "sqlite":
            from kortex.store.sqlite import SQLiteStateStore

            return cls(store=SQLiteStateStore(**kwargs))  # type: ignore[arg-type]
        raise ValueError(f"Unknown backend: {backend!r}. Use 'memory', 'redis', or 'sqlite'.")

    @classmethod
    async def create_and_connect(
        cls,
        backend: Literal["memory", "redis", "sqlite"] = "memory",
        **kwargs: Any,
    ) -> StateManager:
        """Create a StateManager and connect its store, ready for immediate use.

        This is the recommended factory for non-memory backends. Equivalent to
        calling ``create()`` followed by ``start()``.

        Args:
            backend: Storage backend to use.
            **kwargs: Passed to the backend constructor.

        Returns:
            A connected, ready-to-use StateManager instance.
        """
        manager = cls.create(backend=backend, **kwargs)
        await manager.start()
        return manager
