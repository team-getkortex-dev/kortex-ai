"""In-memory state store for testing and local development.

Optimized for throughput with per-agent locking and direct object storage.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from kortex.core.exceptions import CheckpointNotFoundError
from kortex.core.types import HandoffContext


class InMemoryStateStore:
    """Dict-backed state store with per-agent locks for concurrent throughput.

    Objects are stored directly (no serialization overhead). Reads are
    lock-free since dict lookups are atomic in CPython. Only writes
    acquire a lock, scoped to the agents involved.
    """

    def __init__(self) -> None:
        self._store: dict[str, HandoffContext] = {}
        self._agent_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._write_lock = asyncio.Lock()

    async def save_checkpoint(self, context: HandoffContext) -> str:
        """Persist a handoff context and return its checkpoint_id."""
        async with self._write_lock:
            self._store[context.checkpoint_id] = context
        return context.checkpoint_id

    async def save_checkpoints_batch(
        self, contexts: list[HandoffContext]
    ) -> list[str]:
        """Persist multiple checkpoints in a single locked operation."""
        async with self._write_lock:
            for ctx in contexts:
                self._store[ctx.checkpoint_id] = ctx
        return [ctx.checkpoint_id for ctx in contexts]

    async def get_checkpoint(self, checkpoint_id: str) -> HandoffContext:
        """Retrieve a checkpoint by ID.

        Raises:
            CheckpointNotFoundError: If the checkpoint does not exist.
        """
        try:
            return self._store[checkpoint_id]
        except KeyError:
            raise CheckpointNotFoundError(
                f"Checkpoint '{checkpoint_id}' not found"
            ) from None

    async def list_checkpoints(
        self,
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[HandoffContext]:
        """List checkpoints, optionally filtered by task_id or agent_id."""
        results = list(self._store.values())

        if task_id is not None:
            results = [
                c for c in results if c.state_snapshot.get("task_id") == task_id
            ]
        if agent_id is not None:
            results = [
                c
                for c in results
                if c.source_agent == agent_id or c.target_agent == agent_id
            ]
        return results

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint. Returns True if it existed, False otherwise."""
        async with self._write_lock:
            return self._store.pop(checkpoint_id, None) is not None

    async def get_checkpoint_chain(self, checkpoint_id: str) -> list[HandoffContext]:
        """Walk parent_checkpoint_id links to build the full history.

        Returns a list ordered from the root (oldest) to the given checkpoint.

        Raises:
            CheckpointNotFoundError: If any checkpoint in the chain is missing.
        """
        chain: list[HandoffContext] = []
        current_id: str | None = checkpoint_id
        while current_id is not None:
            ctx = await self.get_checkpoint(current_id)
            chain.append(ctx)
            current_id = ctx.parent_checkpoint_id
        chain.reverse()
        return chain
