"""Abstract state store protocol for Kortex.

Defines the interface that all storage backends (memory, Redis, SQLite) must
implement for checkpoint persistence and retrieval.
"""

from __future__ import annotations

from typing import Protocol

from kortex.core.types import HandoffContext


class StateStore(Protocol):
    """Protocol for pluggable checkpoint storage backends."""

    async def save_checkpoint(self, context: HandoffContext) -> str:
        """Persist a handoff context and return its checkpoint_id."""
        ...

    async def get_checkpoint(self, checkpoint_id: str) -> HandoffContext:
        """Retrieve a checkpoint by ID.

        Raises:
            CheckpointNotFoundError: If the checkpoint does not exist.
        """
        ...

    async def list_checkpoints(
        self,
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[HandoffContext]:
        """List checkpoints, optionally filtered by task_id or agent_id.

        Args:
            task_id: If provided, filter to checkpoints whose state_snapshot
                contains a "task_id" key matching this value.
            agent_id: If provided, filter to checkpoints where source_agent
                or target_agent matches this value.
        """
        ...

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint. Returns True if it existed, False otherwise."""
        ...

    async def save_checkpoints_batch(
        self, contexts: list[HandoffContext]
    ) -> list[str]:
        """Persist multiple handoff contexts in a single batch.

        Returns:
            List of checkpoint_ids in the same order as inputs.
        """
        ...

    async def get_checkpoint_chain(self, checkpoint_id: str) -> list[HandoffContext]:
        """Walk parent_checkpoint_id links to build the full history.

        Returns a list ordered from the root (oldest) to the given checkpoint.

        Raises:
            CheckpointNotFoundError: If any checkpoint in the chain is missing.
        """
        ...
