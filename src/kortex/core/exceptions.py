"""Exception hierarchy for Kortex.

All exceptions inherit from KortexError so callers can catch the base
class for broad error handling or catch specific subclasses for targeted recovery.
"""


class KortexError(Exception):
    """Base exception for all Kortex errors."""


class RouterError(KortexError):
    """Raised when the task routing engine encounters an error."""


class StateError(KortexError):
    """Raised when state or checkpoint operations fail."""


class HandoffError(KortexError):
    """Raised when an agent-to-agent handoff fails."""


class ProviderError(KortexError):
    """Raised when an LLM provider call fails or is unavailable."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider returns HTTP 429 after retries are exhausted."""


class ProviderOverloadError(ProviderError):
    """Raised when a provider returns 500/502/503/504 after retries are exhausted."""


class ProviderAuthError(ProviderError):
    """Raised when a provider returns 401 or 403 (not retryable)."""


class CircuitOpenError(ProviderError):
    """Raised when the circuit breaker is open and rejecting requests."""


class CheckpointNotFoundError(StateError):
    """Raised when a requested checkpoint does not exist in the store."""


class RoutingFailedError(RouterError):
    """Raised when no suitable model can be found for a task's constraints.

    Args:
        message: Human-readable error description.
        failed_models: List of ``(model_key, reason)`` pairs for each
            candidate that was eliminated.
        closest_model: The candidate that came closest to passing all
            constraints, if any.
        suggestion: A concrete suggestion for resolving the failure.
    """

    def __init__(
        self,
        message: str,
        failed_models: list[tuple[str, str]] | None = None,
        closest_model: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failed_models: list[tuple[str, str]] = failed_models or []
        self.closest_model: str | None = closest_model
        self.suggestion: str | None = suggestion
