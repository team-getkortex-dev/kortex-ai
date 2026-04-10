"""Canonical capability vocabulary for Kortex.

Defines the single source of truth for capability strings used across
providers, agents, tasks, and adapters. Ensures that capability filtering
in the router actually works — no more silent mismatches between what
providers advertise and what tasks require.
"""

from __future__ import annotations

import enum
from difflib import get_close_matches


class Capability(enum.Enum):
    """Canonical capability tags.

    Every capability string in the system must be one of these values.
    Providers advertise them, tasks require them, the router filters on them.
    """

    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CODE_GENERATION = "code_generation"
    CONTENT_GENERATION = "content_generation"
    VISION = "vision"
    AUDIO = "audio"
    QUALITY_ASSURANCE = "quality_assurance"
    DATA_PROCESSING = "data_processing"
    PLANNING = "planning"
    RESEARCH = "research"
    TESTING = "testing"


# All valid capability strings for fast lookup
_VALID: set[str] = {c.value for c in Capability}


# Common aliases that map to canonical values
CAPABILITY_ALIASES: dict[str, str] = {
    "coding": "code_generation",
    "code": "code_generation",
    "writing": "content_generation",
    "review": "quality_assurance",
    "qa": "quality_assurance",
    "design": "planning",
    "coordination": "planning",
    "manage": "planning",
}


def validate_capabilities(caps: list[str]) -> list[str]:
    """Validate that all capability strings are canonical.

    Args:
        caps: List of capability strings to validate.

    Returns:
        The input list unchanged if all values are valid.

    Raises:
        ValueError: If any capability string is not a valid Capability value.
            The error message lists all invalid values and suggests close
            matches when available.
    """
    invalid = [c for c in caps if c not in _VALID]
    if not invalid:
        return caps

    parts: list[str] = []
    for bad in invalid:
        matches = get_close_matches(bad, list(_VALID), n=1, cutoff=0.5)
        if bad in CAPABILITY_ALIASES:
            parts.append(
                f"  '{bad}' is not canonical — use alias resolution "
                f"via normalize_capabilities() to map it to "
                f"'{CAPABILITY_ALIASES[bad]}'"
            )
        elif matches:
            parts.append(f"  '{bad}' is not valid. Did you mean '{matches[0]}'?")
        else:
            parts.append(f"  '{bad}' is not valid.")

    valid_list = ", ".join(sorted(_VALID))
    raise ValueError(
        f"Invalid capabilities: {invalid}. "
        f"Details:\n" + "\n".join(parts) + "\n"
        f"Valid capabilities: [{valid_list}]"
    )


def normalize_capabilities(caps: list[str]) -> list[str]:
    """Resolve aliases and validate capability strings.

    Args:
        caps: List of capability strings, possibly containing aliases.

    Returns:
        Deduplicated list of canonical capability strings.

    Raises:
        ValueError: If any capability is neither canonical nor a known alias.
    """
    result: list[str] = []
    seen: set[str] = set()

    for cap in caps:
        canonical = CAPABILITY_ALIASES.get(cap, cap)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)

    validate_capabilities(result)
    return result
