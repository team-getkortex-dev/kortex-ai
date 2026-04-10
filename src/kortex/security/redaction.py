"""API key redaction utilities for Kortex.

All outbound text paths — traces, error messages, CLI output, logs —
should pass through :func:`scan_and_redact` so that API keys never appear
in plain-text output.  The redaction is best-effort: it catches all common
API key patterns but is **not** a substitute for keeping secrets out of
code in the first place.

Supported patterns
------------------
- ``sk-[A-Za-z0-9]{32,}``  — OpenAI / generic "sk-" keys
- ``gsk_[A-Za-z0-9]{32,}`` — Groq API keys
- ``csk_[A-Za-z0-9]{32,}`` — Cerebras API keys
- ``Bearer [A-Za-z0-9._-]{20,}`` — Authorization header values
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Patterns (compiled once at import time)
# ---------------------------------------------------------------------------

_PATTERNS: list[re.Pattern[str]] = [
    # OpenAI / generic "sk-" keys  (also catches sk-ant-... Anthropic keys)
    re.compile(r"sk-[A-Za-z0-9_\-]{32,}"),
    # Groq keys start with gsk_
    re.compile(r"gsk_[A-Za-z0-9]{32,}"),
    # Cerebras keys start with csk_
    re.compile(r"csk_[A-Za-z0-9]{32,}"),
    # HTTP Authorization header bearer tokens
    re.compile(r"Bearer [A-Za-z0-9._\-]{20,}"),
]


def redact_api_key(key: str) -> str:
    """Mask an API key, preserving the first 3 and last 3 characters.

    Keys shorter than 9 characters are replaced entirely with ``[REDACTED]``.

    Args:
        key: The raw API key string.

    Returns:
        A masked string like ``"sk-...xyz"`` or ``"[REDACTED]"``.

    Examples::

        >>> redact_api_key("sk-abcdefghijklmnopqrstuvwxyz123456")
        'sk-...456'
        >>> redact_api_key("short")
        '[REDACTED]'
    """
    if len(key) < 9:
        return "[REDACTED]"
    return f"{key[:3]}...{key[-3:]}"


def scan_and_redact(text: str) -> str:
    """Search ``text`` for API key patterns and replace each match with a
    redacted form.

    Replaces the full match with ``<prefix>...<suffix>`` (first 6 chars of
    match, ``...``, last 3 chars of match) so the output is unambiguously
    redacted while remaining identifiable.

    Args:
        text: Any string that might contain API keys (log lines, error
            messages, JSON blobs, etc.).

    Returns:
        The input text with all detected API keys replaced by redacted forms.

    Examples::

        >>> scan_and_redact("key=sk-abcdefghijklmnopqrstuvwxyz123456")
        'key=sk-abc...456'
    """
    for pattern in _PATTERNS:
        text = pattern.sub(_redact_match, text)
    return text


def _redact_match(m: re.Match[str]) -> str:
    """Replace a regex match with a redacted representation."""
    raw = m.group(0)
    if len(raw) < 9:
        return "[REDACTED]"
    return f"{raw[:6]}...{raw[-3:]}"
