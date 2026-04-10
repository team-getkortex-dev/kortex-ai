"""Natural-language policy compiler for Kortex.

Converts English descriptions of routing requirements into validated
``RoutingPolicy`` objects via an LLM call. Retries up to 3 times on
parse failures.

Requires a provider connector capable of ``complete()`` calls. The LLM
is prompted with the canonical RoutingPolicy schema, a set of few-shot
examples, and the user's natural language description.

Example::

    from kortex.core.nl_policy import NLPolicyCompiler

    compiler = NLPolicyCompiler(connector)
    policy = await compiler.compile(
        "Prefer the cheapest model, never use OpenAI, max cost $0.005 per call"
    )
    print(policy.name)
    # → "nl_compiled"
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from kortex.core.policy import (
    FallbackRule,
    RoutingConstraint,
    RoutingObjective,
    RoutingPolicy,
)

logger = structlog.get_logger(component="nl_policy_compiler")

# ---------------------------------------------------------------------------
# Schema + few-shot prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert routing policy compiler for the Kortex AI framework.
Your job is to convert a natural-language description of routing requirements
into a valid JSON RoutingPolicy object.

RoutingPolicy JSON schema:
{
  "name": "<string — short identifier>",
  "description": "<string — one-line explanation>",
  "constraints": {
    "max_cost_usd": <float or null>,
    "max_latency_ms": <float or null>,
    "required_capabilities": [<string>, ...],
    "allowed_providers": [<string>, ...] or null,
    "denied_providers": [<string>, ...],
    "allowed_models": [<string>, ...] or null,
    "denied_models": [<string>, ...]
  },
  "objective": {
    "minimize": "cost" | "latency" | "none",
    "prefer_tier": "fast" | "balanced" | "powerful" | "any",
    "prefer_provider": <string or null>
  },
  "fallback": {
    "strategy": "next_cheapest" | "next_fastest" | "same_tier" | "explicit",
    "explicit_model_identity": <string or null>
  },
  "budget_ceiling_usd": <float or null>
}

RULES:
- Required capabilities must be from: reasoning, code_generation,
  content_generation, vision, analysis, math, function_calling, data_analysis
- "prefer_tier" controls which tier (fast / balanced / powerful) to bias toward
- Set "minimize" = "cost" when the user wants cheap, "latency" when fast
- Set "denied_providers" to exclude providers the user doesn't want
- Only output valid JSON — no markdown, no prose

EXAMPLES:

User: "I need the cheapest option available, latency doesn't matter"
{
  "name": "cheapest",
  "description": "Minimize cost above all else.",
  "constraints": { "max_cost_usd": null, "max_latency_ms": null,
                   "required_capabilities": [], "denied_providers": [] },
  "objective": { "minimize": "cost", "prefer_tier": "fast",
                 "prefer_provider": null },
  "fallback": { "strategy": "next_cheapest", "explicit_model_identity": null },
  "budget_ceiling_usd": null
}

User: "Fast responses only — must complete in under 500ms, I'm willing to pay more"
{
  "name": "low_latency",
  "description": "Sub-500ms latency required.",
  "constraints": { "max_cost_usd": null, "max_latency_ms": 500.0,
                   "required_capabilities": [], "denied_providers": [] },
  "objective": { "minimize": "latency", "prefer_tier": "fast",
                 "prefer_provider": null },
  "fallback": { "strategy": "next_fastest", "explicit_model_identity": null },
  "budget_ceiling_usd": null
}

User: "Use only Anthropic models, prefer the most capable one for complex reasoning tasks"
{
  "name": "anthropic_quality",
  "description": "Anthropic-only, quality-first.",
  "constraints": { "max_cost_usd": null, "max_latency_ms": null,
                   "required_capabilities": ["reasoning"],
                   "allowed_providers": ["anthropic"], "denied_providers": [] },
  "objective": { "minimize": "none", "prefer_tier": "powerful",
                 "prefer_provider": "anthropic" },
  "fallback": { "strategy": "same_tier", "explicit_model_identity": null },
  "budget_ceiling_usd": null
}
"""


# ---------------------------------------------------------------------------
# NLPolicyCompiler
# ---------------------------------------------------------------------------


class NLPolicyCompiler:
    """Converts natural-language routing requirements into RoutingPolicy objects.

    Args:
        connector: A provider connector with a ``complete(prompt, model)`` method.
        model: The model to use for compilation.
        max_retries: Maximum number of parse attempts on LLM failure.
    """

    def __init__(
        self,
        connector: Any,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
    ) -> None:
        self._connector = connector
        self._model = model
        self._max_retries = max_retries
        self._log = structlog.get_logger(component="nl_policy_compiler")

    async def compile(self, description: str) -> RoutingPolicy:
        """Compile a natural-language description into a RoutingPolicy.

        Args:
            description: English description of the routing requirements.

        Returns:
            A validated RoutingPolicy.

        Raises:
            ValueError: If the LLM output cannot be parsed into a valid policy
                after ``max_retries`` attempts.
        """
        last_error: str = ""
        prompt = f"{_SYSTEM_PROMPT}\n\nUser: \"{description}\"\n\nJSON:"

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._connector.complete(prompt, self._model)
                raw_text = response.content if hasattr(response, "content") else str(response)
                policy = self._parse(raw_text)
                self._log.info(
                    "nl_policy_compiled",
                    description=description[:60],
                    policy_name=policy.name,
                    attempt=attempt,
                )
                return policy
            except Exception as exc:
                last_error = str(exc)
                self._log.warning(
                    "nl_policy_compile_retry",
                    attempt=attempt,
                    error=last_error,
                )

        raise ValueError(
            f"Failed to compile policy after {self._max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _parse(self, raw_text: str) -> RoutingPolicy:
        """Parse the LLM output into a RoutingPolicy.

        Args:
            raw_text: Raw LLM response text.

        Returns:
            Validated RoutingPolicy.

        Raises:
            ValueError: If the JSON is invalid or fails validation.
        """
        # Strip markdown fences if present
        text = raw_text.strip()
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            text = json_match.group(1).strip()

        # Try to find a JSON object in the text
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            text = brace_match.group(0)

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {exc}. Raw: {raw_text[:200]}") from exc

        return self._dict_to_policy(data)

    def _dict_to_policy(self, data: dict[str, Any]) -> RoutingPolicy:
        """Convert a parsed dict to a validated RoutingPolicy.

        Args:
            data: Parsed JSON dict.

        Returns:
            RoutingPolicy.

        Raises:
            ValueError: If any field has an invalid value.
        """
        c = data.get("constraints", {})
        o = data.get("objective", {})
        f = data.get("fallback", {})

        minimize = o.get("minimize", "cost")
        if minimize not in ("cost", "latency", "none"):
            minimize = "cost"

        prefer_tier = o.get("prefer_tier", "any")
        if prefer_tier not in ("fast", "balanced", "powerful", "any"):
            prefer_tier = "any"

        fallback_strategy = f.get("strategy", "next_cheapest")
        if fallback_strategy not in ("next_cheapest", "next_fastest", "same_tier", "explicit"):
            fallback_strategy = "next_cheapest"

        constraint = RoutingConstraint(
            max_cost_usd=c.get("max_cost_usd") or None,
            max_latency_ms=c.get("max_latency_ms") or None,
            required_capabilities=c.get("required_capabilities", []),
            allowed_providers=c.get("allowed_providers") or None,
            denied_providers=c.get("denied_providers", []),
            allowed_models=c.get("allowed_models") or None,
            denied_models=c.get("denied_models", []),
        )
        objective = RoutingObjective(
            minimize=minimize,  # type: ignore[arg-type]
            prefer_tier=prefer_tier,  # type: ignore[arg-type]
            prefer_provider=o.get("prefer_provider") or None,
        )
        fallback = FallbackRule(
            strategy=fallback_strategy,  # type: ignore[arg-type]
            explicit_model_identity=f.get("explicit_model_identity") or None,
        )

        return RoutingPolicy(
            name=data.get("name", "nl_compiled"),
            description=data.get("description", ""),
            constraints=constraint,
            objective=objective,
            fallback=fallback,
            budget_ceiling_usd=data.get("budget_ceiling_usd") or None,
        )

    def compile_from_dict(self, data: dict[str, Any]) -> RoutingPolicy:
        """Compile a RoutingPolicy directly from a dict (no LLM call).

        Useful for testing and for programmatic policy generation.

        Args:
            data: Dict matching the RoutingPolicy JSON schema.

        Returns:
            Validated RoutingPolicy.
        """
        return self._dict_to_policy(data)
