"""Cost estimation before a paid request is made.

Estimates are deliberately conservative.  Token counts come from a
characters-per-token heuristic that overestimates for English and German
prose, and the governor multiplies the money figure by a safety factor before
reserving it.  An estimate that turns out high releases budget on settlement;
an estimate that turns out low is what the safety factor is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gateway.config import Pricing

#: Prose averages ~4 characters per token in English and ~3.3 in German; code
#: and JSON run denser.  3.2 overestimates all three, which is the point.
CHARS_PER_TOKEN = 3.2


def estimate_tokens(text: str) -> int:
    text = str(text or "")
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN) + 1)


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    tool_cost_eur: float
    pricing: Pricing
    currency: str = "EUR"

    @property
    def estimated_eur(self) -> float:
        fresh = max(0, self.input_tokens - self.cached_input_tokens)
        return round(
            fresh * self.pricing.input_per_m / 1_000_000
            + self.cached_input_tokens * self.pricing.cached_input_per_m / 1_000_000
            + self.output_tokens * self.pricing.output_per_m / 1_000_000
            + self.tool_cost_eur,
            6,
        )

    @property
    def is_free(self) -> bool:
        return self.estimated_eur <= 0.0

    def range_eur(self, *, low_factor: float = 0.6, high_factor: float = 1.5) -> tuple[float, float]:
        """An owner-facing bracket: output length is the uncertain part."""

        return round(self.estimated_eur * low_factor, 4), round(self.estimated_eur * high_factor, 4)

    def to_dict(self) -> dict[str, Any]:
        low, high = self.range_eur()
        return {
            "input_tokens": self.input_tokens, "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens, "tool_cost_eur": self.tool_cost_eur,
            "estimated_eur": self.estimated_eur, "range_eur": [low, high],
            "pricing_confirmed": self.pricing.confirmed, "currency": self.currency,
        }


def estimate_cost(
    *,
    prompt: str,
    system: str = "",
    expected_output_tokens: int,
    pricing: Pricing,
    cached_prefix: str = "",
    tool_cost_eur: float = 0.0,
) -> CostEstimate:
    input_tokens = estimate_tokens(system) + estimate_tokens(prompt)
    cached = min(input_tokens, estimate_tokens(cached_prefix)) if cached_prefix else 0
    return CostEstimate(
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=max(0, int(expected_output_tokens)),
        tool_cost_eur=max(0.0, float(tool_cost_eur)),
        pricing=pricing,
    )


def actual_cost(usage: dict[str, Any], pricing: Pricing) -> float:
    """Money for what the provider reports it billed."""

    fresh = max(0, int(usage.get("input_tokens", 0) or 0) - int(usage.get("cached_input_tokens", 0) or 0))
    return round(
        fresh * pricing.input_per_m / 1_000_000
        + int(usage.get("cached_input_tokens", 0) or 0) * pricing.cached_input_per_m / 1_000_000
        + int(usage.get("output_tokens", 0) or 0) * pricing.output_per_m / 1_000_000
        + float(usage.get("tool_cost_eur", 0.0) or 0.0),
        6,
    )
