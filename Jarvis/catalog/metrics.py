"""How much context a decision or an engineer actually needed, in tokens.

The point of the catalog and the contracts is to keep future engineering
context small.  That is only true if it is measured, so every planner report
and every engineering job carries one of these:

    catalog_context_tokens   -- the narrow catalog context (contracts, dependents, tests)
    manifest_tokens          -- the capability manifests the planner reasoned over
    source_tokens            -- retrieved source files, when an engineer needed them
    total_tokens             -- the sum, i.e. what left for a model

Token counts are the gateway's conservative estimate (about 3.2 characters
per token), the same one the budget governor reserves on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gateway.estimate import estimate_tokens


@dataclass
class ContextMetrics:
    catalog_context_tokens: int = 0
    manifest_tokens: int = 0
    source_tokens: int = 0
    other_tokens: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.catalog_context_tokens + self.manifest_tokens + self.source_tokens + self.other_tokens

    def add(self, kind: str, text: str, note: str = "") -> int:
        tokens = estimate_tokens(text)
        if kind == "catalog":
            self.catalog_context_tokens += tokens
        elif kind == "manifest":
            self.manifest_tokens += tokens
        elif kind == "source":
            self.source_tokens += tokens
        else:
            self.other_tokens += tokens
        if note:
            self.notes.append(f"{kind}: {note} ({tokens} tokens)")
        return tokens

    def to_dict(self) -> dict[str, Any]:
        return {"catalog_context_tokens": self.catalog_context_tokens, "manifest_tokens": self.manifest_tokens,
                "source_tokens": self.source_tokens, "other_tokens": self.other_tokens, "total_tokens": self.total_tokens,
                "notes": list(self.notes)}
