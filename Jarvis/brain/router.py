"""Routes a request to the model tier that should handle it.

The routing decision is deliberately cheap and deterministic -- a keyword match,
not a model call.  Asking a language model which model should answer costs a
model call to save a model call, and on a machine where loading the 7B coder
takes 45 seconds, guessing wrong occasionally is far cheaper than paying that
round trip every time.

The important change from the earlier version is what the heavy tier *is*.  It
used to be ``BUILD_REMOTE``, so anything resembling development work required
remote compute and Jarvis was useless without it.  The heavy tier is now
:attr:`BrainTier.BUILD_LOCAL`, served by the local coder model.  Remote and
cloud tiers still exist, but as opt-in escalations rather than as the only way
to get work done -- deleting every credential on the machine must leave
autonomous development fully functional.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from brain.tiers import ModelCatalog, ModelProbe, ModelTier


class BrainTier(str, Enum):
    """The tiers the router can select between."""

    FAST_LOCAL = "FAST_LOCAL"
    BUILD_LOCAL = "BUILD_LOCAL"
    #: A stronger model on a machine the user controls.  Opt-in.
    SELF_HOSTED = "SELF_HOSTED"
    #: A paid API.  Opt-in, never a default, never automatic.
    EXPERT_CLOUD = "EXPERT_CLOUD"

    def to_model_tier(self) -> ModelTier:
        return ModelTier(self.value)


@dataclass(frozen=True)
class RouteDecision:
    tier: BrainTier
    reason: str
    #: True when the request should become a durable project rather than a
    #: single reply -- building software is not a chat turn.
    is_project: bool = False


class BrainUnavailable(RuntimeError):
    """The selected tier cannot serve this request right now."""


#: Kept under its old name so existing callers and tests still catch it.
RemoteBrainUnavailable = BrainUnavailable


class BrainRouter:
    """Picks a tier for each request, and can answer simple ones directly."""

    #: Phrases that mean "build or change software".  German and English,
    #: because this user works in both.
    BUILD_HINTS = (
        "implementiere",
        "implementieren",
        "programmiere",
        "schreibe code",
        "schreib code",
        "ändere den code",
        "ändere code",
        "aendere den code",
        "repository",
        "repo",
        "fixe den bug",
        "behebe den bug",
        "debugge",
        "patch",
        "baue eine app",
        "baue mir",
        "entwickle",
        "erstelle eine app",
        "erstelle ein programm",
        "schreibe ein programm",
        "schreib mir ein",
        "refactor",
        "refaktor",
        "build an app",
        "build me",
        "build a",
        "write a program",
        "write a script",
        "write code",
        "implement",
        "modify the code",
        "fix the bug",
        "fix a bug",
        "debug this",
        "run the tests",
        "add a test",
        "self-develop",
        "self develop",
        "create a tool",
        "create a script",
    )

    #: Phrases that mean "acquire something you cannot currently do".
    CAPABILITY_HINTS = (
        "lerne",
        "bring dir bei",
        "kannst du",
        "learn how to",
        "teach yourself",
        "acquire the ability",
        "new capability",
        "neue faehigkeit",
        "neue fähigkeit",
    )

    def __init__(
        self,
        *,
        catalog: ModelCatalog | None = None,
        probe: ModelProbe | None = None,
        fast_brain: Any | None = None,
        build_brain: Any | None = None,
    ) -> None:
        self.catalog = catalog or ModelCatalog()
        self.probe = probe or ModelProbe(self.catalog)
        self._overrides: dict[BrainTier, Any] = {}
        if fast_brain is not None:
            self._overrides[BrainTier.FAST_LOCAL] = fast_brain
        if build_brain is not None:
            self._overrides[BrainTier.BUILD_LOCAL] = build_brain
        self._providers: dict[BrainTier, Any] = {}

    # -- routing ---------------------------------------------------------

    def route(self, message: str) -> RouteDecision:
        normalized = f" {message.strip().lower()} "

        for hint in self.CAPABILITY_HINTS:
            if hint in normalized:
                return RouteDecision(
                    BrainTier.BUILD_LOCAL, f"capability acquisition intent matched: {hint.strip()}", is_project=True
                )

        for hint in self.BUILD_HINTS:
            if hint in normalized:
                return RouteDecision(
                    BrainTier.BUILD_LOCAL, f"development intent matched: {hint.strip()}", is_project=True
                )

        return RouteDecision(BrainTier.FAST_LOCAL, "conversational request suits the fast local model")

    # -- providers -------------------------------------------------------

    def brain(self, tier: BrainTier) -> Any:
        """The provider for a tier, or an explanation of why it is unusable."""

        if tier in self._overrides:
            return self._overrides[tier]

        spec = self.catalog.get(tier.to_model_tier())
        if not spec.enabled:
            raise BrainUnavailable(
                f"{tier.value} is disabled in the model catalog. "
                "Local tiers are enabled by default; paid tiers must be turned on deliberately."
            )
        if not spec.configured:
            raise BrainUnavailable(f"{tier.value} has no model configured.")

        if tier not in self._providers:
            from brain.providers import provider_for_spec

            self._providers[tier] = provider_for_spec(spec)
        return self._providers[tier]

    def require_build_brain(self) -> Any:
        """The development model, verified to actually work.

        Verified rather than assumed: reporting a model as ready and then
        failing on the first generation is the specific dishonesty this system
        is meant to avoid.
        """

        health = self.probe.probe(ModelTier.BUILD_LOCAL)
        if not health.online:
            raise BrainUnavailable(f"BUILD_LOCAL is not usable: {health.summary()}")
        return self.brain(BrainTier.BUILD_LOCAL)

    # -- direct answers --------------------------------------------------

    def respond(self, message: str) -> tuple[str, RouteDecision]:
        """Answer a conversational request directly.

        Requests routed to BUILD_LOCAL are *projects*, not chat turns, so this
        deliberately refuses them rather than producing an essay about what it
        would do.  The caller is expected to start a project instead.
        """

        decision = self.route(message)
        if decision.is_project:
            raise BrainUnavailable(
                "This request describes work to be done, not a question to answer. Start a project for it."
            )
        brain = self.brain(BrainTier.FAST_LOCAL)
        max_tokens = int(os.getenv("JARVIS_FAST_MAX_TOKENS", "768"))
        return self._generate(brain, message, max_tokens=max_tokens), decision

    @staticmethod
    def _generate(brain: Any, prompt: str, *, max_tokens: int) -> str:
        if hasattr(brain, "generate"):
            return str(brain.generate(prompt, max_tokens=max_tokens, temperature=0.3, top_p=0.9))
        return str(brain.think(prompt, max_tokens=max_tokens))

    # -- status ----------------------------------------------------------

    def status(self, *, force: bool = False) -> dict[str, Any]:
        health = self.probe.probe_all(force=force)
        return {
            "tiers": {tier.value: item.to_dict() for tier, item in health.items()},
            "fast_tier": BrainTier.FAST_LOCAL.value,
            "fast_model": self.catalog.get(ModelTier.FAST_LOCAL).model,
            "fast_online": health[ModelTier.FAST_LOCAL].online,
            "build_tier": BrainTier.BUILD_LOCAL.value,
            "build_model": self.catalog.get(ModelTier.BUILD_LOCAL).model,
            "build_online": health[ModelTier.BUILD_LOCAL].online,
            "paid_tiers_enabled": [tier.value for tier in self.catalog.paid_tiers_enabled()],
        }
