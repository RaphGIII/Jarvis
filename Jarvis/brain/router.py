from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from brain.providers import (
    BrainProvider,
    make_brain_provider_from_env,
    make_build_remote_brain_provider_from_env,
)


class BrainTier(str, Enum):
    FAST_LOCAL = "FAST_LOCAL"
    BUILD_REMOTE = "BUILD_REMOTE"


@dataclass(frozen=True)
class RouteDecision:
    tier: BrainTier
    reason: str


class RemoteBrainUnavailable(RuntimeError):
    pass


class BrainRouter:
    """Cheap deterministic V1 router between local interaction and heavy build work."""

    BUILD_HINTS = (
        "implementiere",
        "implementieren",
        "programmiere",
        "schreibe code",
        "ändere den code",
        "ändere code",
        "repository",
        " repo ",
        "fixe den bug",
        "behebe den bug",
        "debugge",
        "patch",
        "baue eine app",
        "baue mir eine app",
        "entwickle eine app",
        "erstelle eine app",
        "build an app",
        "build the app",
        "implement",
        "modify the code",
        "fix the bug",
        "debug this",
        "repository",
        "run the tests",
        "self-develop",
        "self develop",
    )

    def __init__(
        self,
        *,
        fast_brain: BrainProvider | None = None,
        remote_brain: BrainProvider | None = None,
        remote_enabled: bool | None = None,
    ) -> None:
        self.fast_brain = fast_brain or make_brain_provider_from_env()

        if remote_enabled is None:
            remote_enabled = os.getenv(
                "JARVIS_BUILD_REMOTE_ENABLED", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}

        self.remote_enabled = bool(remote_enabled)
        self.remote_brain = remote_brain
        self.remote_load_error = ""

        if self.remote_enabled and self.remote_brain is None:
            try:
                self.remote_brain = (
                    make_build_remote_brain_provider_from_env()
                )
            except Exception as exc:
                self.remote_load_error = str(exc)

    def route(self, message: str) -> RouteDecision:
        normalized = f" {message.strip().lower()} "

        for hint in self.BUILD_HINTS:
            if hint in normalized:
                return RouteDecision(
                    BrainTier.BUILD_REMOTE,
                    f"complex build/development intent matched: {hint.strip()}",
                )

        return RouteDecision(
            BrainTier.FAST_LOCAL,
            "normal interaction fits the local brain",
        )

    def require_build_brain(self) -> BrainProvider:
        if not self.remote_enabled:
            raise RemoteBrainUnavailable(
                "This task requires BUILD_REMOTE, "
                "but remote compute is disabled."
            )

        if self.remote_brain is None:
            detail = (
                f" Configuration error: {self.remote_load_error}"
                if self.remote_load_error
                else ""
            )

            raise RemoteBrainUnavailable(
                "BUILD_REMOTE is enabled but no remote brain "
                f"is available.{detail}"
            )

        return self.remote_brain

    def respond(self, message: str) -> tuple[str, RouteDecision]:
        decision = self.route(message)

        if decision.tier is BrainTier.FAST_LOCAL:
            return self._generate(
                self.fast_brain,
                message,
                max_tokens=int(os.getenv("JARVIS_FAST_MAX_TOKENS", "512")),
            ), decision

        build_brain = self.require_build_brain()

        return self._generate(
            build_brain,
            message,
            max_tokens=int(os.getenv("JARVIS_BUILD_MAX_TOKENS", "1600")),
        ), decision

    @staticmethod
    def _generate(brain: BrainProvider, prompt: str, *, max_tokens: int) -> str:
        if hasattr(brain, "generate"):
            return str(
                brain.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    top_p=0.9,
                )
            )

        return str(brain.think(prompt, max_tokens=max_tokens))

    def status(self) -> dict[str, Any]:
        config = getattr(self.fast_brain, "config", None)

        return {
            "fast_tier": BrainTier.FAST_LOCAL.value,
            "fast_provider": getattr(
                self.fast_brain, "provider_name", type(self.fast_brain).__name__
            ),
            "fast_model": getattr(self.fast_brain, "model_name", None),
            "fast_base_url": getattr(config, "base_url", None),
            "build_tier": BrainTier.BUILD_REMOTE.value,
            "build_remote_enabled": self.remote_enabled,
            "build_remote_loaded": self.remote_brain is not None,
        }
