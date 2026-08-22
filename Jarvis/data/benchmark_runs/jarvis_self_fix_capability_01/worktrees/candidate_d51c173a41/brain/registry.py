from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from config import MODEL_ID


@dataclass(frozen=True)
class BrainConfig:
    profile: str
    provider: str
    model: str
    max_context_tokens: int = 8192
    generation: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self, profiles: dict[str, BrainConfig] | None = None) -> None:
        self._profiles = profiles or default_profiles()

    def get(self, name: str | None = None) -> BrainConfig:
        profile = name or os.getenv("JARVIS_BRAIN_PROFILE", "local_qwen4b")
        if profile not in self._profiles:
            raise KeyError(f"Unknown brain profile: {profile}")
        return self._profiles[profile]

    def profiles(self) -> dict[str, BrainConfig]:
        return dict(self._profiles)


def default_profiles() -> dict[str, BrainConfig]:
    return {
        "local_qwen4b": BrainConfig(
            profile="local_qwen4b",
            provider="local_transformers",
            model=MODEL_ID,
            generation={"temperature": 0.6, "top_p": 0.9, "coding_max_tokens": 450},
            capabilities={"chat": True, "coding": True, "local": True},
        ),
        "remote_qwen4b": BrainConfig(
            profile="remote_qwen4b",
            provider="openai_compatible",
            model="Qwen/Qwen3-4B-Instruct-2507",
            generation={"temperature": 0.6, "top_p": 0.9, "coding_max_tokens": 450},
            capabilities={"chat": True, "coding": True, "local": False},
        ),
        "remote_large_coder": BrainConfig(
            profile="remote_large_coder",
            provider="openai_compatible",
            model="configured-remote-coder",
            generation={"temperature": 0.4, "top_p": 0.9, "coding_max_tokens": 600},
            capabilities={"chat": True, "coding": True, "local": False},
        ),
        "future_jarvis": BrainConfig(
            profile="future_jarvis",
            provider="openai_compatible",
            model="future-jarvis",
            generation={"temperature": 0.4, "top_p": 0.9, "coding_max_tokens": 600},
            capabilities={"chat": True, "coding": True, "self_trained": True},
        ),
    }
