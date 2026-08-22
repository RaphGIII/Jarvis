from __future__ import annotations

from dataclasses import dataclass, field

from learning.meta.metrics import ExponentialMovingAverage


@dataclass
class CapabilityEstimate:
    capability: str
    attempts: int = 0
    successes: int = 0
    mean_reward: float = 0.0
    uncertainty: float = 1.0
    ema_recent: ExponentialMovingAverage = field(default_factory=lambda: ExponentialMovingAverage(alpha=0.3))
    ema_previous_value: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    @property
    def trend(self) -> float:
        return float((self.ema_recent.value or 0.0) - self.ema_previous_value)

    def update(self, success: bool, reward: float, uncertainty: float) -> None:
        self.attempts += 1
        self.successes += int(success)
        self.mean_reward += (reward - self.mean_reward) / self.attempts
        self.uncertainty = uncertainty
        self.ema_previous_value = self.ema_recent.value or 0.0
        self.ema_recent.update(1.0 if success else 0.0)


@dataclass
class SelfModel:
    capabilities: dict[str, CapabilityEstimate] = field(default_factory=dict)

    def update_capability(self, capability: str, success: bool, reward: float, uncertainty: float) -> CapabilityEstimate:
        estimate = self.capabilities.setdefault(capability, CapabilityEstimate(capability=capability))
        estimate.update(success, reward, uncertainty)
        return estimate

    def weakest_capabilities(self, limit: int = 5) -> list[CapabilityEstimate]:
        return sorted(self.capabilities.values(), key=lambda item: (item.success_rate, -item.uncertainty))[:limit]
