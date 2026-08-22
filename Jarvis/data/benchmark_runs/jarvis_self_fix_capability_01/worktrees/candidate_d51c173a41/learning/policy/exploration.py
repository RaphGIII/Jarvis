from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


class SafeExplorationLevel(str, Enum):
    SIMULATION = "simulation"
    SANDBOX = "sandbox"
    REAL_WORLD = "real_world"


@dataclass(frozen=True)
class CandidateAction:
    name: str
    payload: Any = None
    q_value: float = 0.0
    expected_information_gain: float = 0.0
    risk: float = 0.0


class CandidateActionScorer:
    """Scores LLM-proposed candidates without making the LLM train online."""

    def __init__(self, information_gain_weight: float = 0.2, risk_weight: float = 1.0) -> None:
        self.information_gain_weight = information_gain_weight
        self.risk_weight = risk_weight

    def score(self, action: CandidateAction) -> float:
        return float(
            action.q_value
            + self.information_gain_weight * action.expected_information_gain
            - self.risk_weight * action.risk
        )

    def rank(self, actions: Sequence[CandidateAction]) -> list[CandidateAction]:
        return sorted(actions, key=self.score, reverse=True)


class GreedyExploration:
    def select(self, q_values: Tensor) -> int:
        return int(torch.argmax(q_values).item())


class EpsilonGreedyExploration:
    def __init__(self, epsilon: float = 0.1, seed: int | None = None) -> None:
        self.epsilon = epsilon
        self._rng = random.Random(seed)

    def select(self, q_values: Tensor) -> int:
        values = q_values.reshape(-1)
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(values.numel())
        return int(torch.argmax(values).item())


class SoftmaxExploration:
    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def probabilities(self, q_values: Tensor) -> Tensor:
        return F.softmax(q_values.reshape(-1).float() / self.temperature, dim=-1)

    def select(self, q_values: Tensor) -> int:
        probabilities = self.probabilities(q_values)
        return int(torch.distributions.Categorical(probabilities).sample().item())


class UCBExploration:
    def __init__(self, c: float = 1.0) -> None:
        self.c = c

    def scores(self, q_values: Tensor, action_counts: Tensor, total_count: int) -> Tensor:
        counts = action_counts.float().clamp_min(1.0)
        bonus = self.c * torch.sqrt(torch.tensor(math.log(max(total_count, 2))) / counts)
        return q_values.reshape(-1).float() + bonus

    def select(self, q_values: Tensor, action_counts: Tensor, total_count: int) -> int:
        return int(torch.argmax(self.scores(q_values, action_counts, total_count)).item())


class SafeExplorationGate:
    """Prevents risky exploration from silently jumping into real-world actions."""

    def __init__(self, allowed_level: SafeExplorationLevel = SafeExplorationLevel.SIMULATION) -> None:
        self.allowed_level = allowed_level

    def allows(self, requested_level: SafeExplorationLevel) -> bool:
        order = [
            SafeExplorationLevel.SIMULATION,
            SafeExplorationLevel.SANDBOX,
            SafeExplorationLevel.REAL_WORLD,
        ]
        return order.index(requested_level) <= order.index(self.allowed_level)
