from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from learning.experience.prioritization import (
    PriorityConfig,
    compute_priority,
    importance_sampling_weights,
    sampling_probabilities,
)
from learning.experience.transition import Transition


@dataclass
class PrioritizedBatch:
    transitions: list[Transition]
    indices: list[int]
    weights: Tensor


class ReplayBuffer:
    """Cyclic replay buffer with uniform and prioritized sampling."""

    def __init__(
        self,
        capacity: int,
        priority_config: PriorityConfig | None = None,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.priority_config = priority_config or PriorityConfig()
        self._storage: list[Transition] = []
        self._priorities: list[float] = []
        self._position = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def priorities(self) -> list[float]:
        return list(self._priorities)

    def add(
        self,
        transition: Transition,
        priority: float | None = None,
        error: float | None = None,
    ) -> None:
        if priority is None:
            priority = compute_priority(
                transition.prediction_error if error is None else error,
                self.priority_config,
            )
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
            self._priorities.append(float(priority))
            return

        self._storage[self._position] = transition
        self._priorities[self._position] = float(priority)
        self._position = (self._position + 1) % self.capacity

    def add_many(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        self._validate_can_sample(batch_size)
        return self._rng.sample(self._storage, batch_size)

    def sample_recent(self, batch_size: int) -> list[Transition]:
        self._validate_can_sample(batch_size)
        return self._ordered_storage()[-batch_size:]

    def sample_failures(self, batch_size: int) -> list[Transition]:
        return self._sample_filtered(lambda transition: not transition.success, batch_size)

    def sample_successes(self, batch_size: int) -> list[Transition]:
        return self._sample_filtered(lambda transition: transition.success, batch_size)

    def sample_priority(self, batch_size: int) -> PrioritizedBatch:
        self._validate_can_sample(batch_size)
        priority_tensor = torch.tensor(self._priorities, dtype=torch.float32)
        probabilities = sampling_probabilities(priority_tensor)
        indices_tensor = torch.multinomial(probabilities, batch_size, replacement=False)
        weights = importance_sampling_weights(
            probabilities,
            indices_tensor,
            len(self._storage),
            beta=self.priority_config.beta,
        )
        indices = [int(index) for index in indices_tensor.tolist()]
        return PrioritizedBatch(
            transitions=[self._storage[index] for index in indices],
            indices=indices,
            weights=weights,
        )

    def update_priorities(self, indices: Iterable[int], errors: Iterable[float]) -> None:
        for index, error in zip(indices, errors):
            self._priorities[index] = compute_priority(error, self.priority_config)

    def _sample_filtered(self, predicate, batch_size: int) -> list[Transition]:
        candidates = [transition for transition in self._ordered_storage() if predicate(transition)]
        if not candidates:
            return []
        return self._rng.sample(candidates, min(batch_size, len(candidates)))

    def _validate_can_sample(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._storage):
            raise ValueError("batch_size cannot exceed replay size")

    def _ordered_storage(self) -> list[Transition]:
        if len(self._storage) < self.capacity or self._position == 0:
            return list(self._storage)
        return self._storage[self._position :] + self._storage[: self._position]
