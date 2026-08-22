from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from learning.representations.latent_state import BeliefState, LatentState


@dataclass(frozen=True)
class POMDPDefinition:
    """Formal shell M=(S,A,O,T,R,gamma) without assuming full observability."""

    state_space: str
    action_space: str
    observation_space: str
    transition_model: str
    reward_model: str
    discount_factor: float


class BeliefStateEstimator:
    """Adapter boundary for future trainable f_theta belief models."""

    def __init__(self, encode_observation: Callable[[Any], LatentState]) -> None:
        self.encode_observation = encode_observation

    def initial(self, observation: Any, memory_refs: Iterable[str] | None = None) -> BeliefState:
        return BeliefState(
            latent=self.encode_observation(observation),
            history_length=1,
            memory_refs=list(memory_refs or []),
        )

    def update(self, belief: BeliefState, observation: Any, memory_refs: Iterable[str] | None = None) -> BeliefState:
        return belief.update(self.encode_observation(observation), list(memory_refs or []))
