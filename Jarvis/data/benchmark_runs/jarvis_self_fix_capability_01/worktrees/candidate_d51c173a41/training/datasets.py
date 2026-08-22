from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from learning.experience.trajectory import Trajectory
from learning.experience.transition import Transition
from learning.rewards.preferences import PreferenceSample


class DatasetBuilder(Protocol):
    def build(self, data: Any) -> list[dict[str, Any]]:
        ...


@dataclass
class SFTDatasetBuilder:
    def build(self, samples: list[tuple[Any, Any]]) -> list[dict[str, Any]]:
        return [{"input": input_value, "ideal_output": output_value} for input_value, output_value in samples]


@dataclass
class PreferenceDatasetBuilder:
    def build(self, samples: list[PreferenceSample]) -> list[dict[str, Any]]:
        return [
            {"prompt": sample.prompt, "chosen": sample.chosen, "rejected": sample.rejected, "metadata": sample.metadata or {}}
            for sample in samples
        ]


@dataclass
class RLDatasetBuilder:
    def build(self, transitions: list[Transition]) -> list[dict[str, Any]]:
        return [
            {
                "state": transition.latent_state,
                "action": transition.action,
                "reward": transition.reward,
                "next_state": transition.next_latent_state,
                "done": transition.done,
            }
            for transition in transitions
        ]


@dataclass
class WorldModelDatasetBuilder:
    def build(self, transitions: list[Transition]) -> list[dict[str, Any]]:
        return [
            {
                "state": transition.latent_state,
                "action": transition.action,
                "next_state": transition.next_latent_state,
                "reward": transition.reward,
            }
            for transition in transitions
        ]


@dataclass
class SkillDatasetBuilder:
    def build(self, trajectories: list[Trajectory]) -> list[dict[str, Any]]:
        return [
            {
                "goal": trajectory.metadata.get("goal"),
                "trajectory": trajectory.actions,
                "outcome": trajectory.final_success,
                "total_reward": trajectory.total_reward,
            }
            for trajectory in trajectories
        ]
