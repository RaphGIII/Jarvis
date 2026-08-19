from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from learning.experience.trajectory import Trajectory


class RuntimeMode(str, Enum):
    TRAIN = "train"
    EVAL = "eval"
    CHAT = "chat"


@dataclass
class RuntimeState:
    mode: RuntimeMode = RuntimeMode.TRAIN
    task_id: str | None = None
    episode_id: str | None = None
    step_count: int = 0
    total_reward: float = 0.0
    trajectory: Trajectory = field(default_factory=Trajectory)
    latest_metrics: dict[str, float | int | str | bool] = field(default_factory=dict)

    def reset_episode(self, task_id: str, episode_id: str) -> None:
        self.task_id = task_id
        self.episode_id = episode_id
        self.step_count = 0
        self.total_reward = 0.0
        self.trajectory = Trajectory(task_id=task_id, metadata={"episode_id": episode_id})
        self.latest_metrics = {}
