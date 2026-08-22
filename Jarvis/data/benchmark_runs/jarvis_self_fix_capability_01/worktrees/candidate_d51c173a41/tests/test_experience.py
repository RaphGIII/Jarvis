import torch

from learning.experience.prioritization import importance_sampling_weights, sampling_probabilities
from learning.experience.replay_buffer import ReplayBuffer
from learning.experience.transition import Transition
from learning.experience.trajectory import Trajectory


def make_transition(index: int, success: bool = False) -> Transition:
    z = torch.tensor([float(index), 0.0])
    return Transition(
        observation=[index],
        latent_state=z,
        action=index % 2,
        reward=float(index),
        next_observation=[index + 1],
        next_latent_state=z + 1.0,
        done=False,
        uncertainty=0.1,
        novelty=0.2,
        success=success,
        metadata={"prediction_error": float(index)},
    )


def test_replay_buffer_sampling_modes():
    buffer = ReplayBuffer(capacity=5, seed=1)
    for index in range(6):
        buffer.add(make_transition(index, success=index % 2 == 0))

    assert len(buffer) == 5
    assert len(buffer.sample(2)) == 2
    assert [item.observation[0] for item in buffer.sample_recent(2)] == [4, 5]
    assert all(item.success for item in buffer.sample_successes(10))
    assert all(not item.success for item in buffer.sample_failures(10))


def test_prioritized_sampling_and_weights():
    buffer = ReplayBuffer(capacity=4, seed=2)
    for index in range(4):
        buffer.add(make_transition(index), error=float(index + 1))

    batch = buffer.sample_priority(2)
    assert len(batch.transitions) == 2
    assert batch.weights.shape == (2,)
    assert float(batch.weights.max()) <= 1.0

    priorities = torch.tensor(buffer.priorities)
    probabilities = sampling_probabilities(priorities)
    weights = importance_sampling_weights(probabilities, torch.tensor(batch.indices), len(buffer))
    assert torch.all(weights > 0)


def test_trajectory_records_success_and_reward():
    trajectory = Trajectory(task_id="task")
    trajectory.add(make_transition(1, success=False))
    trajectory.add(make_transition(2, success=True))

    assert len(trajectory) == 2
    assert trajectory.total_reward == 3.0
    assert trajectory.success_rate == 0.5
    assert trajectory.final_success is True
    assert trajectory.to_rl_records()[0]["action"] == 1
