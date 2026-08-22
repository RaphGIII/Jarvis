from __future__ import annotations

from learning.rewards.reward_model import RewardSignal


def task_outcome_reward(success: bool, correctness: float = 0.0, user_feedback: float = 0.0) -> RewardSignal:
    return RewardSignal(
        task_success=1.0 if success else 0.0,
        correctness=float(correctness),
        user_feedback=float(user_feedback),
        error_penalty=0.0 if success else 1.0,
    )
