from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearningConfig:
    """Central defaults for the trainable learning substrate."""

    latent_dim: int = 256
    action_dim: int = 9
    discount_factor: float = 0.99

    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_epsilon: float = 1e-5

    info_nce_temperature: float = 0.2

    reward_task_weight: float = 1.0
    reward_user_weight: float = 1.0
    reward_accuracy_weight: float = 1.0
    reward_efficiency_weight: float = 0.3
    reward_novelty_weight: float = 0.2
    reward_learning_weight: float = 0.5
    reward_error_weight: float = 1.0
    reward_risk_weight: float = 1.0

    world_transition_weight: float = 1.0
    world_reward_weight: float = 0.2

    checkpoint_dir: str = "data/checkpoints"
    experiment_dir: str = "data/experiments"

    num_action_candidates: int = 4
    world_model_train_every_n_steps: int = 4
    world_model_batch_size: int = 8
    world_model_lr: float = 1e-3
    value_policy_train_every_n_steps: int = 4
    value_policy_batch_size: int = 8
    value_lr: float = 1e-3
    policy_lr: float = 1e-3
    priority_prediction_error_weight: float = 0.5
    replay_capacity: int = 500


DEFAULT_CONFIG = LearningConfig()

# Requested explicit names for configuration hooks.
LATENT_DIM = DEFAULT_CONFIG.latent_dim
PER_ALPHA = DEFAULT_CONFIG.per_alpha
PER_BETA = DEFAULT_CONFIG.per_beta
PER_EPSILON = DEFAULT_CONFIG.per_epsilon
