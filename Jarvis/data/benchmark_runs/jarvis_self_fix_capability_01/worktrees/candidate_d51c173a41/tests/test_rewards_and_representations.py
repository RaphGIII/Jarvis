import torch

from learning.objectives.losses import info_nce_loss
from learning.representations.encoder import ObservationAutoencoder, ObservationEncoder
from learning.representations.latent_state import BeliefState, LatentState
from learning.rewards.intrinsic import curiosity_reward, learning_progress_reward, novelty_reward
from learning.rewards.reward_model import MultiObjectiveRewardModel, RewardSignal, RewardWeights


def test_reward_composition_keeps_components_separate():
    signal = RewardSignal(
        task_success=1.0,
        user_feedback=0.5,
        correctness=0.25,
        error_penalty=0.1,
        risk_penalty=0.2,
    )
    weights = RewardWeights(task=2.0, user=1.0, accuracy=1.0, error=1.0, risk=1.0)
    model = MultiObjectiveRewardModel(weights)

    assert model.score(signal) == signal.total(weights)
    assert signal.components()["task_success"] == 1.0


def test_intrinsic_reward_signals():
    latent = torch.tensor([1.0, 0.0])
    known = torch.tensor([[0.0, 0.0], [2.0, 0.0]])

    assert novelty_reward(latent, known) == 1.0
    assert curiosity_reward(torch.tensor([1.0]), torch.tensor([3.0])) == 4.0
    assert learning_progress_reward(0.8, 0.3) == 0.5
    assert learning_progress_reward(0.3, 0.8) == 0.0


def test_latent_state_shapes_and_belief_update():
    encoder = ObservationEncoder(input_dim=3, latent_dim=8, hidden_dim=12)
    latent_vector = encoder(torch.tensor([1.0, 2.0, 3.0]))
    assert latent_vector.shape == (1, 8)

    latent = LatentState(latent_vector.squeeze(0), uncertainty=0.4)
    belief = BeliefState(latent=latent)
    updated = belief.update(latent, ["episode-1"])
    assert updated.history_length == 2
    assert updated.memory_refs == ["episode-1"]
    assert updated.latent.dim == 8


def test_autoencoder_and_contrastive_loss_are_trainable_tensors():
    autoencoder = ObservationAutoencoder(input_dim=4, latent_dim=2, hidden_dim=8)
    x = torch.randn(5, 4)
    loss = autoencoder.reconstruction_loss(x)
    assert loss.dim() == 0
    loss.backward()

    anchors = torch.eye(3)
    positives = torch.eye(3)
    contrastive_loss = info_nce_loss(anchors, positives)
    assert float(contrastive_loss.detach()) >= 0.0
