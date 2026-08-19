import torch

from learning.objectives.optimizer import make_optimizer, set_global_seeds
from learning.world_model.model import WorldModel, WorldModelConfig


def synthetic_batch(samples=64, latent_dim=3, action_dim=2):
    set_global_seeds(3)
    states = torch.randn(samples, latent_dim)
    actions_index = torch.randint(0, action_dim, (samples,))
    actions = torch.nn.functional.one_hot(actions_index, action_dim).float()
    effects = torch.tensor([[0.5, -0.1, 0.0], [-0.2, 0.3, 0.1]], dtype=torch.float32)
    next_states = states + effects[actions_index]
    rewards = next_states.sum(dim=1)
    return states, actions, next_states, rewards


def test_world_model_forward_pass():
    model = WorldModel(WorldModelConfig(latent_dim=3, action_dim=2, hidden_dim=8))
    states, actions, _, _ = synthetic_batch(samples=4)
    output = model(states, actions)
    assert output.next_latent_pred.shape == (4, 3)
    assert output.reward_pred.shape == (4,)


def test_world_model_training_reduces_loss():
    model = WorldModel(WorldModelConfig(latent_dim=3, action_dim=2, hidden_dim=16))
    optimizer = make_optimizer(model, learning_rate=0.03)
    states, actions, next_states, rewards = synthetic_batch()

    before, _ = model.loss(states, actions, next_states, rewards)
    for _ in range(80):
        model.train_step(optimizer, states, actions, next_states, rewards)
    after, _ = model.loss(states, actions, next_states, rewards)

    assert float(after.detach()) < float(before.detach())


def test_checkpoint_save_and_load(tmp_path):
    model = WorldModel(WorldModelConfig(latent_dim=3, action_dim=2, hidden_dim=8))
    states, actions, next_states, rewards = synthetic_batch(samples=8)
    optimizer = make_optimizer(model, learning_rate=0.01)
    model.train_step(optimizer, states, actions, next_states, rewards)
    checkpoint_path = model.save_checkpoint(tmp_path / "world.pt", optimizer=optimizer, metrics={"loss": 1.0})

    loaded = WorldModel(WorldModelConfig(latent_dim=3, action_dim=2, hidden_dim=8))
    payload = loaded.load_checkpoint(checkpoint_path)

    assert payload["version"] == "world-model-0.1"
    for original, restored in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(original, restored)
