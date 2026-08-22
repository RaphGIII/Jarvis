import torch

from learning.policy.exploration import (
    CandidateAction,
    CandidateActionScorer,
    EpsilonGreedyExploration,
    GreedyExploration,
    SafeExplorationGate,
    SafeExplorationLevel,
    SoftmaxExploration,
    UCBExploration,
)
from learning.policy.policy import NeuralPolicy, PolicyConfig, policy_gradient_loss
from learning.policy.value import NeuralValueFunction, ValueConfig, bellman_target, td_error
from training.learning_demo import PolicyLearningDemoConfig, run_policy_learning_demo


def test_policy_and_value_forward_passes():
    policy = NeuralPolicy(PolicyConfig(state_dim=4, num_actions=3, hidden_dim=8))
    value = NeuralValueFunction(ValueConfig(state_dim=4, hidden_dim=8))
    states = torch.randn(5, 4)

    assert policy(states).shape == (5, 3)
    assert policy.action_distribution(states).shape == (5, 3)
    assert value(states).shape == (5,)


def test_policy_learning_demo_improves_reward():
    metrics = run_policy_learning_demo(
        PolicyLearningDemoConfig(samples=96, train_steps=80, learning_rate=0.03, seed=8)
    )
    assert metrics["policy_reward_after"] > metrics["policy_reward_before"]
    assert metrics["value_loss_after"] < metrics["value_loss_before"]
    assert metrics["policy_parameter_l1_delta"] > 0.0


def test_policy_gradient_loss_and_td_error():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    loss = policy_gradient_loss(logits, torch.tensor([0, 1]), torch.tensor([1.0, 0.5]))
    loss.backward()
    assert logits.grad is not None

    target = bellman_target(torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([0.0]), discount=0.5)
    error = td_error(torch.tensor([1.0]), torch.tensor([1.5]), torch.tensor([2.0]), torch.tensor([0.0]), discount=0.5)
    assert torch.allclose(target, torch.tensor([2.0]))
    assert torch.allclose(error, torch.tensor([0.5]))


def test_exploration_strategies_and_candidate_scoring():
    q_values = torch.tensor([0.1, 0.5, 0.2])
    assert GreedyExploration().select(q_values) == 1
    assert EpsilonGreedyExploration(epsilon=0.0, seed=1).select(q_values) == 1
    assert SoftmaxExploration(temperature=1.0).probabilities(q_values).shape == (3,)
    assert UCBExploration(c=1.0).select(q_values, torch.tensor([10, 1, 10]), total_count=21) in {1, 2}

    scorer = CandidateActionScorer(information_gain_weight=0.5, risk_weight=1.0)
    ranked = scorer.rank(
        [
            CandidateAction("safe", q_value=0.4, expected_information_gain=0.1, risk=0.0),
            CandidateAction("risky", q_value=0.8, expected_information_gain=0.1, risk=1.0),
        ]
    )
    assert ranked[0].name == "safe"

    gate = SafeExplorationGate(SafeExplorationLevel.SANDBOX)
    assert gate.allows(SafeExplorationLevel.SIMULATION)
    assert not gate.allows(SafeExplorationLevel.REAL_WORLD)
