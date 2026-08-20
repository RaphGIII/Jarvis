import sys
from pathlib import Path

import torch

from environments.coding.actions import ActionCandidate, ActionType
from environments.coding.environment import CodingEnvironment
from environments.coding.sandbox_backend import DisabledSandboxBackend, DockerSandboxBackend, LocalTestSandboxBackend
from learning.policy.action_value import ActionValueConfig, ActionValueNetwork, soft_update
from learning.representations.action_encoding import SemanticActionEncoder
from learning.representations.semantic import DeterministicTextEncoder, ProjectionEncoder
from runtime.action_generator import HeuristicCodingActionGenerator, QwenActionGenerator
from runtime.checkpoints import RuntimeCheckpointManager
from runtime.jarvis_runtime import JarvisRuntime, JarvisRuntimeConfig
from runtime.runtime_state import RuntimeMode
from runtime.tensorboard import TensorBoardLogger
from training.coding_curriculum import CodingTaskFactory, DatasetSplit


class MockBrain:
    def __init__(self):
        self.calls = 0

    def think(self, prompt, max_tokens=700):
        self.calls += 1
        return '[{"action_type":"READ_FILE","arguments":{"path":"calculator.py"},"reasoning_summary":"inspect","confidence":0.83}]'


def test_production_action_generator_wiring_uses_qwen_mock(tmp_path):
    brain = MockBrain()
    runtime = JarvisRuntime(
        brain=brain,
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10),
        data_dir=tmp_path / "runtime",
        sandbox_backend=DisabledSandboxBackend(),
        semantic_text_encoder=DeterministicTextEncoder(embedding_dim=16),
    )
    assert isinstance(runtime.action_generator, QwenActionGenerator)
    task = CodingTaskFactory(tmp_path / "tasks").make_hidden_addition_task("qwen_wire")
    observation = runtime.start_task(task, RuntimeMode.EVAL)
    candidates = runtime.action_generator.generate(task.description, observation)
    assert candidates[0].action_type == ActionType.READ_FILE
    assert brain.calls == 1
    runtime.action_generator.generate(task.description, observation)
    assert brain.calls == 1


def test_heuristic_generator_contains_no_task_specific_cheat_solution(tmp_path):
    factory = CodingTaskFactory(tmp_path / "tasks")
    task = factory.make_hidden_addition_task("no_cheat")
    env = CodingEnvironment(task, backend=LocalTestSandboxBackend())
    env.step(ActionCandidate(ActionType.READ_FILE, {"path": "calculator.py"}))
    observation = env.observe()
    candidates = HeuristicCodingActionGenerator().generate("repair addition", observation)
    assert all(candidate.action_type != ActionType.PATCH_FILE for candidate in candidates)


def test_semantic_observation_projection_gets_gradients():
    encoder = DeterministicTextEncoder(embedding_dim=16)
    projection = ProjectionEncoder(semantic_dim=16, numeric_dim=4, latent_dim=8, hidden_dim=12)
    before = [parameter.detach().clone() for parameter in projection.parameters()]
    semantic = encoder.encode("fix calculator return statement")
    numeric = torch.tensor([1.0, 0.0, 0.5, 0.2])
    output = projection(semantic, numeric)
    loss = output.pow(2).mean()
    loss.backward()
    optimizer = torch.optim.Adam(projection.parameters(), lr=0.01)
    optimizer.step()
    assert any(not torch.allclose(old, new) for old, new in zip(before, projection.parameters()))


def test_semantic_action_encoding_differentiates_concrete_patches():
    text_encoder = DeterministicTextEncoder(embedding_dim=16)
    action_encoder = SemanticActionEncoder(text_encoder, action_embedding_dim=8, hidden_dim=12)
    patch_small = ActionCandidate(ActionType.PATCH_FILE, {"path": "a.py", "old": "return a - b", "new": "return a + b"})
    patch_large = ActionCandidate(ActionType.PATCH_FILE, {"path": "a.py", "old": "return a - b", "new": ""})
    raw_small = action_encoder.raw_features(patch_small)
    raw_large = action_encoder.raw_features(patch_large)
    assert not torch.allclose(raw_small, raw_large)
    before = [parameter.detach().clone() for parameter in action_encoder.parameters()]
    loss = action_encoder.forward_from_raw(torch.stack([raw_small, raw_large])).pow(2).mean()
    loss.backward()
    torch.optim.Adam(action_encoder.parameters(), lr=0.01).step()
    assert any(not torch.allclose(old, new) for old, new in zip(before, action_encoder.parameters()))


def test_action_value_network_and_target_soft_update():
    online = ActionValueNetwork(ActionValueConfig(state_dim=4, action_dim=3, hidden_dim=8))
    target = ActionValueNetwork(ActionValueConfig(state_dim=4, action_dim=3, hidden_dim=8))
    for parameter in online.parameters():
        parameter.data.add_(1.0)
    before = [parameter.detach().clone() for parameter in target.parameters()]
    soft_update(target, online, tau=0.5)
    assert any(not torch.allclose(old, new) for old, new in zip(before, target.parameters()))
    q = online(torch.randn(2, 4), torch.randn(2, 3))
    assert q.shape == (2,)


class ScriptedGenerator:
    def generate(self, goal, observation):
        if not observation.test_state.get("ran", False):
            return [ActionCandidate(ActionType.RUN_TESTS)]
        if "calculator.py" not in observation.relevant_file_excerpts:
            return [ActionCandidate(ActionType.READ_FILE, {"path": "calculator.py"})]
        if "return a - b" in observation.relevant_file_excerpts.get("calculator.py", ""):
            return [ActionCandidate(ActionType.PATCH_FILE, {"path": "calculator.py", "old": "return a - b", "new": "return a + b"})]
        return [ActionCandidate(ActionType.RUN_TESTS)]


def test_replay_warm_start_and_persistent_runtime_across_restart(tmp_path):
    factory = CodingTaskFactory(tmp_path / "tasks")
    runtime_a = JarvisRuntime(
        action_generator=ScriptedGenerator(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=50, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.TRAIN,
        sandbox_backend=LocalTestSandboxBackend(),
    )
    runtime_a.scheduler.config.value_policy_batch_size = 1
    runtime_a.scheduler.config.value_policy_train_every_n_steps = 1
    runtime_a.run_episode(factory.make_hidden_addition_task("persist_a"), RuntimeMode.TRAIN)
    runtime_a.save_checkpoints({"success_rate": 1.0, "mean_reward": 1.0, "regression_rate": 0.0})
    policy_steps = runtime_a.policy.training_step
    replay_count = len(runtime_a.replay_buffer)

    runtime_b = JarvisRuntime(
        action_generator=ScriptedGenerator(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=50, load_latest_checkpoints=True),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.TRAIN,
        sandbox_backend=LocalTestSandboxBackend(),
    )
    assert len(runtime_b.replay_buffer) >= replay_count
    assert runtime_b.policy.training_step >= policy_steps
    runtime_b.scheduler.config.value_policy_batch_size = 1
    runtime_b.scheduler.config.value_policy_train_every_n_steps = 1
    runtime_b.run_episode(factory.make_hidden_addition_task("persist_b"), RuntimeMode.TRAIN)
    assert runtime_b.policy.training_step > policy_steps


def test_eval_holdout_does_not_enter_replay(tmp_path):
    factory = CodingTaskFactory(tmp_path / "tasks")
    runtime = JarvisRuntime(
        action_generator=ScriptedGenerator(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=20),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.EVAL,
        sandbox_backend=LocalTestSandboxBackend(),
    )
    before_replay = len(runtime.replay_buffer)
    before_store = runtime.experience_store.count()
    runtime.run_episode(factory.make_hidden_addition_task("holdout", split=DatasetSplit.HOLDOUT), RuntimeMode.EVAL)
    assert len(runtime.replay_buffer) == before_replay
    assert runtime.experience_store.count() == before_store


def test_hidden_tests_inaccessible_and_test_files_protected(tmp_path):
    task = CodingTaskFactory(tmp_path / "tasks").make_hidden_addition_task("hidden")
    env = CodingEnvironment(task, backend=LocalTestSandboxBackend())
    assert all("hidden" not in path for path in env.observe().workspace_tree)
    protected = env.step(ActionCandidate(ActionType.PATCH_FILE, {"path": "test_public.py", "old": "2", "new": "999"}))
    assert not protected.action_result.ok
    assert "Protected" in protected.action_result.message


def test_reward_hacking_public_only_change_fails_hidden_verifier(tmp_path):
    task = CodingTaskFactory(tmp_path / "tasks").make_hidden_addition_task("reward_hack")
    env = CodingEnvironment(task, backend=LocalTestSandboxBackend())
    env.step(ActionCandidate(ActionType.WRITE_FILE, {"path": "calculator.py", "content": "def add(a, b):\n    return 5\n"}))
    result = env.step(ActionCandidate(ActionType.RUN_TESTS))
    assert not result.success
    assert result.observation.test_state["hidden_failed"] == 1


def test_sandbox_default_disables_execution_and_policy_declares_no_network(tmp_path):
    task = CodingTaskFactory(tmp_path / "tasks").make_hidden_addition_task("sandbox_disabled")
    env = CodingEnvironment(task)
    result = env.step(ActionCandidate(ActionType.RUN_TESTS))
    assert not result.action_result.ok
    assert result.action_result.return_code == 126
    assert DockerSandboxBackend().policy.network_disabled


def test_candidate_scoring_contains_q_component(tmp_path):
    runtime = JarvisRuntime(
        action_generator=ScriptedGenerator(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=20),
        data_dir=tmp_path / "runtime",
        sandbox_backend=LocalTestSandboxBackend(),
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_hidden_addition_task("score")
    observation = runtime.start_task(task, RuntimeMode.EVAL)
    latent = runtime._encode_features(runtime._observation_features(observation))
    scored = runtime._score_candidates(latent, [ActionCandidate(ActionType.RUN_TESTS)], observation)
    assert "q_value" in scored[0].to_dict()


def test_checkpoint_best_latest_logic(tmp_path):
    manager = RuntimeCheckpointManager(tmp_path / "checkpoints")
    assert manager.should_promote({"success_rate": 0.5, "mean_reward": 1.0, "regression_rate": 0.0}, None)
    assert not manager.should_promote(
        {"success_rate": 0.4, "mean_reward": 2.0, "regression_rate": 0.0},
        {"success_rate": 0.5, "mean_reward": 1.0, "regression_rate": 0.0},
    )
    manager.save_category_metadata("best", {"success_rate": 0.5, "mean_reward": 1.0, "regression_rate": 0.0})
    assert manager.best_metrics()["success_rate"] == 0.5


def test_tensorboard_metrics_smoke(tmp_path):
    logger = TensorBoardLogger(tmp_path / "tb")
    logger.log_scalar("train/reward", 1.0, 1)
    logger.close()
    assert (tmp_path / "tb" / "metrics.jsonl").exists()
