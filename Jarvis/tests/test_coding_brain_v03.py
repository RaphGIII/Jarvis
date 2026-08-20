import json
import sys

import torch

from environments.coding.actions import ActionCandidate, ActionType
from environments.coding.environment import CodingEnvironment
from environments.coding.reward import CodingRewardEngine
from environments.coding.sandbox_backend import LocalTestSandboxBackend
from learning.representations.action_encoding import SemanticActionEncoder
from learning.representations.semantic import DeterministicTextEncoder
from runtime.action_generator import QwenActionGenerator
from runtime.checkpoints import RuntimeCheckpointManager
from runtime.jarvis_runtime import JarvisRuntime, JarvisRuntimeConfig
from runtime.runtime_state import RuntimeMode
from training.coding_benchmark import CodingBenchmark
from training.coding_brain_v03_demo import CodingBrainV03Config, run_coding_brain_v03_demo
from training.coding_curriculum import CodingTaskFactory, DatasetSplit


class StaticBrain:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.model = torch.nn.Linear(1, 1)

    def think(self, prompt, max_tokens=1200):
        self.calls += 1
        return self.response(prompt) if callable(self.response) else self.response


class SequentialPatchBrain:
    def __init__(self):
        self.calls = 0

    def think(self, prompt, max_tokens=1200):
        self.calls += 1
        lowered = prompt.lower()
        if "tests: {'ran': false" in lowered:
            return json.dumps([{"action_type": "RUN_TESTS", "arguments": {}, "reason": "baseline", "confidence": 0.9}])
        if "solution.py:" not in lowered:
            return json.dumps([{"action_type": "READ_FILE", "path": "solution.py", "reason": "inspect", "confidence": 0.9}])
        if "solution.py:\ndef add_numbers(a, b):\n    return a - b" in lowered or "solution.py:\ndef add(a, b):\n    return a - b" in lowered:
            return json.dumps(
                [
                    {
                        "action_type": "PATCH_FILE",
                        "path": "solution.py",
                        "arguments": {"old": "return a - b", "new": "return a + b"},
                        "reason": "concrete arithmetic patch",
                        "confidence": 0.9,
                    }
                ]
            )
        if "hidden_passed': 1" in lowered or "hidden_passed\": 1" in lowered:
            return json.dumps([{"action_type": "FINISH", "arguments": {}, "reason": "verified", "confidence": 0.9}])
        return json.dumps([{"action_type": "RUN_TESTS", "arguments": {}, "reason": "verify", "confidence": 0.9}])


def _runtime(tmp_path, brain, mode=RuntimeMode.TRAIN):
    runtime = JarvisRuntime(
        brain=brain,
        semantic_text_encoder=DeterministicTextEncoder(embedding_dim=32),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=80, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime",
        mode=mode,
        sandbox_backend=LocalTestSandboxBackend(),
    )
    runtime.scheduler.config.value_policy_batch_size = 1
    runtime.scheduler.config.world_model_batch_size = 1
    runtime.scheduler.config.value_policy_train_every_n_steps = 1
    runtime.scheduler.config.world_model_train_every_n_steps = 1
    return runtime


def test_v03_qwen_patch_candidates_have_distinct_semantic_actions():
    encoder = SemanticActionEncoder(DeterministicTextEncoder(embedding_dim=16), action_embedding_dim=8, hidden_dim=8)
    first = ActionCandidate(ActionType.PATCH_FILE, {"path": "solution.py", "old": "return a - b", "new": "return a + b"})
    second = ActionCandidate(ActionType.PATCH_FILE, {"path": "solution.py", "old": "return a - b", "new": "return a * b"})
    assert not torch.allclose(encoder.raw_features(first), encoder.raw_features(second))


def test_v03_qwen_duplicate_patches_are_deduplicated(tmp_path):
    response = json.dumps(
        [
            {"action_type": "PATCH_FILE", "path": "solution.py", "arguments": {"old": "return a - b", "new": "return a + b"}},
            {"action_type": "PATCH_FILE", "path": "solution.py", "arguments": {"old": "return a - b", "new": "return a + b"}},
            {"action_type": "READ_FILE", "path": "solution.py"},
        ]
    )
    generator = QwenActionGenerator(StaticBrain(response), num_candidates=6)
    observation = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    candidates = generator.generate(observation.description, CodingEnvironment(observation, backend=LocalTestSandboxBackend()).observe())
    assert len(candidates) == 2
    assert generator.last_generation_metadata["duplicate_candidates"] == 1


def test_v03_malformed_qwen_output_falls_back_without_crashing(tmp_path):
    generator = QwenActionGenerator(StaticBrain("not json at all"), num_candidates=4)
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    candidates = generator.generate(task.description, CodingEnvironment(task, backend=LocalTestSandboxBackend()).observe())
    assert candidates
    assert generator.last_generation_metadata["parse_error"]


def test_v03_success_reward_dominates_safe_noop(tmp_path):
    task = CodingTaskFactory(tmp_path / "tasks").make_hidden_addition_task("reward_order")
    reward_engine = CodingRewardEngine()
    noop_env = CodingEnvironment(task, backend=LocalTestSandboxBackend())
    before_noop = noop_env.observe()
    noop_step = noop_env.step(ActionCandidate(ActionType.LIST_FILES))
    noop_reward = reward_engine.compute(before_noop, ActionCandidate(ActionType.LIST_FILES), noop_step)

    solved_task = CodingTaskFactory(tmp_path / "tasks2").make_hidden_addition_task("reward_solved")
    solved_env = CodingEnvironment(solved_task, backend=LocalTestSandboxBackend())
    solved_env.step(ActionCandidate(ActionType.PATCH_FILE, {"path": "calculator.py", "old": "return a - b", "new": "return a + b"}))
    before_verify = solved_env.observe()
    solved_step = solved_env.step(ActionCandidate(ActionType.RUN_TESTS))
    solved_reward = reward_engine.compute(before_verify, ActionCandidate(ActionType.RUN_TESTS), solved_step)

    assert solved_step.success
    assert solved_reward.total > noop_reward.total
    assert solved_reward.components["R_completion"] >= 20.0


def test_v03_public_only_reward_hacking_still_fails_hidden(tmp_path):
    task = CodingTaskFactory(tmp_path / "tasks").make_hidden_addition_task("public_only")
    env = CodingEnvironment(task, backend=LocalTestSandboxBackend())
    env.step(ActionCandidate(ActionType.WRITE_FILE, {"path": "calculator.py", "content": "def add(a, b):\n    return 5\n"}))
    result = env.step(ActionCandidate(ActionType.RUN_TESTS))
    assert not result.success
    assert result.observation.test_state["hidden_failed"] == 1


def test_v03_eval_never_updates_replay_or_optimizers(tmp_path):
    runtime = _runtime(tmp_path, SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    benchmark = CodingBenchmark()
    before_steps = {
        "policy": runtime.policy.training_step,
        "value": runtime.value_function.training_step,
        "q": runtime.action_value.training_step,
        "world": runtime.world_model.training_step,
    }
    before_replay = len(runtime.replay_buffer)
    before_store = runtime.experience_store.count()
    tasks = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 2)
    benchmark.evaluate(runtime, tasks)
    assert len(runtime.replay_buffer) == before_replay
    assert runtime.experience_store.count() == before_store
    assert runtime.policy.training_step == before_steps["policy"]
    assert runtime.value_function.training_step == before_steps["value"]
    assert runtime.action_value.training_step == before_steps["q"]
    assert runtime.world_model.training_step == before_steps["world"]


def test_v03_qwen_parameters_remain_frozen_during_training(tmp_path):
    brain = StaticBrain(lambda prompt: json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}]))
    runtime = _runtime(tmp_path, brain)
    before = [parameter.detach().clone() for parameter in brain.model.parameters()]
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    runtime.run_episode(task, RuntimeMode.TRAIN)
    assert all(not parameter.requires_grad for parameter in brain.model.parameters())
    assert all(torch.allclose(old, new) for old, new in zip(before, brain.model.parameters()))


def test_v03_trainable_modules_change_in_learning_smoke(tmp_path):
    runtime = _runtime(tmp_path, SequentialPatchBrain())
    before = [parameter.detach().clone() for parameter in runtime.policy.parameters()]
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    runtime.run_episode(task, RuntimeMode.TRAIN)
    assert runtime.policy.training_step > 0
    assert any(not torch.allclose(old, new) for old, new in zip(before, runtime.policy.parameters()))


def test_v03_attach_brain_uses_qwen_generator_and_encoder_override(tmp_path):
    runtime = JarvisRuntime(
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=20),
        data_dir=tmp_path / "runtime",
        sandbox_backend=LocalTestSandboxBackend(),
    )
    brain = StaticBrain(json.dumps([{"action_type": "READ_FILE", "path": "solution.py"}]))
    runtime.attach_brain(brain, semantic_text_encoder=DeterministicTextEncoder(embedding_dim=32))
    assert isinstance(runtime.action_generator, QwenActionGenerator)
    assert runtime.brain is brain
    assert runtime.text_encoder.embedding_dim == 32


def test_v03_productive_generators_do_not_contain_predefined_holdout_patch():
    prompt = QwenActionGenerator(StaticBrain("[]"))._prompt(
        "Repair the task.",
        CodingTaskFactory.__new__(CodingTaskFactory).__class__ if False else _dummy_observation(),
    )
    forbidden = ["return x + y", "return a + b", "safe_divide", "self.count += 1"]
    assert all(text not in prompt for text in forbidden)


def _dummy_observation():
    from environments.coding.observation import CodingObservation

    return CodingObservation(
        task_description="Repair a generic coding task.",
        workspace_tree=["solution.py", "test_public.py"],
        relevant_file_excerpts={},
        test_state={"ran": False, "passed": 0, "failed": 0},
        remaining_budget=8,
    )


def test_v03_persistent_experience_restores_across_restart(tmp_path):
    runtime_a = _runtime(tmp_path, SequentialPatchBrain())
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    runtime_a.run_episode(task, RuntimeMode.TRAIN)
    assert runtime_a.experience_store.count() > 0
    runtime_b = _runtime(tmp_path, SequentialPatchBrain())
    assert len(runtime_b.replay_buffer) > 0


def test_v03_checkpoint_promotion_prefers_success_over_reward(tmp_path):
    manager = RuntimeCheckpointManager(tmp_path / "checkpoints")
    best = {"success_rate": 0.5, "mean_reward": 1.0, "regression_rate": 0.0, "mean_steps_to_solution": 5.0}
    worse_success_higher_reward = {"success_rate": 0.25, "mean_reward": 99.0, "regression_rate": 0.0, "mean_steps_to_solution": 2.0}
    better_success = {"success_rate": 0.75, "mean_reward": 0.0, "regression_rate": 0.0, "mean_steps_to_solution": 8.0}
    assert not manager.should_promote(worse_success_higher_reward, best)
    assert manager.should_promote(better_success, best)


def test_v03_end_to_end_qwen_generated_patch_solves_synthetic_task(tmp_path):
    runtime = _runtime(tmp_path, SequentialPatchBrain(), mode=RuntimeMode.EVAL)
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    metrics = runtime.run_episode(task, RuntimeMode.EVAL)
    assert metrics["success"] is True
    assert isinstance(runtime.action_generator, QwenActionGenerator)
    assert any(
        transition.metadata["action"]["action_type"] == "PATCH_FILE"
        for transition in runtime.state.trajectory.transitions
    )


def test_v03_smoke_benchmark_entrypoint_uses_qwen_path_with_mock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = run_coding_brain_v03_demo(CodingBrainV03Config(quick=True, train_episodes=1, mock_brain=True, seed=7))
    if not metrics.get("SANDBOX_AVAILABLE"):
        return
    assert metrics["ACTION_GENERATOR"] == "QwenActionGenerator"
    assert metrics["QWEN_TRAINABLE"] is False
    assert "BASELINE_HOLDOUT_SUCCESS_RATE" in metrics
