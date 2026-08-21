import json
import sys
import random
import copy
from dataclasses import replace

import torch

from environments.coding.actions import ActionCandidate, ActionType
from environments.coding.environment import CodingEnvironment
from environments.coding.reward import CodingRewardEngine
from environments.coding.sandbox_backend import LocalTestSandboxBackend
from environments.coding.task import CodingTask
from learning.representations.action_encoding import SemanticActionEncoder
from learning.representations.semantic import DeterministicTextEncoder
from runtime.action_generator import QwenActionGenerator
from runtime.checkpoints import RuntimeCheckpointManager
from runtime.jarvis_runtime import JarvisRuntime, JarvisRuntimeConfig
from runtime.runtime_state import RuntimeMode
from training.benchmark_progress import BenchmarkProgressStore
from training.coding_benchmark import CodingBenchmark
import training.coding_brain_v03_demo as v03_demo
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
        excerpts = lowered.split("excerpts:", 1)[-1]
        if "tests: {'ran': false" in lowered:
            return json.dumps([{"action_type": "RUN_TESTS", "arguments": {}, "reason": "baseline", "confidence": 0.9}])
        if "solution.py:" not in lowered:
            return json.dumps([{"action_type": "READ_FILE", "path": "solution.py", "reason": "inspect", "confidence": 0.9}])
        if "solution.py:" in excerpts and "return a - b" in excerpts:
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
    assert len(candidates) == 4
    assert generator.last_generation_metadata["duplicate_candidates"] == 1
    assert generator.last_generation_metadata["fallback_backfill_count"] == 2


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
    hidden = env.run_final_hidden_verifier()
    assert result.success
    assert hidden["success"] is False
    assert "hidden_failed" not in result.observation.test_state


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


def test_v03_holdout_prompt_contains_no_hidden_oracle_feedback(tmp_path):
    captured = {}

    def response(prompt):
        captured["prompt"] = prompt
        return json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}])

    runtime = _runtime(tmp_path, StaticBrain(response), mode=RuntimeMode.EVAL)
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    runtime.run_episode(task, RuntimeMode.EVAL)
    prompt = captured["prompt"]
    forbidden = ["hidden_passed", "hidden_failed", "hidden_ran", "Hidden verifier failed", "hidden_verifier"]
    assert all(text not in prompt for text in forbidden)
    assert all(text not in runtime.environment.observe().to_text() for text in forbidden)


def test_v03_holdout_hidden_verifier_runs_once_after_episode(tmp_path):
    runtime = _runtime(tmp_path, SequentialPatchBrain(), mode=RuntimeMode.EVAL)
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    runtime.run_episode(task, RuntimeMode.EVAL)
    assert runtime.environment.hidden_state["runs"] == 0
    first = runtime.final_hidden_verification()
    second = runtime.final_hidden_verification()
    assert first["runs"] == 1
    assert second["runs"] == 1


def test_v03_holdout_evaluations_use_pristine_regenerated_workspaces(tmp_path):
    baseline_factory = CodingTaskFactory(tmp_path / "holdout_baseline")
    final_factory = CodingTaskFactory(tmp_path / "holdout_final")
    baseline_task = baseline_factory.make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    original_solution = (baseline_task.workspace / "solution.py").read_bytes()
    original_hidden = (baseline_task.hidden_workspace / "hidden_verifier.py").read_bytes()

    (baseline_task.workspace / "solution.py").write_text("def contaminated():\n    return 'baseline'\n", encoding="utf-8")
    (baseline_task.hidden_workspace / "hidden_verifier.py").write_text("raise AssertionError('mutated hidden verifier')\n", encoding="utf-8")

    final_task = final_factory.make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    assert final_task.workspace != baseline_task.workspace
    assert final_task.hidden_workspace != baseline_task.hidden_workspace
    assert (final_task.workspace / "solution.py").read_bytes() == original_solution
    assert (final_task.hidden_workspace / "hidden_verifier.py").read_bytes() == original_hidden
    assert (baseline_task.workspace / "solution.py").read_bytes() != (final_task.workspace / "solution.py").read_bytes()
    assert (baseline_task.hidden_workspace / "hidden_verifier.py").read_bytes() != (final_task.hidden_workspace / "hidden_verifier.py").read_bytes()


def _public_hidden_truth_task(tmp_path, public_pass: bool, hidden_pass: bool) -> CodingTask:
    label = f"public_{public_pass}_hidden_{hidden_pass}"
    workspace = tmp_path / label / "workspace"
    hidden_workspace = tmp_path / label / "hidden"
    workspace.mkdir(parents=True)
    hidden_workspace.mkdir(parents=True)
    public_expected = 1 if public_pass else 99
    hidden_expected = 1 if hidden_pass else 99
    (workspace / "solution.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (workspace / "test_public.py").write_text(
        "import unittest\n"
        "from solution import value\n\n"
        "class PublicTests(unittest.TestCase):\n"
        "    def test_value(self):\n"
        f"        self.assertEqual(value(), {public_expected})\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    (hidden_workspace / "hidden_verifier.py").write_text(
        "from solution import value\n"
        f"assert value() == {hidden_expected}\n",
        encoding="utf-8",
    )
    return CodingTask(
        description="Run public tests and external hidden verifier.",
        workspace=workspace,
        test_command=[sys.executable, "-m", "unittest", "discover", "-v"],
        hidden_workspace=hidden_workspace,
        hidden_test_command=[sys.executable, "hidden_verifier.py"],
        protected_paths={"test_public.py"},
        task_id=label,
        max_steps=1,
    )


def test_v03_benchmark_success_requires_public_and_hidden_pass(tmp_path):
    class RunTestsOnlyGenerator:
        last_generation_metadata = {}

        def generate(self, goal, observation):
            return [ActionCandidate(ActionType.RUN_TESTS)]

    cases = [
        (False, True, False, 0.0),
        (True, False, False, 1.0),
        (True, True, True, 1.0),
        (False, False, False, 0.0),
    ]
    for public_pass, hidden_pass, expected_success, expected_hidden_runs in cases:
        runtime = _runtime(tmp_path / f"runtime_{public_pass}_{hidden_pass}", StaticBrain(json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}])), mode=RuntimeMode.EVAL)
        runtime.action_generator = RunTestsOnlyGenerator()
        task = _public_hidden_truth_task(tmp_path, public_pass, hidden_pass)
        result = CodingBenchmark().evaluate(runtime, [task])
        assert result.success_rate == (1.0 if expected_success else 0.0)
        assert result.hidden_verifier_runs == expected_hidden_runs


def test_v03_benchmark_public_success_is_success_without_hidden_verifier(tmp_path):
    workspace = tmp_path / "public_only"
    workspace.mkdir()
    (workspace / "solution.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (workspace / "test_public.py").write_text(
        "import unittest\nfrom solution import value\n\n"
        "class PublicTests(unittest.TestCase):\n"
        "    def test_value(self):\n"
        "        self.assertEqual(value(), 1)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    task = CodingTask(
        description="Run public tests only.",
        workspace=workspace,
        test_command=[sys.executable, "-m", "unittest", "discover", "-v"],
        task_id="public_only_success",
        max_steps=1,
    )
    runtime = _runtime(tmp_path / "runtime_public_only", StaticBrain(json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}])), mode=RuntimeMode.EVAL)
    class RunTestsOnlyGenerator:
        last_generation_metadata = {}

        def generate(self, goal, observation):
            return [ActionCandidate(ActionType.RUN_TESTS)]

    runtime.action_generator = RunTestsOnlyGenerator()
    result = CodingBenchmark().evaluate(runtime, [task])
    assert result.success_rate == 1.0
    assert result.hidden_verifier_runs == 0.0


def test_v03_benchmark_invalid_action_rate_counts_transitions(tmp_path):
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    task.max_steps = 2
    runtime = _runtime(
        tmp_path / "runtime_invalid",
        StaticBrain(json.dumps([{"action_type": "READ_FILE", "arguments": {"path": "missing.py"}}])),
        mode=RuntimeMode.EVAL,
    )
    class InvalidOnlyGenerator:
        last_generation_metadata = {}

        def generate(self, goal, observation):
            return [ActionCandidate(ActionType.READ_FILE, {"path": "missing.py"})]

    runtime.action_generator = InvalidOnlyGenerator()
    result = CodingBenchmark().evaluate(runtime, [task])
    assert result.invalid_action_rate == 0.0
    assert result.episodes_with_invalid_action_rate == 0.0
    assert all(
        transition.metadata["scoring"]["feasible"] is True
        for transition in runtime.state.trajectory.transitions
    )


def _weak_public_hidden_task(tmp_path, task_id: str = "weak_public") -> CodingTask:
    workspace = tmp_path / task_id / "workspace"
    hidden_workspace = tmp_path / task_id / "hidden"
    workspace.mkdir(parents=True)
    hidden_workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (workspace / "test_public.py").write_text(
        "import unittest\n\n"
        "class PublicTests(unittest.TestCase):\n"
        "    def test_public_is_weak(self):\n"
        "        self.assertTrue(True)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    (hidden_workspace / "hidden_verifier.py").write_text("from solution import value\nassert value() == 2\n", encoding="utf-8")
    return CodingTask(
        description="Weak public tests pass before the hidden requirement is satisfied.",
        workspace=workspace,
        test_command=[sys.executable, "-m", "unittest", "discover", "-v"],
        hidden_workspace=hidden_workspace,
        hidden_test_command=[sys.executable, "hidden_verifier.py"],
        protected_paths={"test_public.py"},
        task_id=task_id,
        max_steps=6,
    )


class SequenceActionGenerator:
    last_generation_metadata = {}

    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def generate(self, goal, observation):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return [action]


def test_v03_eval_public_pass_does_not_stop_and_agent_can_continue_patch(tmp_path):
    task = _weak_public_hidden_task(tmp_path)
    runtime = _runtime(tmp_path / "runtime_eval_continue", StaticBrain("[]"), mode=RuntimeMode.EVAL)
    runtime.action_generator = SequenceActionGenerator(
        [
            ActionCandidate(ActionType.RUN_TESTS),
            ActionCandidate(ActionType.READ_FILE, {"path": "solution.py"}),
            ActionCandidate(ActionType.PATCH_FILE, {"path": "solution.py", "old": "return 1", "new": "return 2"}),
            ActionCandidate(ActionType.RUN_TESTS),
            ActionCandidate(ActionType.FINISH),
        ]
    )
    metrics = runtime.run_episode(task, RuntimeMode.EVAL)
    actions = [transition.metadata["action"]["action_type"] for transition in runtime.state.trajectory.transitions]
    assert metrics["steps"] == 5
    assert actions == ["RUN_TESTS", "READ_FILE", "PATCH_FILE", "RUN_TESTS", "FINISH"]
    assert runtime.environment.hidden_state["runs"] == 0
    hidden = runtime.final_hidden_verification()
    assert hidden["success"] is True
    assert runtime.final_hidden_verification()["runs"] == 1
    observation_text = runtime.environment.observe().to_text()
    assert "hidden_verifier" not in observation_text
    assert "hidden_passed" not in observation_text


def test_v03_eval_finish_stops_after_public_pass(tmp_path):
    task = _weak_public_hidden_task(tmp_path, "finish_stops")
    runtime = _runtime(tmp_path / "runtime_finish", StaticBrain("[]"), mode=RuntimeMode.EVAL)
    runtime.action_generator = SequenceActionGenerator(
        [
            ActionCandidate(ActionType.RUN_TESTS),
            ActionCandidate(ActionType.FINISH),
            ActionCandidate(ActionType.READ_FILE, {"path": "solution.py"}),
        ]
    )
    metrics = runtime.run_episode(task, RuntimeMode.EVAL)
    assert metrics["steps"] == 2
    actions = [transition.metadata["action"]["action_type"] for transition in runtime.state.trajectory.transitions]
    assert actions == ["RUN_TESTS", "FINISH"]


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


def test_v03_checkpoint_promotion_uses_validation_not_holdout(tmp_path):
    manager = RuntimeCheckpointManager(tmp_path / "checkpoints")
    best = {"split": "validation", "success_rate": 0.5, "mean_reward": 1.0, "regression_rate": 0.0, "mean_steps_to_solution": 5.0}
    holdout_metrics = {"split": "holdout", "success_rate": 1.0, "mean_reward": 999.0, "regression_rate": 0.0, "mean_steps_to_solution": 1.0}
    validation_metrics = {"split": "validation", "success_rate": 0.6, "mean_reward": 0.0, "regression_rate": 0.5, "mean_steps_to_solution": 9.0}
    assert not manager.should_promote(holdout_metrics, best)
    assert manager.should_promote(validation_metrics, best)


def test_v03_best_snapshot_survives_worse_latest_and_load_best(tmp_path):
    runtime = _runtime(tmp_path, SequentialPatchBrain())
    for parameter in runtime.policy.parameters():
        parameter.data.fill_(1.0)
    runtime.save_checkpoints({"success_rate": 1.0, "regression_rate": 0.0, "mean_steps_to_solution": 2.0}, category="latest")
    runtime.save_checkpoints({"success_rate": 1.0, "regression_rate": 0.0, "mean_steps_to_solution": 2.0}, category="best")
    best_policy = [parameter.detach().clone() for parameter in runtime.policy.parameters()]

    for parameter in runtime.policy.parameters():
        parameter.data.fill_(2.0)
    runtime.save_checkpoints({"success_rate": 0.0, "regression_rate": 0.0, "mean_steps_to_solution": 8.0}, category="latest")

    latest_loaded = _runtime(tmp_path / "latest_load", SequentialPatchBrain())
    latest_loaded.checkpoint_manager = runtime.checkpoint_manager
    latest_loaded.load_latest_checkpoints()
    assert all(torch.allclose(parameter, torch.full_like(parameter, 2.0)) for parameter in latest_loaded.policy.parameters())

    best_loaded = _runtime(tmp_path / "best_load", SequentialPatchBrain())
    best_loaded.checkpoint_manager = runtime.checkpoint_manager
    best_loaded.load_best_checkpoints()
    assert all(torch.allclose(old, new) for old, new in zip(best_policy, best_loaded.policy.parameters()))


def _score_eval_candidates(runtime, task):
    observation = runtime.start_task(task, RuntimeMode.EVAL)
    candidates = [
        ActionCandidate(ActionType.RUN_TESTS, confidence=0.2, estimated_cost=2.0),
        ActionCandidate(ActionType.READ_FILE, {"path": "solution.py"}, confidence=0.9, estimated_cost=0.5),
    ]
    features = runtime._observation_features(observation)
    latent = runtime._encode_features(features)
    return runtime._score_candidates(latent, candidates, observation)


def test_v03_controller_ablation_modes_are_eval_only_and_observable(tmp_path):
    runtime = _runtime(tmp_path / "controller_modes", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    runtime.config = replace(runtime.config, learned_gate_min_experiences=0, learned_gate_warmup_experiences=1)
    train_task = CodingTaskFactory(tmp_path / "train_tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    runtime.run_episode(train_task, RuntimeMode.TRAIN)
    assert len(runtime.replay_buffer) > 0
    task = CodingTaskFactory(tmp_path / "eval_tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    for mode in CodingBenchmark.CONTROLLER_MODES:
        runtime.config = replace(runtime.config, eval_controller=mode)
        scored = _score_eval_candidates(runtime, task)
        assert all(item.controller_mode == mode for item in scored)
        assert all("policy" in item.raw_score_components for item in scored)
        assert all("q" in item.normalized_score_components for item in scored)
        if mode == "heuristic":
            assert all(item.learned_score == 0.0 for item in scored)
            assert all(item.controller_gate == 0.0 for item in scored)


def test_v03_safe_learned_override_gate_depends_on_maturity(tmp_path):
    runtime = _runtime(tmp_path / "gate", SequentialPatchBrain(), mode=RuntimeMode.EVAL)
    runtime.config = replace(runtime.config, learned_gate_min_experiences=10, learned_gate_warmup_experiences=20)
    assert runtime._learned_controller_gate() == 0.0
    runtime.config = replace(runtime.config, learned_gate_min_experiences=0, learned_gate_warmup_experiences=1)
    train_task = CodingTaskFactory(tmp_path / "gate_tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    runtime.run_episode(train_task, RuntimeMode.TRAIN)
    assert runtime._learned_controller_gate() > 0.0


def test_v03_component_normalization_clips_large_scales(tmp_path):
    runtime = _runtime(tmp_path / "normalization", SequentialPatchBrain(), mode=RuntimeMode.EVAL)
    runtime.config = replace(runtime.config, learned_component_clip=1.0)
    values = runtime._normalize_component([-1000.0, 0.0, 1000.0])
    assert max(abs(value) for value in values) <= 1.0
    assert sum(values) == 0.0


def test_v03_controller_disagreement_diagnostics_are_aggregated():
    diagnostics = CodingBenchmark()._controller_diagnostics(
        [
            {
                "heuristic_winner": True,
                "controller_changed_heuristic": True,
                "heuristic_score": 1.0,
                "policy_score": -0.5,
                "q_score": 0.25,
                "value_score": 0.0,
                "world_score": 0.1,
                "learned_score": 0.75,
                "controller_gate": 0.5,
            },
            {
                "heuristic_winner": False,
                "controller_changed_heuristic": True,
                "heuristic_score": 0.2,
                "policy_score": -0.1,
                "q_score": 0.75,
                "value_score": 0.3,
                "world_score": 0.0,
                "learned_score": 0.4,
                "controller_gate": 0.5,
            },
        ],
        [1.0],
        [1.0],
    )
    assert diagnostics["action_selection_disagreement_rate_vs_heuristic"] == 1.0
    assert diagnostics["success_rate_when_learned_changed_episode"] == 1.0
    assert diagnostics["mean_abs_contribution"]["q_score"] > 0.0


def test_v03_world_gradients_are_finite_and_clipped(tmp_path):
    runtime = _runtime(tmp_path / "gradients", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    runtime.scheduler.config.gradient_clip_norm = 0.05
    linear = torch.nn.Linear(2, 1)
    for parameter in linear.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)
    norm_before = runtime.scheduler._clip(linear.parameters())
    assert torch.isfinite(torch.tensor(norm_before))
    clipped_norm = sum(float(parameter.grad.detach().norm().item()) ** 2 for parameter in linear.parameters()) ** 0.5
    assert clipped_norm <= runtime.scheduler.config.gradient_clip_norm + 1e-5
    task = CodingTaskFactory(tmp_path / "gradient_tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    runtime.run_episode(task, RuntimeMode.TRAIN)
    assert torch.isfinite(torch.tensor(runtime.state.latest_metrics.get("world_gradient_norm", 0.0)))


def test_v03_q_value_calibration_metrics_are_available(tmp_path):
    runtime = _runtime(tmp_path / "calibration", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    task = CodingTaskFactory(tmp_path / "calibration_tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    runtime.run_episode(task, RuntimeMode.TRAIN)
    diagnostics = runtime.scheduler.calibration_diagnostics(runtime.replay_buffer)
    assert -1.0 <= diagnostics["q_return_rank_correlation"] <= 1.0
    assert -1.0 <= diagnostics["value_return_correlation"] <= 1.0


def test_v03_benchmark_controller_suite_uses_fresh_tasks(tmp_path):
    runtime = _runtime(tmp_path / "suite", StaticBrain(json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}])), mode=RuntimeMode.EVAL)
    factory_root = tmp_path / "suite_tasks"
    result = CodingBenchmark().evaluate_controller_suite(
        runtime,
        lambda mode: CodingTaskFactory(factory_root / mode).make_v03_split_tasks(DatasetSplit.HOLDOUT, 1),
        modes=["heuristic", "full"],
    )
    assert set(result) == {"heuristic", "full"}
    assert result["heuristic"]["controller_diagnostics"]["candidate_count"] > 0
    assert (factory_root / "heuristic").exists()
    assert (factory_root / "full").exists()


def test_v03_progress_store_persists_episode_checkpoint_and_resume_state(tmp_path):
    store = BenchmarkProgressStore(tmp_path / "progress.json")
    store.save(stage="train", train_completed=3, metrics={"reward": 1.5})
    resumed = BenchmarkProgressStore(tmp_path / "progress.json")
    assert resumed.state["stage"] == "train"
    assert resumed.state["train_completed"] == 3
    assert resumed.state["metrics"]["reward"] == 1.5
    assert resumed.state["python_random_state"]
    assert resumed.state["torch_rng_state"]


def test_v03_training_resume_checkpoint_restores_rng_replay_and_counters(tmp_path):
    runtime = _runtime(tmp_path / "resume_a", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    task = CodingTaskFactory(tmp_path / "resume_tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    task.max_steps = 2
    runtime.run_episode(task, RuntimeMode.TRAIN)
    checkpoint = runtime.save_training_resume_checkpoint(tmp_path / "resume.pt", train_completed=1)
    expected_python = __import__("random").random()
    expected_torch = torch.rand(3)
    expected_runtime_rng = runtime._rng.random()

    restored = _runtime(tmp_path / "resume_b", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    payload = restored.load_training_resume_checkpoint(checkpoint, expected_train_completed=1)
    assert payload["train_completed"] == 1
    assert len(restored.replay_buffer) == len(runtime.replay_buffer)
    assert restored.scheduler.runtime_steps == runtime.scheduler.runtime_steps
    assert restored.policy.training_step == runtime.policy.training_step
    assert __import__("random").random() == expected_python
    assert torch.allclose(torch.rand(3), expected_torch)
    assert restored._rng.random() == expected_runtime_rng


def test_v03_resume_checkpoint_refuses_progress_mismatch(tmp_path):
    runtime = _runtime(tmp_path / "resume_mismatch", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    checkpoint = runtime.save_training_resume_checkpoint(tmp_path / "resume.pt", train_completed=2)
    restored = _runtime(tmp_path / "resume_mismatch_b", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    try:
        restored.load_training_resume_checkpoint(checkpoint, expected_train_completed=3)
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched progress/checkpoint pair should fail")


def _train_tasks_for_resume(root, count=6):
    tasks = CodingTaskFactory(root).make_v03_split_tasks(DatasetSplit.TRAIN, count)
    for task in tasks:
        task.max_steps = 2
    return tasks


def _state_signature(runtime):
    return {
        "policy": [parameter.detach().clone() for parameter in runtime.policy.parameters()],
        "value": [parameter.detach().clone() for parameter in runtime.value_function.parameters()],
        "world": [parameter.detach().clone() for parameter in runtime.world_model.parameters()],
        "q": [parameter.detach().clone() for parameter in runtime.action_value.parameters()],
        "replay": len(runtime.replay_buffer),
        "runtime_steps": runtime.scheduler.runtime_steps,
    }


def _force_greedy_selection(runtime):
    runtime._select = lambda scored: ([item for item in scored if item.feasible] or scored)[0]


def test_v03_interrupted_then_resumed_training_matches_uninterrupted(tmp_path):
    uninterrupted = _runtime(tmp_path / "uninterrupted", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    _force_greedy_selection(uninterrupted)
    uninterrupted_tasks = _train_tasks_for_resume(tmp_path / "tasks_uninterrupted")
    for task in uninterrupted_tasks:
        uninterrupted.run_episode(task, RuntimeMode.TRAIN)
    uninterrupted_signature = _state_signature(uninterrupted)

    partial = _runtime(tmp_path / "partial", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    _force_greedy_selection(partial)
    partial_tasks = _train_tasks_for_resume(tmp_path / "tasks_partial")
    for task in partial_tasks[:3]:
        partial.run_episode(task, RuntimeMode.TRAIN)
    checkpoint = partial.save_training_resume_checkpoint(tmp_path / "train_episode_0003.pt", train_completed=3)
    resumed = _runtime(tmp_path / "resumed", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    _force_greedy_selection(resumed)
    resumed.load_training_resume_checkpoint(checkpoint, expected_train_completed=3)
    for task in partial_tasks[3:]:
        resumed.run_episode(task, RuntimeMode.TRAIN)
    resumed_signature = _state_signature(resumed)

    assert resumed_signature["replay"] == uninterrupted_signature["replay"]
    assert resumed_signature["runtime_steps"] == uninterrupted_signature["runtime_steps"]
    for key in ["policy", "value", "world", "q"]:
        max_diff = max(float((left - right).abs().max().item()) for left, right in zip(resumed_signature[key], uninterrupted_signature[key]))
        assert max_diff < 2e-2


def test_v03_resumable_eval_stage_uses_completed_index_and_accumulators(tmp_path):
    runtime = _runtime(tmp_path / "stage_resume", StaticBrain(json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}])), mode=RuntimeMode.EVAL)
    benchmark = CodingBenchmark()
    progress = BenchmarkProgressStore(tmp_path / "progress.json")
    tasks = CodingTaskFactory(tmp_path / "stage_tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 2)
    for task in tasks:
        task.max_steps = 1
    partial_one = benchmark.evaluate(runtime, tasks[:1], eval_controller="heuristic", start_index=0)
    for stage, completed_key, result_key in [
        ("baseline", "baseline_completed", "baseline"),
        ("validation", "validation_completed", "validation"),
        ("final_holdout", "final_holdout_completed", "final"),
    ]:
        progress.save(stage=stage, **{completed_key: 1}, metrics={f"{result_key}_partial": partial_one.to_dict()})
        resumed = v03_demo._evaluate_resumable_stage(
            benchmark=benchmark,
            runtime=runtime,
            tasks=tasks,
            progress=progress,
            result_key=result_key,
            progress_completed_key=completed_key,
            stage_name=stage,
            controller="heuristic",
            resume=True,
        )
        assert resumed.episodes == 2
        assert progress.state[completed_key] == 2


def _three_episode_accumulators():
    return {
        "rewards": [1.0, 2.0, 3.0],
        "steps": [1.0, 1.0, 1.0],
        "successes": [1.0, 0.0, 1.0],
        "tests_passed": [0.0, 0.0, 0.0],
        "regressions": [0.0, 1.0, 0.0],
        "invalids": [],
        "prediction_losses": [0.0, 0.0, 0.0],
        "value_errors": [0.0, 0.0, 0.0],
        "q_errors": [0.0, 0.0, 0.0],
        "hidden_runs": [0.0, 0.0, 0.0],
        "episode_invalids": [0.0, 0.0, 0.0],
        "controller_scores": [],
        "episode_changed": [0.0, 0.0, 0.0],
        "episode_success_when_changed": [],
    }


class SpyResumeBenchmark(CodingBenchmark):
    def __init__(self):
        super().__init__()
        self.seen_start_index = None

    def evaluate(self, runtime, tasks, *, eval_controller=None, after_episode=None, start_index=0, initial_accumulators=None):
        self.seen_start_index = start_index
        result = self.result_from_accumulators(initial_accumulators)
        if after_episode is not None and start_index < len(tasks):
            after_episode(start_index, result)
        return result


def _assert_mid_episode_interrupt_preserves_committed_rng(tmp_path, stage_name, completed_key, result_key):
    runtime = _runtime(tmp_path / f"{stage_name}_runtime", StaticBrain(json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}])), mode=RuntimeMode.EVAL)
    progress = BenchmarkProgressStore(tmp_path / f"{stage_name}_progress.json")
    partial = CodingBenchmark().result_from_accumulators(_three_episode_accumulators())
    progress.save(
        runtime=runtime,
        commit_rng=True,
        stage=stage_name,
        **{completed_key: 3},
        metrics={f"{result_key}_partial": partial.to_dict()},
    )
    committed = copy.deepcopy(progress.state["committed_rng"])
    BenchmarkProgressStore.restore_rng_snapshot(committed, runtime=runtime)
    expected_python = random.random()
    expected_torch = torch.rand(3)
    expected_runtime = runtime._rng.random()

    random.random()
    torch.rand(5)
    runtime._rng.random()
    progress.save(stage="interrupted", interrupted=True, interrupted_from=stage_name, commit_rng=False, capture_live_rng=True)

    assert progress.state[completed_key] == 3
    assert progress.state["committed_rng"] == committed
    assert progress.state["live_rng"] != committed

    restored_runtime = _runtime(tmp_path / f"{stage_name}_restored", StaticBrain(json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}])), mode=RuntimeMode.EVAL)
    v03_demo._restore_committed_rng_after_runtime(progress, restored_runtime)
    assert random.random() == expected_python
    assert torch.allclose(torch.rand(3), expected_torch)
    assert restored_runtime._rng.random() == expected_runtime

    spy = SpyResumeBenchmark()
    tasks = CodingTaskFactory(tmp_path / f"{stage_name}_tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 4)
    v03_demo._evaluate_resumable_stage(
        benchmark=spy,
        runtime=restored_runtime,
        tasks=tasks,
        progress=progress,
        result_key=result_key,
        progress_completed_key=completed_key,
        stage_name=stage_name,
        controller="heuristic",
        resume=True,
    )
    assert spy.seen_start_index == 3


def test_v03_final_mid_episode_interrupt_preserves_committed_rng_and_resumes_at_episode_four(tmp_path):
    _assert_mid_episode_interrupt_preserves_committed_rng(
        tmp_path,
        "final_holdout",
        "final_holdout_completed",
        "final",
    )


def test_v03_validation_mid_episode_interrupt_preserves_committed_rng_and_resumes_at_episode_four(tmp_path):
    _assert_mid_episode_interrupt_preserves_committed_rng(
        tmp_path,
        "validation",
        "validation_completed",
        "validation",
    )


def test_v03_eval_committed_rng_is_restored_after_training_checkpoint(tmp_path):
    training_runtime = _runtime(tmp_path / "ordering_train", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    task = CodingTaskFactory(tmp_path / "ordering_train_tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    task.max_steps = 2
    training_runtime.run_episode(task, RuntimeMode.TRAIN)
    checkpoint = training_runtime.save_training_resume_checkpoint(tmp_path / "resume.pt", train_completed=1)

    eval_runtime = _runtime(tmp_path / "ordering_eval", StaticBrain(json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}])), mode=RuntimeMode.EVAL)
    progress = BenchmarkProgressStore(tmp_path / "ordering_progress.json")
    progress.save(runtime=eval_runtime, commit_rng=True, stage="final_holdout", train_completed=1, resume_checkpoint=str(checkpoint), final_holdout_completed=3)
    committed = copy.deepcopy(progress.state["committed_rng"])
    BenchmarkProgressStore.restore_rng_snapshot(committed, runtime=eval_runtime)
    expected_python = random.random()
    expected_torch = torch.rand(2)
    expected_runtime = eval_runtime._rng.random()

    restored = _runtime(tmp_path / "ordering_restored", StaticBrain(json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}])), mode=RuntimeMode.EVAL)
    restored.load_training_resume_checkpoint(checkpoint, expected_train_completed=1)
    v03_demo._restore_committed_rng_after_runtime(progress, restored)
    assert random.random() == expected_python
    assert torch.allclose(torch.rand(2), expected_torch)
    assert restored._rng.random() == expected_runtime


def test_v03_ctrl_c_partial_save(tmp_path, monkeypatch):
    def interrupt(config):
        raise KeyboardInterrupt()

    monkeypatch.setattr(v03_demo, "_run_coding_brain_v03_demo_impl", interrupt)
    metrics = run_coding_brain_v03_demo(CodingBrainV03Config(benchmark_dir=str(tmp_path), quiet=True))
    assert metrics["INTERRUPTED"] is True
    assert (tmp_path / "benchmark_results" / "coding_brain_v03_progress.json").exists()


def test_v03_suite_only_ablation_does_not_update_trainable_parameters(tmp_path):
    runtime = _runtime(tmp_path / "suite_only_runtime", SequentialPatchBrain(), mode=RuntimeMode.TRAIN)
    task = CodingTaskFactory(tmp_path / "suite_only_train").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    task.max_steps = 2
    runtime.run_episode(task, RuntimeMode.TRAIN)
    runtime.save_checkpoints({"split": "validation", "success_rate": 1.0, "regression_rate": 0.0, "mean_steps_to_solution": 1.0}, category="best")
    before = _state_signature(runtime)
    result = CodingBenchmark().evaluate_controller_suite(
        runtime,
        lambda mode: CodingTaskFactory(tmp_path / f"suite_only_{mode}").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1),
    )
    after = _state_signature(runtime)
    assert set(result) == set(CodingBenchmark.CONTROLLER_MODES)
    assert after["runtime_steps"] >= before["runtime_steps"]
    for key in ["policy", "value", "world", "q"]:
        assert all(torch.allclose(left, right) for left, right in zip(before[key], after[key]))


def test_v03_suite_only_cli_path_loads_checkpoint_without_optimizer_updates(tmp_path):
    config = CodingBrainV03Config(quick=True, mock_brain=True, benchmark_dir=str(tmp_path), quiet=True, max_steps=1)
    brain, encoder = v03_demo._make_brain(config)
    runtime = v03_demo._make_runtime(config, tmp_path / "runtime", brain, encoder)
    runtime.save_checkpoints({"split": "validation", "success_rate": 0.5, "regression_rate": 0.0, "mean_steps_to_solution": 1.0}, category="best")
    result = run_coding_brain_v03_demo(
        CodingBrainV03Config(
            quick=True,
            mock_brain=True,
            benchmark_dir=str(tmp_path),
            quiet=True,
            max_steps=1,
            eval_controller_suite_only=True,
        )
    )
    if not result.get("SANDBOX_AVAILABLE"):
        return
    assert result["SUITE_ONLY"] is True
    assert set(result["CONTROLLER_ABLATIONS"]) == set(CodingBenchmark.CONTROLLER_MODES)
    assert all(value == "NO" for value in result["PARAMETERS_CHANGED"].values())
    assert result["TRAINING_STEPS_BEFORE"] == result["TRAINING_STEPS_AFTER"]


def test_v03_full_split_fingerprints_are_unique_and_disjoint(tmp_path):
    factory = CodingTaskFactory(tmp_path / "tasks")
    splits = {
        DatasetSplit.TRAIN: factory.make_v03_split_tasks(DatasetSplit.TRAIN),
        DatasetSplit.VALIDATION: factory.make_v03_split_tasks(DatasetSplit.VALIDATION),
        DatasetSplit.HOLDOUT: factory.make_v03_split_tasks(DatasetSplit.HOLDOUT),
    }
    fingerprints = {split: [factory.structural_fingerprint(task) for task in tasks] for split, tasks in splits.items()}
    assert len(fingerprints[DatasetSplit.TRAIN]) == 30
    assert len(fingerprints[DatasetSplit.VALIDATION]) == 10
    assert len(fingerprints[DatasetSplit.HOLDOUT]) == 20
    for values in fingerprints.values():
        assert len(values) == len(set(values))
    assert set(fingerprints[DatasetSplit.TRAIN]).isdisjoint(fingerprints[DatasetSplit.VALIDATION])
    assert set(fingerprints[DatasetSplit.TRAIN]).isdisjoint(fingerprints[DatasetSplit.HOLDOUT])
    assert set(fingerprints[DatasetSplit.VALIDATION]).isdisjoint(fingerprints[DatasetSplit.HOLDOUT])


def test_v03_catalog_pristine_public_tests_all_fail_and_metadata_is_complete(tmp_path):
    factory = CodingTaskFactory(tmp_path / "catalog")
    failures = []
    total = 0
    for split in DatasetSplit:
        for task in factory.make_v03_split_tasks(split):
            total += 1
            assert task.description.strip()
            assert (task.workspace / "test_public.py").exists()
            assert task.hidden_workspace is not None
            assert task.hidden_test_command is not None
            assert "test_public.py" in task.protected_paths
            env = CodingEnvironment(task, backend=LocalTestSandboxBackend(), terminate_on_public_success=False)
            result = env.step(ActionCandidate(ActionType.RUN_TESTS))
            if result.success:
                failures.append((split.value, task.task_id, task.metadata.get("family")))
    assert total == 60
    assert failures == []


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
    assert metrics["TRAIN_TASK_COUNT"] == 1
    assert metrics["VALIDATION_TASK_COUNT"] == 2
    assert metrics["HOLDOUT_TASK_COUNT"] == 3
