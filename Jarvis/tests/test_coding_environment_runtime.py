import sys
from pathlib import Path

import torch

from environments.coding.actions import ActionCandidate, ActionEncoder, ActionType
from environments.coding.environment import CodingEnvironment
from environments.coding.observation import ObservationAdapter
from environments.coding.reward import CodingRewardEngine
from environments.coding.task import CodingTask
from runtime.jarvis_runtime import JarvisRuntime, JarvisRuntimeConfig
from runtime.runtime_state import RuntimeMode


class ScriptedActionGenerator:
    def generate(self, goal, observation):
        if not observation.test_state.get("ran", False):
            return [ActionCandidate(ActionType.RUN_TESTS, reasoning_summary="baseline", confidence=1.0)]
        if not observation.relevant_file_excerpts:
            return [ActionCandidate(ActionType.READ_FILE, {"path": "calculator.py"}, "read", 1.0)]
        if "return a - b" in observation.relevant_file_excerpts.get("calculator.py", ""):
            return [
                ActionCandidate(
                    ActionType.PATCH_FILE,
                    {"path": "calculator.py", "old": "return a - b", "new": "return a + b"},
                    "patch",
                    1.0,
                )
            ]
        return [ActionCandidate(ActionType.RUN_TESTS, reasoning_summary="verify", confidence=1.0)]


def make_task(tmp_path: Path, task_id="task", max_steps=8):
    workspace = tmp_path / task_id
    workspace.mkdir()
    (workspace / "calculator.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (workspace / "test_calculator.py").write_text(
        "import unittest\n"
        "from calculator import add\n\n"
        "class CalculatorTests(unittest.TestCase):\n"
        "    def test_adds(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    return CodingTask(
        description="Repair the addition bug in calculator.py.",
        workspace=workspace,
        test_command=[sys.executable, "-m", "unittest", "discover", "-v"],
        task_id=task_id,
        max_steps=max_steps,
    )


def test_coding_environment_isolates_paths_and_executes_actions(tmp_path):
    task = make_task(tmp_path)
    env = CodingEnvironment(task)

    list_step = env.step(ActionCandidate(ActionType.LIST_FILES))
    assert list_step.action_result.ok
    assert "calculator.py" in list_step.action_result.stdout

    escape = env.step(ActionCandidate(ActionType.READ_FILE, {"path": "../secret.txt"}))
    assert not escape.action_result.ok
    assert "Parent path traversal" in escape.action_result.message

    read_step = env.step(ActionCandidate(ActionType.READ_FILE, {"path": "calculator.py"}))
    assert read_step.action_result.ok
    assert "return a - b" in read_step.action_result.stdout

    patch_step = env.step(
        ActionCandidate(ActionType.PATCH_FILE, {"path": "calculator.py", "old": "return a - b", "new": "return a + b"})
    )
    assert patch_step.action_result.ok

    test_step = env.step(ActionCandidate(ActionType.RUN_TESTS))
    assert test_step.success
    assert test_step.done


def test_observation_and_action_encoding(tmp_path):
    task = make_task(tmp_path)
    env = CodingEnvironment(task)
    observation = env.observe()

    features = ObservationAdapter(feature_dim=24).encode(observation)
    action_embedding = ActionEncoder().encode(ActionCandidate(ActionType.RUN_TESTS))

    assert features.shape == (24,)
    assert action_embedding.shape == (len(ActionType),)
    assert action_embedding[ActionType.RUN_TESTS] == 1.0


def test_reward_from_test_improvement(tmp_path):
    task = make_task(tmp_path)
    env = CodingEnvironment(task)
    reward_engine = CodingRewardEngine()
    before = env.observe()
    env.step(ActionCandidate(ActionType.PATCH_FILE, {"path": "calculator.py", "old": "return a - b", "new": "return a + b"}))
    after_step = env.step(ActionCandidate(ActionType.RUN_TESTS))

    reward = reward_engine.compute(before, ActionCandidate(ActionType.RUN_TESTS), after_step)
    assert reward.components["R_tests"] > 0.0
    assert reward.total > 0.0


def test_runtime_step_transition_and_persistence(tmp_path):
    runtime = JarvisRuntime(
        action_generator=ScriptedActionGenerator(),
        config=JarvisRuntimeConfig(latent_dim=16, hidden_dim=16, replay_capacity=20, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime_data",
        mode=RuntimeMode.EVAL,
    )
    task = make_task(tmp_path, "runtime_task")
    runtime.start_task(task, RuntimeMode.EVAL)
    result = runtime.step(task.description)

    assert result.transition.latent_state.shape == (16,)
    assert result.transition.metadata["action"]["action_type"] == "RUN_TESTS"
    assert len(runtime.replay_buffer) == 1
    assert runtime.experience_store.count() == 1
    assert result.training_report.did_update is False


def test_train_eval_separation_and_online_parameter_update(tmp_path):
    runtime = JarvisRuntime(
        action_generator=ScriptedActionGenerator(),
        config=JarvisRuntimeConfig(latent_dim=16, hidden_dim=16, replay_capacity=50, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime_data",
        mode=RuntimeMode.TRAIN,
    )
    runtime.scheduler.config.world_model_batch_size = 1
    runtime.scheduler.config.value_policy_batch_size = 1
    runtime.scheduler.config.world_model_train_every_n_steps = 1
    runtime.scheduler.config.value_policy_train_every_n_steps = 1

    before_policy = [parameter.detach().clone() for parameter in runtime.policy.parameters()]
    eval_task = make_task(tmp_path, "eval_task")
    runtime.start_task(eval_task, RuntimeMode.EVAL)
    runtime.step(eval_task.description)
    assert runtime.policy.training_step == 0

    train_task = make_task(tmp_path, "train_task")
    runtime.start_task(train_task, RuntimeMode.TRAIN)
    runtime.step(train_task.description)

    assert runtime.policy.training_step > 0
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(before_policy, runtime.policy.parameters())
    )


def test_runtime_checkpoint_save_load(tmp_path):
    runtime = JarvisRuntime(
        action_generator=ScriptedActionGenerator(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10),
        data_dir=tmp_path / "runtime_data",
    )
    paths = runtime.save_checkpoints({"loss": 1.0})
    assert "Policy" in paths

    loaded = JarvisRuntime(
        action_generator=ScriptedActionGenerator(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10),
        data_dir=tmp_path / "runtime_data",
    )
    result = loaded.load_latest_checkpoints()
    assert result["Policy"]
    assert result["WorldModel"]


def test_end_to_end_coding_loop_terminates_and_updates(tmp_path):
    runtime = JarvisRuntime(
        action_generator=ScriptedActionGenerator(),
        config=JarvisRuntimeConfig(latent_dim=16, hidden_dim=16, replay_capacity=50, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime_data",
        mode=RuntimeMode.TRAIN,
    )
    runtime.scheduler.config.world_model_batch_size = 1
    runtime.scheduler.config.value_policy_batch_size = 1
    runtime.scheduler.config.world_model_train_every_n_steps = 1
    runtime.scheduler.config.value_policy_train_every_n_steps = 1
    task = make_task(tmp_path, "e2e_task", max_steps=6)

    metrics = runtime.run_episode(task, RuntimeMode.TRAIN)

    assert metrics["success"] is True
    assert metrics["steps"] <= 6
    assert len(runtime.state.trajectory.transitions) >= 3
    assert runtime.world_model.training_step > 0
    assert runtime.value_function.training_step > 0
    assert runtime.policy.training_step > 0


def test_max_step_termination(tmp_path):
    runtime = JarvisRuntime(
        action_generator=lambda_goal_observation_generator(ActionCandidate(ActionType.LIST_FILES)),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime_data",
        mode=RuntimeMode.EVAL,
    )
    task = make_task(tmp_path, "max_step_task", max_steps=2)
    metrics = runtime.run_episode(task, RuntimeMode.EVAL)
    assert metrics["success"] is False
    assert metrics["steps"] == 2


def lambda_goal_observation_generator(action):
    class SingleActionGenerator:
        def generate(self, goal, observation):
            return [action]

    return SingleActionGenerator()
