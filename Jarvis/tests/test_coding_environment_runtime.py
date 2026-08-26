import sys
from pathlib import Path

import torch

from environments.coding.actions import ActionCandidate, ActionEncoder, ActionType
from environments.coding.environment import CodingEnvironment
from environments.coding.observation import ObservationAdapter
from environments.coding.reward import CodingRewardEngine
from environments.coding.sandbox_backend import LocalTestSandboxBackend
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
    env = CodingEnvironment(task, backend=LocalTestSandboxBackend())

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
    env = CodingEnvironment(task, backend=LocalTestSandboxBackend())
    observation = env.observe()

    features = ObservationAdapter(feature_dim=24).encode(observation)
    action_embedding = ActionEncoder().encode(ActionCandidate(ActionType.RUN_TESTS))

    assert features.shape == (24,)
    assert action_embedding.shape == (len(ActionType),)
    assert action_embedding[ActionType.RUN_TESTS] == 1.0


def test_reward_from_test_improvement(tmp_path):
    task = make_task(tmp_path)
    env = CodingEnvironment(task, backend=LocalTestSandboxBackend())
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
        sandbox_backend=LocalTestSandboxBackend(),
    )
    task = make_task(tmp_path, "runtime_task")
    runtime.start_task(task, RuntimeMode.EVAL)
    result = runtime.step(task.description)

    assert result.transition.latent_state.shape == (16,)
    assert result.transition.metadata["action"]["action_type"] == "RUN_TESTS"
    assert len(runtime.replay_buffer) == 0
    assert runtime.experience_store.count() == 0
    assert result.training_report.did_update is False


def test_train_eval_separation_and_online_parameter_update(tmp_path):
    runtime = JarvisRuntime(
        action_generator=ScriptedActionGenerator(),
        config=JarvisRuntimeConfig(latent_dim=16, hidden_dim=16, replay_capacity=50, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime_data",
        mode=RuntimeMode.TRAIN,
        sandbox_backend=LocalTestSandboxBackend(),
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
        sandbox_backend=LocalTestSandboxBackend(),
    )
    paths = runtime.save_checkpoints({"loss": 1.0})
    assert "Policy" in paths

    loaded = JarvisRuntime(
        action_generator=ScriptedActionGenerator(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10),
        data_dir=tmp_path / "runtime_data",
        sandbox_backend=LocalTestSandboxBackend(),
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
        sandbox_backend=LocalTestSandboxBackend(),
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
        sandbox_backend=LocalTestSandboxBackend(),
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


# --------------------------------------------------------------------------
# A clock running out is not a failing test
# --------------------------------------------------------------------------
#
# `_classify_process_failure` already separated infrastructure from a real test
# failure, and it did so by reading the output for "timed out" -- which a killed
# process never gets to print, because it is killed rather than returning. The
# one case the classifier most needed to catch was the one case it could not
# see, and the exception propagated instead.
#
# `a9968e5` fixed the same confusion in the acceptance cascade, where a
# 120-second timeout was read as "capability broken" and retired a verified
# capability. Observed here as a v04 acquisition test that passed alone and
# failed under a loaded full suite, on a five-second budget that does not cover
# starting CPython and importing pytest.


class _TimingOutBackend:
    """A backend whose every process is killed for taking too long."""

    def __init__(self):
        from environments.coding.sandbox_backend import SandboxPolicy

        self.policy = SandboxPolicy()

    def run(self, command, cwd, timeout_seconds, env, verifier_workspace=None):
        import subprocess

        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout_seconds, output=b"partial")


def test_a_process_killed_for_time_is_reported_as_infrastructure(tmp_path):
    from environments.coding.environment import CodingEnvironment
    from environments.coding.task import CodingTask

    task = CodingTask(description="do a thing", workspace=tmp_path)
    environment = CodingEnvironment(task=task, backend=_TimingOutBackend())

    result = environment._run_process(["python", "-c", "pass"], "tests")

    assert not result.ok
    assert result.data.get("failure_kind") == "infrastructure"
    assert result.data.get("timed_out") is True


def test_the_reason_says_it_timed_out_rather_than_failed(tmp_path):
    """What the next step reads decides what it repairs. "The tests failed"
    sends it into the code; "it timed out" does not."""

    from environments.coding.environment import CodingEnvironment
    from environments.coding.task import CodingTask

    task = CodingTask(description="do a thing", workspace=tmp_path)
    environment = CodingEnvironment(task=task, backend=_TimingOutBackend())

    result = environment._run_process(["python", "-c", "pass"], "tests")

    assert "timed out, it did not fail" in result.stderr


def test_a_timeout_does_not_escape_as_an_exception(tmp_path):
    """It used to, and nothing above was catching it."""

    import subprocess

    from environments.coding.environment import CodingEnvironment
    from environments.coding.task import CodingTask

    task = CodingTask(description="do a thing", workspace=tmp_path)
    environment = CodingEnvironment(task=task, backend=_TimingOutBackend())

    try:
        environment._run_process(["python", "-c", "pass"], "tests")
    except subprocess.TimeoutExpired:  # pragma: no cover - the defect
        raise AssertionError("the timeout escaped instead of being reported")


def test_the_local_test_backend_allows_time_to_start_an_interpreter():
    """Five seconds is a resource bound written for the container backend, and
    `run` takes the smaller of the two -- so a caller asking for twenty got
    five, and starting CPython and importing pytest does not fit in it."""

    from environments.coding.sandbox_backend import LocalTestSandboxBackend

    assert LocalTestSandboxBackend().policy.timeout_seconds >= 30


def test_an_explicit_policy_is_still_obeyed():
    """Only the default is loosened. A caller that brought a policy meant it."""

    from environments.coding.sandbox_backend import LocalTestSandboxBackend, SandboxPolicy

    backend = LocalTestSandboxBackend(policy=SandboxPolicy(timeout_seconds=2.0))

    assert backend.policy.timeout_seconds == 2.0
    assert backend.policy.network_disabled
