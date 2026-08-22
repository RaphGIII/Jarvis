import sys

import pytest

from environments.coding.actions import ActionCandidate, ActionType
from environments.coding.environment import CodingEnvironment
from environments.coding.sandbox_backend import DockerSandboxBackend, SandboxPolicy
from environments.coding.task import CodingTask
from training.coding_curriculum import CodingTaskFactory


pytestmark = pytest.mark.skipif(
    not DockerSandboxBackend.is_available(),
    reason="Docker engine is not available.",
)


def _docker_backend() -> DockerSandboxBackend:
    return DockerSandboxBackend(policy=SandboxPolicy(timeout_seconds=20.0))


def test_docker_run_python_executes_workspace_script(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "script.py").write_text("print('docker-script-ok')\n", encoding="utf-8")
    task = CodingTask(
        description="Run a simple Python script.",
        workspace=workspace,
        test_command=[sys.executable, "-m", "unittest", "discover", "-v"],
    )
    env = CodingEnvironment(task, timeout_seconds=12, backend=_docker_backend())

    result = env.step(ActionCandidate(ActionType.RUN_PYTHON, {"path": "script.py"}))

    assert result.action_result.ok
    assert "docker-script-ok" in result.action_result.stdout


def test_docker_unittest_command_normalizes_host_python_path(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
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
    task = CodingTask(
        description="Run unittest inside Docker.",
        workspace=workspace,
        test_command=[sys.executable, "-m", "unittest", "discover", "-v"],
    )
    env = CodingEnvironment(task, timeout_seconds=12, backend=_docker_backend())

    result = env.step(ActionCandidate(ActionType.RUN_TESTS))

    assert result.success
    assert result.action_result.ok


def test_docker_correct_patch_passes_public_and_hidden_tests(tmp_path):
    task = CodingTaskFactory(tmp_path / "tasks").make_hidden_addition_task("docker_correct")
    env = CodingEnvironment(task, timeout_seconds=12, backend=_docker_backend())

    env.step(
        ActionCandidate(
            ActionType.PATCH_FILE,
            {"path": "calculator.py", "old": "return a - b", "new": "return a + b"},
        )
    )
    result = env.step(ActionCandidate(ActionType.RUN_TESTS))
    hidden = env.run_final_hidden_verifier()

    assert result.success
    assert hidden["success"] is True
    assert hidden["runs"] == 1
    assert "hidden_passed" not in result.observation.test_state


def test_docker_public_only_patch_fails_hidden_verifier(tmp_path):
    task = CodingTaskFactory(tmp_path / "tasks").make_hidden_addition_task("docker_hidden_fail")
    env = CodingEnvironment(task, timeout_seconds=12, backend=_docker_backend())

    env.step(
        ActionCandidate(
            ActionType.WRITE_FILE,
            {"path": "calculator.py", "content": "def add(a, b):\n    return 5\n"},
        )
    )
    result = env.step(ActionCandidate(ActionType.RUN_TESTS))
    hidden = env.run_final_hidden_verifier()

    assert result.success
    assert result.action_result.ok
    assert hidden["success"] is False
    assert hidden["runs"] == 1
    assert "hidden_failed" not in result.observation.test_state
    assert "Hidden verifier failed" not in result.action_result.stderr


def test_docker_hidden_verifier_is_not_agent_visible_or_writable(tmp_path):
    task = CodingTaskFactory(tmp_path / "tasks").make_hidden_addition_task("docker_hidden_isolated")
    env = CodingEnvironment(task, timeout_seconds=12, backend=_docker_backend())

    assert all("hidden_verifier.py" not in path for path in env.observe().workspace_tree)

    read_attempt = env.step(
        ActionCandidate(ActionType.READ_FILE, {"path": "../docker_hidden_isolated_hidden/hidden_verifier.py"})
    )
    write_attempt = env.step(
        ActionCandidate(
            ActionType.WRITE_FILE,
            {"path": "../docker_hidden_isolated_hidden/hidden_verifier.py", "content": "print('tamper')"},
        )
    )

    assert not read_attempt.action_result.ok
    assert not write_attempt.action_result.ok
