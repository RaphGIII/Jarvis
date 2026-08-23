from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from development.repository_engineer import (
    RepositoryEngineer,
    SelfImprovementGoal,
)


class FakeBrain:
    pass


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "calculator.py").write_text(
        "def subtract(a, b):\\n"
        "    return a - b\\n",
        encoding="utf-8",
    )

    (repo / "test_calculator.py").write_text(
        "from calculator import subtract\\n\\n"
        "def test_subtract():\\n"
        "    assert subtract(7, 2) == 5\\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["git", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Jarvis Test"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "jarvis@test.local"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    return repo


def _engineer(tmp_path: Path) -> RepositoryEngineer:
    return RepositoryEngineer(
        brain=FakeBrain(),
        worktree_root=tmp_path / "worktrees",
    )


def _goal() -> SelfImprovementGoal:
    return SelfImprovementGoal(
        objective="Add add(a, b) to calculator.py",
        allowed_paths=["calculator.py"],
        protected_paths=["test_calculator.py"],
    )


def test_protected_test_file_remains_readable(tmp_path):
    repo = _repo(tmp_path)
    engineer = _engineer(tmp_path)
    goal = _goal()

    observation = engineer._run_repository_tool(
        repo,
        {
            "tool": "read_file",
            "path": "test_calculator.py",
        },
        goal,
    )

    assert "error" not in observation
    assert "test_subtract" in observation["content"]


def test_write_scope_gate_rejects_modified_protected_file(
    tmp_path,
):
    repo = _repo(tmp_path)
    engineer = _engineer(tmp_path)
    goal = _goal()

    (repo / "test_calculator.py").write_text(
        "# tampered by candidate\\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="write scope violation",
    ):
        engineer._assert_changed_files_allowed(
            repo,
            goal,
        )


def test_write_scope_gate_detects_untracked_file(tmp_path):
    repo = _repo(tmp_path)
    engineer = _engineer(tmp_path)
    goal = _goal()

    (repo / "backdoor.py").write_text(
        "print('should not exist')\\n",
        encoding="utf-8",
    )

    changed = engineer._changed_files(repo)

    assert "backdoor.py" in changed

    with pytest.raises(
        ValueError,
        match="backdoor.py",
    ):
        engineer._assert_changed_files_allowed(
            repo,
            goal,
        )


def test_write_scope_accepts_only_allowed_change(tmp_path):
    repo = _repo(tmp_path)
    engineer = _engineer(tmp_path)
    goal = _goal()

    (repo / "calculator.py").write_text(
        "def subtract(a, b):\\n"
        "    return a - b\\n\\n"
        "def add(a, b):\\n"
        "    return a + b\\n",
        encoding="utf-8",
    )

    changed = engineer._assert_changed_files_allowed(
        repo,
        goal,
    )

    assert changed == ["calculator.py"]



def test_protected_state_ignores_crlf_only_difference(tmp_path):
    from development.repository_engineer import ProtectionState

    source = tmp_path / "source"
    candidate = tmp_path / "candidate"

    source.mkdir()
    candidate.mkdir()

    relative = "test_calculator.py"

    (source / relative).write_bytes(
        b"def test_add():\n"
        b"    assert 1 + 1 == 2\n"
    )

    # Same logical text, Windows CRLF materialization.
    (candidate / relative).write_bytes(
        b"def test_add():\r\n"
        b"    assert 1 + 1 == 2\r\n"
    )

    engineer = RepositoryEngineer(
        brain=FakeBrain(),
        worktree_root=tmp_path / "worktrees",
    )

    hashes = engineer._hash_protected(
        source,
        [relative],
    )

    assert (
        engineer._protected_state(
            source,
            candidate,
            hashes,
        )
        == ProtectionState.PRISTINE
    )


def test_protected_state_detects_real_content_change(tmp_path):
    from development.repository_engineer import ProtectionState

    source = tmp_path / "source"
    candidate = tmp_path / "candidate"

    source.mkdir()
    candidate.mkdir()

    relative = "test_calculator.py"

    (source / relative).write_text(
        "def test_add():\n"
        "    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )

    (candidate / relative).write_text(
        "def test_add():\n"
        "    assert 1 + 1 == 3\n",
        encoding="utf-8",
    )

    engineer = RepositoryEngineer(
        brain=FakeBrain(),
        worktree_root=tmp_path / "worktrees",
    )

    hashes = engineer._hash_protected(
        source,
        [relative],
    )

    assert (
        engineer._protected_state(
            source,
            candidate,
            hashes,
        )
        == ProtectionState.MODIFIED
    )
