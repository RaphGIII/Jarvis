"""The owner core: protected in the edit engine, the promoter and on disk;
changed only through an approved transaction."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from owner.core import DEFAULTS, OwnerCore
from owner.protected import PROTECTED_PATHS, is_protected, protected_violations


def test_protected_paths_cover_the_recovery_and_policy_machinery() -> None:
    for expected in ("zeus_supervisor", "owner", "config/owner", "deployment/promotion.py", "runtime/cost_policy.py"):
        assert expected in PROTECTED_PATHS
    assert is_protected("zeus_supervisor/supervisor.py")
    assert is_protected("config\\owner\\policy.json")
    assert not is_protected("ui/app.js")
    assert not is_protected("deployment/other.py")
    assert protected_violations(["ui/app.js", "owner/core.py"]) == ["owner/core.py"]


def test_defaults_are_read_without_files(tmp_path: Path) -> None:
    owner = OwnerCore(tmp_path / "cfg", tmp_path / "state")
    assert owner.read("identity")["assistant_name"] == "Zeus"
    assert "calm" in owner.personality_prompt()
    assert owner.policy("self_development", "enabled") is True
    assert owner.read("spending")["paid_api"] is False


def test_transaction_shows_diff_then_writes_read_only_and_audits(tmp_path: Path) -> None:
    owner = OwnerCore(tmp_path / "cfg", tmp_path / "state")
    tx = owner.propose({"personality": {"humour": "none"}}, reason="test", origin="ui")
    assert tx.diff() == [{"document": "personality", "key": "humour",
                          "from": DEFAULTS["personality"]["humour"], "to": "none"}]
    assert owner.read("personality")["humour"] != "none", "proposing writes nothing"

    record = owner.approve(tx.transaction_id)
    assert owner.read("personality")["humour"] == "none"
    path = owner.path("personality")
    assert not os.access(path, os.W_OK) or not (path.stat().st_mode & stat.S_IWRITE)
    with pytest.raises(PermissionError):
        path.write_text("{}", encoding="utf-8")
    assert owner.history()[-1]["audit_id"] == record["audit_id"]

    rolled = owner.rollback(record["audit_id"])
    assert owner.read("personality")["humour"] == DEFAULTS["personality"]["humour"]
    assert rolled["kind"] == "rollback" and owner.history()[-1]["rolled_back"] == record["audit_id"]


def test_approve_requires_a_pending_transaction(tmp_path: Path) -> None:
    owner = OwnerCore(tmp_path / "cfg", tmp_path / "state")
    with pytest.raises(KeyError):
        owner.approve("nope")
    tx = owner.propose({"policy": {"restart": {"allowed": False}}})
    assert owner.reject(tx.transaction_id) is True
    with pytest.raises(KeyError):
        owner.approve(tx.transaction_id)


def test_promoter_refuses_protected_paths(tmp_path: Path) -> None:
    from deployment.promotion import HealthCheck, Promoter

    repo = tmp_path / "live"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "ui").mkdir()
    (repo / "ui" / "app.js").write_text("// live\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-qm", "base"], cwd=repo, check=True)

    candidate = tmp_path / "candidate"
    (candidate / "owner").mkdir(parents=True)
    (candidate / "owner" / "core.py").write_text("# weakened\n", encoding="utf-8")
    (candidate / "ui").mkdir()
    (candidate / "ui" / "app.js").write_text("// candidate\n", encoding="utf-8")

    promoter = Promoter(repo, snapshot_root=tmp_path / "snap")
    record = promoter.promote(
        candidate, changed_files=["owner/core.py", "ui/app.js"],
        health_check=HealthCheck(command=["python", "-c", "print('ok')"], expect_output="ok"),
    )
    assert record.outcome.value == "REJECTED"
    assert "owner-protected" in record.reason
    assert (repo / "ui" / "app.js").read_text(encoding="utf-8") == "// live\n", "nothing moved"
    assert not (repo / "owner").exists()


def test_engineer_merges_owner_protected_paths(tmp_path: Path) -> None:
    from development.repository_engineer import RepositoryEngineer, SelfImprovementGoal

    # A repository that looks like ZEUS gets the owner's list whatever the goal says.
    repo = tmp_path / "Jarvis"
    (repo / "zeus_supervisor").mkdir(parents=True)
    (repo / "zeus_supervisor" / "x.py").write_text("x = 1\n", encoding="utf-8")
    goal = SelfImprovementGoal(objective="anything", protected_paths=["extra.py"])
    engineer = RepositoryEngineer(brain=object(), worktree_root=tmp_path / "wt")

    seen: dict[str, list[str]] = {}

    def fake_hash(source: Path, protected_paths: list[str]) -> dict[str, str]:
        seen["paths"] = list(protected_paths)
        raise RuntimeError("stop here")

    engineer._hash_protected = fake_hash  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        engineer.improve(repo, goal)
    assert "extra.py" in seen["paths"]
    assert "zeus_supervisor" in seen["paths"] and "owner" in seen["paths"]


def test_personality_reaches_the_provider_prompt() -> None:
    import config

    prompt = config.system_prompt()
    assert "Zeus" in prompt
    assert "Character:" in prompt
    assert "never claim an action was performed" in prompt.lower()
