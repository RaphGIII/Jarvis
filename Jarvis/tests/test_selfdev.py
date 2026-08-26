"""Self-development through the product: durable mission, verification from
exit codes, promotion refused for protected paths, verdict from the receipt."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from service.events import EventType
from service.selfdev import SelfDevMission, SelfDevRunner, SelfDevStore, describe, settle_after_restart


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "Jarvis"
    for rel, text in {
        "jarvis/serve.py": "print('serve')\n",
        "service/core.py": "VALUE = 1\n",
        "zeus_supervisor/__init__.py": "",
        "owner/core.py": "# owner\n",
        "ui/app.js": "// app\n",
        "tests/test_core_value.py": "from service.core import VALUE\n\ndef test_value():\n    assert VALUE >= 1\n",
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "base")
    return root


class FakeOwner:
    def __init__(self, policy: dict | None = None) -> None:
        self._policy = policy or {"self_development": {"enabled": True, "auto_promote": True, "max_seconds": 60}}

    def read(self, name: str) -> dict:
        return dict(self._policy) if name == "policy" else {}


class FakeLifecycle:
    def __init__(self, supervised: bool = True, revision: str = "") -> None:
        self.supervised = supervised
        self.requests: list[dict] = []
        self._revision = revision
        self.receipts: list[dict] = []

    def request_restart(self, reason: str, **kw) -> dict:
        self.requests.append({"reason": reason, **kw})
        return {"ok": True}

    def revision(self) -> str:
        return self._revision

    def supervisor_status(self) -> dict:
        return {"deployments": self.receipts}


def _runner(repo: Path, tmp_path: Path, *, build, owner=None, lifecycle=None, gateway=None):
    events: list[tuple] = []
    runner = SelfDevRunner(
        repository=repo, store=SelfDevStore(tmp_path / "missions"), kernel=SimpleNamespace(provider=lambda tier: object()),
        owner=owner or FakeOwner(), lifecycle=lifecycle or FakeLifecycle(), gateway=gateway,
        emit=lambda kind, payload: events.append((kind, payload)), set_state=lambda *a, **k: None,
    )
    runner._build = build.__get__(runner)  # type: ignore[method-assign]
    runner.events = events  # type: ignore[attr-defined]
    runner.health_command = [runner.python, "-c", "import service.core; print('FAKE_OK')"]
    runner.health_marker = "FAKE_OK"
    return runner


def _candidate(runner: SelfDevRunner, mission: SelfDevMission, edits: dict[str, str]) -> None:
    """Stand in for BUILD_LOCAL: make a worktree and apply edits."""

    worktree = runner.repository.parent / f"wt_{mission.mission_id}"
    subprocess.run(["git", "worktree", "add", "--detach", "-q", str(worktree), "HEAD"], cwd=runner.repository, check=True)
    for rel, text in edits.items():
        path = worktree / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    mission.worktree = str(worktree)
    mission.changed_files = runner._changed_files(str(worktree))
    mission.local_attempts += 1


def test_mission_is_durable_and_phases_are_recorded(repo: Path, tmp_path: Path) -> None:
    def build(self, mission, max_seconds):
        _candidate(self, mission, {"service/core.py": "VALUE = 2\n"})

    runner = _runner(repo, tmp_path, build=build)
    mission = SelfDevMission(request="make VALUE bigger in your code", language="en")
    runner.store.save(mission)
    result = runner.run(mission)

    phases = [e["phase"] for e in result.events]
    assert phases[:2] == ["UNDERSTAND", "INVESTIGATE"] and "VERIFY" in phases, result.reason
    assert result.verification["ok"] is True, result.verification
    assert result.verification["tests"] == ["tests/test_core_value.py"], "targeted tests chosen from changed modules"
    assert result.phase == "RESTARTING" and result.promotion["outcome"] == "PROMOTED"
    assert (repo / "service" / "core.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert "self-development" in _git(repo, "log", "-1", "--pretty=%s").lower()
    saved = json.loads(runner.store.path_for(mission.mission_id).read_text(encoding="utf-8"))
    assert saved["phase"] == "RESTARTING" and saved["promotion_id"]
    assert runner.lifecycle.requests[0]["promotion_id"] == saved["promotion_id"]  # type: ignore[attr-defined]
    assert any(k is EventType.PROGRESS for k, _ in runner.events)  # type: ignore[attr-defined]


def test_candidate_touching_protected_paths_never_reaches_promotion(repo: Path, tmp_path: Path) -> None:
    def build(self, mission, max_seconds):
        _candidate(self, mission, {"owner/core.py": "# weakened\n", "service/core.py": "VALUE = 3\n"})

    runner = _runner(repo, tmp_path, build=build)
    result = runner.run(SelfDevMission(request="loosen your own owner policy"))
    assert result.phase == "FAILED"
    assert "owner-protected" in result.reason
    assert (repo / "owner" / "core.py").read_text(encoding="utf-8") == "# owner\n"
    assert (repo / "service" / "core.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_failed_verification_fails_the_mission_without_promotion(repo: Path, tmp_path: Path) -> None:
    def build(self, mission, max_seconds):
        _candidate(self, mission, {"service/core.py": "VALUE = 0\n"})  # breaks the targeted test

    runner = _runner(repo, tmp_path, build=build)
    result = runner.run(SelfDevMission(request="set VALUE in your code to zero"))
    assert result.phase == "FAILED" and result.verification["ok"] is False
    assert (repo / "service" / "core.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_changed_files_ignores_caches_runtime_state_and_directories(repo: Path, tmp_path: Path) -> None:
    runner = _runner(repo, tmp_path, build=lambda self, m, s: None)
    mission = SelfDevMission(request="x")
    _candidate(runner, mission, {"service/core.py": "VALUE = 5\n", "service/__pycache__/core.cpython-314.pyc": "x",
                                 "data/jarvis/state.json": "{}", "newdir/keep.txt": "k"})
    files = runner._changed_files(mission.worktree)
    assert sorted(files) == ["newdir/keep.txt", "service/core.py"], files


def test_resume_reverifies_and_promotes_an_existing_candidate(repo: Path, tmp_path: Path) -> None:
    def build(self, mission, max_seconds):
        _candidate(self, mission, {"service/core.py": "VALUE = 2\n"})

    runner = _runner(repo, tmp_path, build=build)
    mission = SelfDevMission(request="make VALUE bigger in your code")
    runner.store.save(mission)
    # Simulate a crash after BUILD: candidate exists, mission marked failed.
    build(runner, mission, 0)
    mission.phase, mission.outcome, mission.reason = "FAILED", "failed", "PermissionError: pretend"
    runner.store.save(mission)

    resumed = runner.resume(mission)
    assert resumed.phase == "RESTARTING" and resumed.promotion["outcome"] == "PROMOTED", resumed.reason
    assert [e["phase"] for e in resumed.events][:2] == ["RESUME", "VERIFY"]
    assert (repo / "service" / "core.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_disabled_policy_refuses(repo: Path, tmp_path: Path) -> None:
    runner = _runner(repo, tmp_path, build=lambda self, m, s: None,
                     owner=FakeOwner({"self_development": {"enabled": False}}))
    result = runner.run(SelfDevMission(request="anything about yourself"))
    assert result.phase == "FAILED" and "disabled" in result.reason


def test_settlement_takes_the_verdict_from_the_receipt(tmp_path: Path) -> None:
    store = SelfDevStore(tmp_path / "m")
    good = SelfDevMission(request="a", phase="RESTARTING", promotion_id="p1", expected_revision="abc", language="de")
    bad = SelfDevMission(request="b", phase="RESTARTING", promotion_id="p2", expected_revision="def")
    store.save(good)
    store.save(bad)
    lifecycle = FakeLifecycle(revision="abc")
    lifecycle.receipts = [
        {"promotion_id": "p1", "outcome": "healthy", "duration_seconds": 6.0},
        {"promotion_id": "p2", "outcome": "rolled_back", "reason": "process exited with code 1 before READY"},
    ]
    settled = {m.mission_id: m for m in settle_after_restart(store, lifecycle)}
    assert settled[good.mission_id].outcome == "promoted"
    assert settled[bad.mission_id].outcome == "rolled_back"
    assert "abgeschlossen" in describe(settled[good.mission_id], "de")
    assert "rolled back" in describe(settled[bad.mission_id], "en")
    assert store.awaiting_restart() == []
