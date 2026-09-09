"""What is promoted is what was verified.

Mission a00d4506ca: Codex changed three files, verification passed at 15:12,
the mission parked at AWAITING_AUTHORIZATION -- and ``_finish`` released the
worktree.  At 17:00 the owner typed the password; the promoter found an
empty directory where the candidate had been, read every missing file as
"the candidate deleted it", removed three live modules, passed the import
health check, and committed a 1362-line deletion.  The supervisor rolled it
back at the next start.

These tests pin the four things that now prevent it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment.promotion import HealthCheck, PromotionAudit, PromotionOutcome, Promoter, fingerprint_files
from owner.security_gate import SecurityGate
from service.isolation import CandidateWorkspace
from service.selfdev import SelfDevMission, SelfDevRunner, SelfDevStore

PASSWORD = "the-owner-types-this-into-the-ui"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=60)


@pytest.fixture
def live(tmp_path: Path) -> Path:
    root = tmp_path / "live"
    root.mkdir()
    (root / "app.py").write_bytes(b"VERSION = 1\n\ndef greet():\n    return 'hello'\n")
    (root / "desktop.py").write_bytes(b"def show():\n    return 'window'\n")
    (root / "README.md").write_bytes(b"live\n")
    _git(root, "init")
    _git(root, "config", "user.email", "jarvis@example.invalid")
    _git(root, "config", "user.name", "Jarvis Test")
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-m", "baseline")
    return root


def _promoter(live: Path, tmp_path: Path) -> Promoter:
    return Promoter(live, audit=PromotionAudit(tmp_path / "promotions.jsonl"), snapshot_root=tmp_path / "snapshots")


def _ok() -> HealthCheck:
    return HealthCheck(command=[sys.executable, "-c", "print('OK')"], expect_output="OK")


# --------------------------------------------------------------------------
# 1. The promoter refuses an empty shell
# --------------------------------------------------------------------------

def test_an_emptied_candidate_is_refused_and_the_live_files_survive(live, tmp_path):
    shell = tmp_path / "candidate_gone"
    shell.mkdir()  # the directory exists; nothing is in it -- exactly the released worktree

    record = _promoter(live, tmp_path).promote(shell, changed_files=["app.py", "desktop.py"], health_check=_ok(),
                                               commit_message="would have deleted two modules")

    assert record.outcome is PromotionOutcome.REJECTED
    assert "empty shell" in record.reason
    assert record.stages[-1]["stage"] == "PREFLIGHT" and record.stages[-1]["ok"] is False
    assert (live / "app.py").exists() and (live / "desktop.py").exists()
    assert _git(live, "status", "--porcelain").stdout.strip() == ""
    assert "would have deleted" not in _git(live, "log", "--oneline").stdout


def test_a_genuine_single_deletion_is_still_a_deletion(live, tmp_path):
    candidate = tmp_path / "deletes_one"
    candidate.mkdir()
    (candidate / "app.py").write_bytes(b"VERSION = 2\n\ndef greet():\n    return 'hello'\n")
    # README.md absent on purpose: one file changed, one deleted, not an empty shell.
    record = _promoter(live, tmp_path).promote(candidate, changed_files=["app.py", "README.md"], health_check=_ok(),
                                               commit_message="delete the readme")
    assert record.outcome is PromotionOutcome.PROMOTED
    assert not (live / "README.md").exists()


# --------------------------------------------------------------------------
# 2. The promoter refuses a candidate that is not what was verified
# --------------------------------------------------------------------------

def test_a_candidate_that_changed_since_verification_is_refused(live, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "app.py").write_bytes(b"VERSION = 2\n")
    verified = fingerprint_files(candidate, ["app.py"])
    (candidate / "app.py").write_bytes(b"VERSION = 3  # edited after verification\n")

    record = _promoter(live, tmp_path).promote(candidate, changed_files=["app.py"], health_check=_ok(), expected=verified)

    assert record.outcome is PromotionOutcome.REJECTED
    assert "not what was verified" in record.reason and "content changed" in record.reason
    assert (live / "app.py").read_bytes().startswith(b"VERSION = 1")


def test_a_file_missing_since_verification_is_refused_not_mirrored(live, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "app.py").write_bytes(b"VERSION = 2\n")
    (candidate / "desktop.py").write_bytes(b"def show():\n    return 'fullscreen'\n")
    verified = fingerprint_files(candidate, ["app.py", "desktop.py"])
    (candidate / "desktop.py").unlink()  # the worktree lost a file, the record still says it was modified

    record = _promoter(live, tmp_path).promote(candidate, changed_files=["app.py", "desktop.py"], health_check=_ok(),
                                               expected=verified)

    assert record.outcome is PromotionOutcome.REJECTED
    assert "desktop.py (missing now)" in record.reason
    assert (live / "desktop.py").exists()


def test_a_matching_fingerprint_promotes(live, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "app.py").write_bytes(b"VERSION = 2\n\ndef greet():\n    return 'hello world'\n")
    verified = fingerprint_files(candidate, ["app.py"])

    record = _promoter(live, tmp_path).promote(candidate, changed_files=["app.py"], health_check=_ok(),
                                               expected=verified, commit_message="ok")

    assert record.outcome is PromotionOutcome.PROMOTED
    assert "fingerprint verified" in record.stages[0]["detail"]


def test_a_file_the_verification_never_saw_is_refused(live, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "app.py").write_bytes(b"VERSION = 2\n")
    (candidate / "desktop.py").write_bytes(b"def show():\n    return 'smuggled'\n")
    verified = fingerprint_files(candidate, ["app.py"])  # desktop.py was not part of the verified set

    record = _promoter(live, tmp_path).promote(candidate, changed_files=["app.py", "desktop.py"], health_check=_ok(),
                                               expected=verified)
    assert record.outcome is PromotionOutcome.REJECTED
    assert "desktop.py (not in the verified set)" in record.reason


# --------------------------------------------------------------------------
# 3. and 4. The runner keeps the candidate while the owner decides, and
#    rebuilds it from evidence if it is gone anyway
# --------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    """A git repository shaped like the product: <root>/Jarvis is what ZEUS runs from."""

    monkeypatch.setenv("TEMP", str(tmp_path / "temp"))  # candidate worktrees land here
    (tmp_path / "temp").mkdir()
    root = tmp_path / "repo"
    jarvis = root / "Jarvis"
    (jarvis / "service").mkdir(parents=True)
    (jarvis / "service" / "desktop.py").write_text("def show():\n    return 'window'\n", encoding="utf-8")
    (jarvis / "service" / "__init__.py").write_text("", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.email", "jarvis@example.invalid")
    _git(root, "config", "user.name", "Jarvis Test")
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-m", "baseline")
    return jarvis


def _runner(tmp_path: Path, repository: Path, gate: SecurityGate) -> SelfDevRunner:
    from capabilities.codex import CodexAvailabilityState, StaticCodexAvailability

    return SelfDevRunner(
        repository=repository,
        store=SelfDevStore(tmp_path / "state" / "selfdev"),
        kernel=SimpleNamespace(state_root=tmp_path / "state"),
        owner=SimpleNamespace(read=lambda _name: {}),
        lifecycle=SimpleNamespace(supervised=False),
        gateway=None,
        availability=StaticCodexAvailability(CodexAvailabilityState.READY, "ready"),
        security=gate,
        emit=lambda *a, **k: None,
        set_state=lambda *a, **k: None,
    )


def _verified_candidate(runner: SelfDevRunner, mission: SelfDevMission) -> CandidateWorkspace:
    """An engineer's change in a real worktree, verified the way the runner verifies."""

    workspace = runner._prepare_workspace(mission)
    target = workspace.root / "service" / "desktop.py"
    target.write_text("def show():\n    return 'fullscreen'\n", encoding="utf-8")
    mission.changed_files = runner._changed_files(mission.worktree)
    mission.acceptance = [{"criterion": "prints OK", "command": [sys.executable, "-c", "print('OK')"], "expect": "OK"}]
    mission.area = "code"
    assert runner._verify(mission), mission.verification
    assert mission.verification["fingerprint"]["service/desktop.py"] is not None
    return workspace


def test_a_candidate_awaiting_the_owner_is_kept_not_released(tmp_path, repo):
    gate = SecurityGate(tmp_path / "state" / "owner" / "auth.json")
    gate.setup(PASSWORD)
    runner = _runner(tmp_path, repo, gate)
    mission = SelfDevMission(request="Zeus, starte immer im Vollbild")
    workspace = _verified_candidate(runner, mission)

    runner._await_authorization(mission, runner._promotion_refusal(mission))
    runner._finish(mission)  # what run()'s finally does

    assert mission.phase == "AWAITING_AUTHORIZATION"
    assert workspace.exists(), "the candidate the password will promote must still be there"
    assert (workspace.root / "service" / "desktop.py").read_text(encoding="utf-8").endswith("'fullscreen'\n")
    assert mission.worktree == str(workspace.root)
    assert not any(e.get("phase") == "RELEASE" for e in mission.events)


def test_an_authorized_promotion_with_an_emptied_worktree_rebuilds_from_evidence(tmp_path, repo):
    gate = SecurityGate(tmp_path / "state" / "owner" / "auth.json")
    gate.setup(PASSWORD)
    runner = _runner(tmp_path, repo, gate)
    mission = SelfDevMission(request="Zeus, starte immer im Vollbild")
    workspace = _verified_candidate(runner, mission)
    runner._await_authorization(mission, runner._promotion_refusal(mission))
    runner._finish(mission)

    # The old failure, reproduced: the worktree is released behind the mission's back.
    report = workspace.release(evidence_root=runner.evidence_root, reason="simulated release")
    mission.evidence_patch = report["evidence"]
    Path(mission.worktree).mkdir(parents=True, exist_ok=True)  # the empty shell that was found at 17:00
    assert fingerprint_files(mission.worktree, mission.changed_files) == {"service/desktop.py": None}
    runner._workspace = None

    mission.authorization = gate.unlock(PASSWORD, "SELFDEV_PROMOTE")["authorization"]
    finished = runner.promote_authorized(mission)

    assert finished.outcome == "promoted", (finished.reason, [e for e in finished.events if e["phase"] in {"RESTORE", "PROMOTE", "FAILED"}])
    assert any(e.get("phase") == "RESTORE" and "rebuilt" in e.get("detail", "") for e in finished.events)
    live = repo / "service" / "desktop.py"
    assert live.read_text(encoding="utf-8").endswith("'fullscreen'\n"), "the change, not a deletion, reached the live tree"
    assert "delete mode" not in _git(repo.parent, "show", "--stat", "HEAD").stdout


def test_an_authorized_promotion_with_no_evidence_fails_cleanly_and_deletes_nothing(tmp_path, repo):
    gate = SecurityGate(tmp_path / "state" / "owner" / "auth.json")
    gate.setup(PASSWORD)
    runner = _runner(tmp_path, repo, gate)
    mission = SelfDevMission(request="Zeus, starte immer im Vollbild")
    workspace = _verified_candidate(runner, mission)
    runner._await_authorization(mission, runner._promotion_refusal(mission))

    workspace.release(evidence_root=None, reason="lost without evidence")
    Path(mission.worktree).mkdir(parents=True, exist_ok=True)
    runner._workspace = None
    mission.evidence_patch = ""

    mission.authorization = gate.unlock(PASSWORD, "SELFDEV_PROMOTE")["authorization"]
    finished = runner.promote_authorized(mission)

    assert finished.outcome == "failed" and "nothing was promoted" in finished.reason
    assert (repo / "service" / "desktop.py").read_text(encoding="utf-8").endswith("'window'\n")
    tracked_changes = [line for line in _git(repo.parent, "status", "--porcelain").stdout.splitlines()
                       if line.strip() and "Jarvis/data/" not in line]  # runtime state under data/ is not a change
    assert tracked_changes == []


def test_the_live_tree_verification_runs_the_targeted_tests(tmp_path, repo, monkeypatch):
    """The import check alone let the deletion through; the tests would not have."""

    gate = SecurityGate(tmp_path / "state" / "owner" / "auth.json")
    gate.setup(PASSWORD)
    runner = _runner(tmp_path, repo, gate)
    mission = SelfDevMission(request="Zeus, starte immer im Vollbild")
    _verified_candidate(runner, mission)
    mission.verification["tests"] = ["tests/test_desktop_lifecycle.py"]
    seen: list[list[str]] = []
    real_run = runner._run

    def spy(command, cwd, timeout=600.0):
        seen.append(list(command))
        if "pytest" in command:
            return True, "1 passed"
        return real_run(command, cwd, timeout)

    monkeypatch.setattr(runner, "_run", spy)
    mission.authorization = gate.unlock(PASSWORD, "SELFDEV_PROMOTE")["authorization"]
    finished = runner.promote_authorized(mission)

    assert finished.outcome == "promoted", finished.reason
    pytest_calls = [c for c in seen if "pytest" in c]
    assert pytest_calls and "tests/test_desktop_lifecycle.py" in pytest_calls[-1]
    verify_stage = next(s for s in finished.promotion["stages"] if s["stage"] == "VERIFY")
    assert "targeted test" in verify_stage["detail"]
