"""Hard SelfDev isolation: experimental code never reaches the live tree.

Every test here builds a real git repository with ZEUS in a subdirectory (the
production layout: ``repo/Jarvis``), runs a mission whose "BUILD_LOCAL" is a
deliberately misbehaving stand-in, and checks the live tree afterwards with
``git status`` -- an observer that is not the code under test.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from service.isolation import CandidateWorkspace, LiveTreeGuard, MissionCancelled
from service.selfdev import SelfDevMission, SelfDevRunner, SelfDevStore


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def layout(tmp_path: Path) -> tuple[Path, Path]:
    """``repo/Jarvis`` inside a git repository whose root is ``repo``."""

    top = tmp_path / "repo"
    zeus = top / "Jarvis"
    for rel, text in {
        "Jarvis/service/core.py": "VALUE = 1\n",
        "Jarvis/service/__init__.py": "",
        "Jarvis/ui/app.js": "// app\n",
        "Jarvis/owner/core.py": "# owner\n",
        "Jarvis/zeus_supervisor/__init__.py": "",
        "Jarvis/tests/test_core_value.py": "from service.core import VALUE\n\ndef test_value():\n    assert VALUE >= 1\n",
        "README.md": "root\n",
    }.items():
        path = top / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(top, "init", "-q")
    _git(top, "config", "user.email", "t@example.com")
    _git(top, "config", "user.name", "t")
    _git(top, "add", ".")
    _git(top, "-c", "commit.gpgsign=false", "commit", "-qm", "base")
    return top, zeus


def _status(top: Path) -> str:
    return _git(top, "status", "--porcelain", "--untracked-files=all")


class FakeOwner:
    def read(self, name: str) -> dict:
        return {"self_development": {"enabled": True, "auto_promote": True, "max_seconds": 60}} if name == "policy" else {}


class FakeLifecycle:
    supervised = False

    def request_restart(self, *a, **k):
        return {"ok": True}

    def revision(self):
        return ""

    def supervisor_status(self):
        return {"deployments": []}


def _runner(zeus: Path, tmp_path: Path, build):
    events: list = []
    runner = SelfDevRunner(
        repository=zeus, store=SelfDevStore(tmp_path / "missions"), kernel=SimpleNamespace(provider=lambda tier: object()),
        owner=FakeOwner(), lifecycle=FakeLifecycle(), gateway=None,
        emit=lambda kind, payload: events.append((kind, payload)), set_state=lambda *a, **k: None,
    )
    runner._build = build.__get__(runner)  # type: ignore[method-assign]
    runner.events = events  # type: ignore[attr-defined]
    runner.health_command = [runner.python, "-c", "import service.core; print('FAKE_OK')"]
    runner.health_marker = "FAKE_OK"
    return runner


# --------------------------------------------------------------------------
# The workspace
# --------------------------------------------------------------------------

def test_workspace_is_a_worktree_outside_the_repository(layout, tmp_path):
    top, zeus = layout
    ws = CandidateWorkspace(repository=zeus, mission_id="m1", base=tmp_path / "cand").create()
    assert ws.path.is_dir() and not ws.path.resolve().is_relative_to(top.resolve())
    assert ws.root == ws.path / "Jarvis"
    assert (ws.root / "service" / "core.py").read_text() == "VALUE = 1\n"
    assert "m1" in json.loads(CandidateWorkspace.registry_path(zeus).read_text())
    assert _status(top) == "" or all(line[3:].startswith("Jarvis/data/") for line in _status(top).splitlines())


def test_workspace_env_points_every_zeus_root_into_the_candidate(layout, tmp_path, monkeypatch):
    top, zeus = layout
    monkeypatch.setenv("JARVIS_STATE_ROOT", str(zeus / "data" / "jarvis"))
    monkeypatch.setenv("JARVIS_CONFIG_ROOT", str(zeus / "config"))
    monkeypatch.setenv("ZEUS_SUPERVISED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-for-candidates")
    ws = CandidateWorkspace(repository=zeus, mission_id="m2", base=tmp_path / "cand").create()
    env = ws.env()
    assert Path(env["JARVIS_STATE_ROOT"]).is_relative_to(ws.root)
    assert Path(env["JARVIS_CONFIG_ROOT"]).is_relative_to(ws.root)
    assert env["PYTHONPATH"] == str(ws.root)
    assert "ZEUS_SUPERVISED" not in env and "ANTHROPIC_API_KEY" not in env
    assert env["ZEUS_CANDIDATE"] == "1"


def test_changed_files_are_relative_to_zeus_not_the_git_root(layout, tmp_path):
    top, zeus = layout
    ws = CandidateWorkspace(repository=zeus, mission_id="m3", base=tmp_path / "cand").create()
    (ws.root / "service" / "core.py").write_text("VALUE = 2\n")
    (ws.root / "service" / "new.py").write_text("x = 1\n")
    (ws.root / "data" / "jarvis").mkdir(parents=True)
    (ws.root / "data" / "jarvis" / "state.json").write_text("{}")
    assert ws.changed_files() == ["service/core.py", "service/new.py"]


def test_release_keeps_the_diff_and_removes_the_worktree(layout, tmp_path):
    top, zeus = layout
    ws = CandidateWorkspace(repository=zeus, mission_id="m4", base=tmp_path / "cand").create()
    (ws.root / "service" / "core.py").write_text("VALUE = 3\n")
    report = ws.release(evidence_root=tmp_path / "evidence", reason="test")
    assert report["removed"] and not ws.path.exists()
    patch = Path(report["evidence"]).read_text()
    assert "VALUE = 3" in patch and "m4" in patch
    assert "m4" not in CandidateWorkspace._registry(zeus)
    assert "candidate_m4" not in _git(top, "worktree", "list")


def test_reap_removes_candidates_of_missions_not_kept(layout, tmp_path):
    top, zeus = layout
    base = tmp_path / "cand"
    CandidateWorkspace(repository=zeus, mission_id="old", base=base).create()
    keep = CandidateWorkspace(repository=zeus, mission_id="live", base=base).create()
    removed = CandidateWorkspace.reap(zeus, keep=["live"], base=base)
    assert removed == ["old"]
    assert keep.path.exists() and not (base / "candidate_old").exists()


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------

def test_guard_restores_only_files_that_match_the_candidate(layout, tmp_path):
    top, zeus = layout
    guard = LiveTreeGuard(zeus)
    # The owner's own uncommitted edit, present before the mission.
    (zeus / "ui" / "app.js").write_text("// owner was here\n")
    before = guard.fingerprint()
    ws = CandidateWorkspace(repository=zeus, mission_id="m5", base=tmp_path / "cand").create()
    (ws.root / "service" / "core.py").write_text("VALUE = 9\n")
    (ws.root / "service" / "leak.py").write_text("leak = True\n")
    # Contamination: candidate bytes land in the live tree.
    (zeus / "service" / "core.py").write_text("VALUE = 9\n")
    (zeus / "service" / "leak.py").write_text("leak = True\n")
    # An unrelated owner edit made during the mission -- not the candidate's bytes.
    (zeus / "README_owner.md").write_text("mine\n")
    report = guard.check(before, ws, phase="TEST")
    assert sorted(report["contamination"]) == ["Jarvis/service/core.py", "Jarvis/service/leak.py"]
    assert sorted(report["restored"]) == ["Jarvis/service/core.py", "Jarvis/service/leak.py"]
    assert (zeus / "service" / "core.py").read_text() == "VALUE = 1\n"
    assert not (zeus / "service" / "leak.py").exists()
    assert (zeus / "README_owner.md").read_text() == "mine\n"  # never touched
    assert (zeus / "ui" / "app.js").read_text() == "// owner was here\n"  # never touched


# --------------------------------------------------------------------------
# Missions, end to end, against git status
# --------------------------------------------------------------------------

def test_a_failing_candidate_leaves_the_live_tree_byte_identical(layout, tmp_path):
    top, zeus = layout
    head = _git(top, "rev-parse", "HEAD")
    assert _status(top) == ""

    def build(self, mission, max_seconds):
        ws = CandidateWorkspace(repository=self.repository, mission_id=mission.mission_id, base=tmp_path / "cand").create()
        self._workspace = ws
        mission.worktree = str(ws.root)
        (ws.root / "service" / "core.py").write_text("VALUE = 'broken'\nraise SystemExit(3)\n")
        mission.changed_files = self._changed_files(mission.worktree)
        return SimpleNamespace(worktree=str(ws.root), status="candidate", error="", cycles=1)

    runner = _runner(zeus, tmp_path, build)
    mission = runner.run(SelfDevMission(request="break yourself"))
    assert mission.phase == "FAILED" and mission.outcome == "failed"
    assert "no verified candidate" in mission.reason
    # The live tree: same commit, nothing dirty but runtime state.
    assert _git(top, "rev-parse", "HEAD") == head
    dirt = [l for l in _status(top).splitlines() if not l[3:].startswith("Jarvis/data/")]
    assert dirt == []
    assert (zeus / "service" / "core.py").read_text() == "VALUE = 1\n"
    # The candidate is gone, its diff is kept, the guard reported clean phases.
    assert mission.worktree == "" and mission.evidence_patch and Path(mission.evidence_patch).is_file()
    assert "broken" in Path(mission.evidence_patch).read_text()
    assert all(r["clean"] for r in mission.isolation) and {r["phase"] for r in mission.isolation} >= {"BUILD", "VERIFY"}
    assert f"candidate_{mission.mission_id}" not in _git(top, "worktree", "list")


def test_a_build_that_writes_into_the_live_tree_is_caught_and_undone(layout, tmp_path):
    top, zeus = layout

    def build(self, mission, max_seconds):
        ws = CandidateWorkspace(repository=self.repository, mission_id=mission.mission_id, base=tmp_path / "cand").create()
        self._workspace = ws
        mission.worktree = str(ws.root)
        (ws.root / "service" / "core.py").write_text("VALUE = 2\n")
        # The breach: the same bytes written straight into production.
        (self.repository / "service" / "core.py").write_text("VALUE = 2\n")
        (self.repository / "service" / "smuggled.py").write_text("VALUE = 2\n")
        (ws.root / "service" / "smuggled.py").write_text("VALUE = 2\n")
        mission.changed_files = self._changed_files(mission.worktree)
        return SimpleNamespace(worktree=str(ws.root), status="candidate", error="", cycles=1)

    runner = _runner(zeus, tmp_path, build)
    mission = runner.run(SelfDevMission(request="leak"))
    assert mission.phase == "FAILED"
    assert "isolation breach in BUILD" in mission.reason
    assert _status(top) == "" or all(l[3:].startswith("Jarvis/data/") for l in _status(top).splitlines())
    assert (zeus / "service" / "core.py").read_text() == "VALUE = 1\n"
    assert not (zeus / "service" / "smuggled.py").exists()
    breach = [r for r in mission.isolation if r["contamination"]][0]
    assert sorted(breach["restored"]) == ["Jarvis/service/core.py", "Jarvis/service/smuggled.py"]
    errors = [p for k, p in runner.events if getattr(k, "value", k) == "error"]
    assert errors and "wrote into the live tree" in errors[0]["error"]


def test_a_crashing_build_still_releases_the_candidate(layout, tmp_path):
    top, zeus = layout

    def build(self, mission, max_seconds):
        ws = CandidateWorkspace(repository=self.repository, mission_id=mission.mission_id, base=tmp_path / "cand").create()
        self._workspace = ws
        mission.worktree = str(ws.root)
        (ws.root / "service" / "core.py").write_text("VALUE = 5\n")
        raise RuntimeError("child process crashed")

    runner = _runner(zeus, tmp_path, build)
    mission = runner.run(SelfDevMission(request="crash"))
    assert mission.phase == "FAILED" and "child process crashed" in mission.reason
    assert not (tmp_path / "cand" / f"candidate_{mission.mission_id}").exists()
    assert "VALUE = 5" in Path(mission.evidence_patch).read_text()
    assert all(l[3:].startswith("Jarvis/data/") for l in _status(top).splitlines())


def test_cancel_stops_the_mission_at_the_next_phase_and_releases_it(layout, tmp_path):
    top, zeus = layout

    def build(self, mission, max_seconds):
        ws = CandidateWorkspace(repository=self.repository, mission_id=mission.mission_id, base=tmp_path / "cand").create()
        self._workspace = ws
        mission.worktree = str(ws.root)
        (ws.root / "service" / "core.py").write_text("VALUE = 7\n")
        mission.changed_files = self._changed_files(mission.worktree)
        # The owner cancels while the build is running.
        latest = self.store.load(mission.mission_id)
        latest.cancel_requested = True
        self.store.save(latest)
        return SimpleNamespace(worktree=str(ws.root), status="candidate", error="", cycles=1)

    runner = _runner(zeus, tmp_path, build)
    mission = runner.run(SelfDevMission(request="cancel me"))
    assert mission.phase == "CANCELLED" and mission.outcome == "cancelled" and mission.finished
    assert not (tmp_path / "cand" / f"candidate_{mission.mission_id}").exists()
    assert all(l[3:].startswith("Jarvis/data/") for l in _status(top).splitlines())
    assert (zeus / "service" / "core.py").read_text() == "VALUE = 1\n"


def test_the_candidate_health_check_cannot_reach_the_live_state_root(layout, tmp_path, monkeypatch):
    """The environment leak: a candidate honouring JARVIS_STATE_ROOT would write into production."""

    top, zeus = layout
    live_state = zeus / "data" / "jarvis"
    monkeypatch.setenv("JARVIS_STATE_ROOT", str(live_state))

    def build(self, mission, max_seconds):
        ws = CandidateWorkspace(repository=self.repository, mission_id=mission.mission_id, base=tmp_path / "cand").create()
        self._workspace = ws
        mission.worktree = str(ws.root)
        (ws.root / "service" / "core.py").write_text("VALUE = 2\n")
        mission.changed_files = self._changed_files(mission.worktree)
        return SimpleNamespace(worktree=str(ws.root), status="candidate", error="", cycles=1)

    runner = _runner(zeus, tmp_path, build)
    # The health command writes a marker wherever JARVIS_STATE_ROOT points.
    runner.health_command = [runner.python, "-c",
                             "import os, pathlib; r = pathlib.Path(os.environ['JARVIS_STATE_ROOT']); r.mkdir(parents=True, exist_ok=True); "
                             "(r / 'marker').write_text('x'); print('FAKE_OK')"]
    runner.lifecycle.supervised = False
    mission = runner.run(SelfDevMission(request="env"))
    assert not (live_state / "marker").exists(), "the candidate wrote into the live state root"
    assert mission.verification.get("ok") is True, mission.verification
