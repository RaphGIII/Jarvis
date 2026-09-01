"""The release pipeline without PyInstaller: a fake builder writes a fake exe."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from deployment.release import ReleaseManager, launcher_fingerprint, read_version


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo" / "Jarvis"
    (root / "zeus_supervisor").mkdir(parents=True)
    (root / "zeus_supervisor" / "__init__.py").write_text("__version__ = '1.0.0'\n")
    (root / "zeus_supervisor" / "supervisor.py").write_text("x = 1\n")
    (root / "data").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(root.parent), check=True)
    return root


def fake_builder(exe_body: str = "ok", version: dict | None = None):
    def build(repository: Path, dist: Path) -> Path:
        out = dist / "ZEUS"
        (out / "_internal").mkdir(parents=True)
        exe = out / "ZEUS.exe"
        exe.write_bytes((exe_body * 200_000).encode()[:150_000])
        (out / "VERSION.json").write_text(json.dumps(version or {"product": "ZEUS", "supervisor_version": "1.0.0"}))
        return exe
    return build


def ok_runner(command):
    if command[-1] == "--version":
        return 0, "ZEUS supervisor 1.0.0"
    if command[-1] == "check":
        return 0, "  [ok] models: FAST_LOCAL=x\nall good"
    return 1, "unknown"


def test_fingerprint_changes_when_launcher_sources_change(repo):
    before = launcher_fingerprint(repo)
    (repo / "zeus_supervisor" / "supervisor.py").write_text("x = 2\n")
    assert launcher_fingerprint(repo) != before
    (repo / "zeus_supervisor" / "supervisor.py").write_text("x = 1\n")
    assert launcher_fingerprint(repo) == before


def test_build_verify_promote_keep_previous_and_roll_back(repo, tmp_path):
    rm = ReleaseManager(repo, dist=tmp_path / "dist")
    assert rm.needs_rebuild()[0]
    built = rm.build_candidate(builder=fake_builder("A"))
    assert built.outcome == "built" and Path(built.candidate).is_dir()
    assert read_version(Path(built.candidate) / "ZEUS")["launcher_fingerprint"] == launcher_fingerprint(repo)
    assert not (rm.known_good / "ZEUS.exe").exists(), "a build never touches the known-good release"

    refused = rm.promote(built.candidate)
    assert refused.outcome == "refused" and "not been verified" in refused.reason

    verified = rm.verify_candidate(built.candidate, runner=ok_runner)
    assert verified.outcome == "verified", verified.reason
    assert (Path(built.candidate) / "VERIFIED.json").is_file()

    promoted = rm.promote(built.candidate)
    assert promoted.outcome == "promoted"
    assert (rm.known_good / "ZEUS.exe").read_bytes()[:1] == b"A"
    assert not (rm.previous / "ZEUS.exe").exists()
    assert not rm.needs_rebuild()[0]

    # A second release: the first becomes the previous.
    second = rm.build_candidate(builder=fake_builder("B"))
    rm.verify_candidate(second.candidate, runner=ok_runner)
    assert rm.promote(second.candidate).outcome == "promoted"
    assert (rm.known_good / "ZEUS.exe").read_bytes()[:1] == b"B"
    assert (rm.previous / "ZEUS.exe").read_bytes()[:1] == b"A"

    rolled = rm.rollback("test")
    assert rolled.outcome == "rolled_back"
    assert (rm.known_good / "ZEUS.exe").read_bytes()[:1] == b"A"
    assert any(p.name.startswith("ZEUS.failed.") for p in (tmp_path / "dist").iterdir())
    kinds = [r["kind"] for r in rm.history()]
    assert kinds == ["build", "promote", "verify", "promote", "build", "verify", "promote", "rollback"]


def test_a_candidate_from_other_sources_is_rejected(repo, tmp_path):
    rm = ReleaseManager(repo, dist=tmp_path / "dist")
    built = rm.build_candidate(builder=fake_builder("A"))
    (repo / "zeus_supervisor" / "supervisor.py").write_text("x = 3\n")  # the source moved on
    verified = rm.verify_candidate(built.candidate, runner=ok_runner)
    assert verified.outcome == "rejected" and "fingerprint: FAILED" in verified.reason


def test_a_candidate_whose_exe_fails_preflight_is_rejected(repo, tmp_path):
    rm = ReleaseManager(repo, dist=tmp_path / "dist")
    built = rm.build_candidate(builder=fake_builder("A"))

    def bad_runner(command):
        return (0, "ZEUS supervisor 1.0.0") if command[-1] == "--version" else (1, "  [FAIL] models: missing")

    verified = rm.verify_candidate(built.candidate, runner=bad_runner)
    assert verified.outcome == "rejected" and "preflight: FAILED" in verified.reason
    assert rm.promote(built.candidate).outcome == "refused"


def test_a_failed_build_is_a_record_not_a_crash(repo, tmp_path):
    rm = ReleaseManager(repo, dist=tmp_path / "dist")

    def broken(repository, dist):
        raise RuntimeError("PyInstaller exploded")

    record = rm.build_candidate(builder=broken)
    assert record.outcome == "failed" and "PyInstaller exploded" in record.reason
    assert not (rm.known_good / "ZEUS.exe").exists()


def test_the_watchdog_restores_the_previous_release_when_the_new_one_never_gets_ready(tmp_path, monkeypatch):
    from zeus_supervisor import relaunch

    dist = tmp_path / "dist"
    (dist / "ZEUS").mkdir(parents=True)
    (dist / "ZEUS.previous").mkdir()
    (dist / "ZEUS" / "ZEUS.exe").write_text("new")
    (dist / "ZEUS.previous" / "ZEUS.exe").write_text("old")
    state = tmp_path / "state"
    state.mkdir()
    (state / "token").write_text("t")
    started = []
    monkeypatch.setattr(relaunch, "_start", lambda exe: started.append(str(exe)) or 4242)
    monkeypatch.setattr(relaunch, "_alive", lambda pid: False)
    monkeypatch.setattr(relaunch, "_health", lambda port, token: {})
    monkeypatch.setattr(relaunch.time, "sleep", lambda s: None)
    monkeypatch.setattr(relaunch.subprocess, "run", lambda *a, **k: None)
    code = relaunch.main(["--wait-pid", "1", "--exe", str(dist / "ZEUS" / "ZEUS.exe"), "--previous", str(dist / "ZEUS.previous"),
                          "--state", str(state), "--timeout", "0.01"])
    assert code == 1
    assert (dist / "ZEUS" / "ZEUS.exe").read_text() == "old"
    assert any(p.name.startswith("ZEUS.failed.") for p in dist.iterdir())
    assert started[0].endswith("ZEUS.exe") and len(started) == 2  # the new one, then the restored one
    receipts = [json.loads(l) for l in (state / "releases.jsonl").read_text().splitlines()]
    assert receipts[-1]["outcome"] == "rolled_back"


def test_the_watchdog_reports_healthy_when_the_new_release_gets_ready(tmp_path, monkeypatch):
    from zeus_supervisor import relaunch

    dist = tmp_path / "dist"
    (dist / "ZEUS").mkdir(parents=True)
    (dist / "ZEUS" / "ZEUS.exe").write_text("new")
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(relaunch, "_start", lambda exe: 4242)
    monkeypatch.setattr(relaunch, "_alive", lambda pid: False)
    monkeypatch.setattr(relaunch, "_health", lambda port, token: {"ready": True, "revision": "abc"})
    code = relaunch.main(["--wait-pid", "1", "--exe", str(dist / "ZEUS" / "ZEUS.exe"), "--state", str(state)])
    assert code == 0
    receipts = [json.loads(l) for l in (state / "releases.jsonl").read_text().splitlines()]
    assert receipts[-1]["outcome"] == "healthy"


def test_a_promotion_blocked_by_the_running_exe_is_staged_and_swapped_by_the_watchdog(repo, tmp_path):
    """Windows: the running ZEUS.exe holds dist/ZEUS open, so the release manager cannot
    rename it.  The candidate is staged; the relaunch watchdog swaps it after the old
    supervisor exits; the previous release is kept."""

    import sys

    from zeus_supervisor import relaunch

    rm = ReleaseManager(repo, dist=tmp_path / "dist")
    first = rm.build_candidate(builder=fake_builder("A"))
    rm.verify_candidate(first.candidate, runner=ok_runner)
    assert rm.promote(first.candidate).outcome == "promoted"
    second = rm.build_candidate(builder=fake_builder("B"))
    rm.verify_candidate(second.candidate, runner=ok_runner)

    state = repo / "data" / "jarvis" / "supervisor"
    if sys.platform == "win32":
        holder = (rm.known_good / "ZEUS.exe").open("rb")  # the "running" exe keeps its directory locked
    else:
        holder = None
        original = Path.rename

        def refuse(self, target):
            if self == rm.known_good:
                raise PermissionError("locked")
            return original(self, target)

        Path.rename = refuse  # type: ignore[assignment]
    try:
        staged = rm.promote(second.candidate)
    finally:
        if holder is not None:
            holder.close()
        else:
            Path.rename = original  # type: ignore[assignment]
    assert staged.outcome == "staged", staged.reason
    assert (rm.staged / "ZEUS.exe").read_bytes()[:1] == b"B"
    assert (rm.known_good / "ZEUS.exe").read_bytes()[:1] == b"A", "known-good untouched while locked"
    assert rm.staged_pointer.is_file()

    logs, receipts = [], []
    previous = relaunch._swap_staged(state, logs.append, lambda outcome, **f: receipts.append((outcome, f)))
    assert previous == rm.previous
    assert (rm.known_good / "ZEUS.exe").read_bytes()[:1] == b"B"
    assert (rm.previous / "ZEUS.exe").read_bytes()[:1] == b"A"
    assert not rm.staged.exists() and not rm.staged_pointer.exists()
    assert receipts and receipts[0][0] == "swapped"
    assert json.loads((rm.known_good / "PROMOTED.json").read_text())["swapped_by"] == "relaunch watchdog"
    # idempotent: nothing to swap the second time
    assert relaunch._swap_staged(state, logs.append, lambda *a, **k: None) is None
