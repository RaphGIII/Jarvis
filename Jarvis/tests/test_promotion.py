from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from deployment.promotion import (
    HealthCheck,
    PromotionAudit,
    PromotionOutcome,
    Promoter,
    Snapshot,
    default_health_check,
)


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=30)


@pytest.fixture
def live(tmp_path):
    """A committed 'installation' with one importable module."""

    root = tmp_path / "live"
    root.mkdir()
    (root / "app.py").write_bytes(b"VERSION = 1\n\ndef greet():\n    return 'hello'\n")
    (root / "README.md").write_bytes(b"live\n")
    _git(root, "init")
    _git(root, "config", "user.email", "jarvis@example.invalid")
    _git(root, "config", "user.name", "Jarvis Test")
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-m", "baseline")
    return root


@pytest.fixture
def candidate(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "app.py").write_bytes(b"VERSION = 2\n\ndef greet():\n    return 'hello world'\n")
    return root


def _promoter(live, tmp_path, **kwargs):
    return Promoter(
        live,
        audit=PromotionAudit(tmp_path / "promotions.jsonl"),
        snapshot_root=tmp_path / "snapshots",
        **kwargs,
    )


def _passing_check():
    return HealthCheck(command=[sys.executable, "-c", "print('OK')"], expect_output="OK")


def _failing_check():
    return HealthCheck(command=[sys.executable, "-c", "raise SystemExit(1)"])


# ------------------------------------------------------------------ happy path


def test_a_verified_candidate_is_promoted(live, candidate, tmp_path):
    record = _promoter(live, tmp_path).promote(
        candidate,
        changed_files=["app.py"],
        health_check=_passing_check(),
        commit_message="promote: greet returns hello world",
    )

    assert record.outcome is PromotionOutcome.PROMOTED
    assert record.success
    assert (live / "app.py").read_text(encoding="utf-8").startswith("VERSION = 2")
    assert record.promoted_revision and record.promoted_revision != record.known_good_revision


def test_promotion_commits_only_the_declared_files(live, candidate, tmp_path):
    (candidate / "sneaky.py").write_bytes(b"SNEAKY = True\n")
    _promoter(live, tmp_path).promote(
        candidate, changed_files=["app.py"], health_check=_passing_check(), commit_message="promote"
    )
    assert not (live / "sneaky.py").exists(), "a file the reviewer never saw must not be promoted"


def test_promotion_is_recorded_in_the_audit_log(live, candidate, tmp_path):
    audit = PromotionAudit(tmp_path / "promotions.jsonl")
    Promoter(live, audit=audit, snapshot_root=tmp_path / "snapshots").promote(
        candidate, changed_files=["app.py"], health_check=_passing_check(), commit_message="promote"
    )
    history = audit.history()
    assert len(history) == 1
    assert history[0]["outcome"] == "PROMOTED"
    assert [stage["stage"] for stage in history[0]["stages"]][:3] == ["PREFLIGHT", "SNAPSHOT", "APPLY"]


# ------------------------------------------------------------------ refusals


def test_a_dirty_tree_is_refused_before_anything_changes(live, candidate, tmp_path):
    """A rollback would destroy uncommitted work, so promotion must not start."""

    (live / "app.py").write_bytes(b"VERSION = 1\n# work in progress\n")
    before = (live / "app.py").read_bytes()

    record = _promoter(live, tmp_path).promote(
        candidate, changed_files=["app.py"], health_check=_passing_check()
    )

    assert record.outcome is PromotionOutcome.REJECTED
    assert "dirty" in record.reason
    assert (live / "app.py").read_bytes() == before


def test_a_non_repository_target_is_refused(tmp_path, candidate):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "app.py").write_bytes(b"x\n")
    record = _promoter(plain, tmp_path).promote(
        candidate, changed_files=["app.py"], health_check=_passing_check()
    )
    assert record.outcome is PromotionOutcome.REJECTED
    assert "git repository" in record.reason


def test_an_empty_candidate_is_refused(live, candidate, tmp_path):
    record = _promoter(live, tmp_path).promote(candidate, changed_files=[], health_check=_passing_check())
    assert record.outcome is PromotionOutcome.REJECTED


@pytest.mark.parametrize("escape", ["../outside.py", "/etc/passwd", "a/../../outside.py"])
def test_paths_that_escape_the_repository_are_refused(live, candidate, tmp_path, escape):
    record = _promoter(live, tmp_path).promote(
        candidate, changed_files=[escape], health_check=_passing_check()
    )
    assert record.outcome in {PromotionOutcome.REJECTED, PromotionOutcome.ROLLED_BACK}
    assert (live / "app.py").read_text(encoding="utf-8").startswith("VERSION = 1")


# ------------------------------------------------------------------ rollback


def test_a_failing_health_check_rolls_back_automatically(live, candidate, tmp_path):
    """Mission requirement K: the installation defends itself."""

    original = (live / "app.py").read_bytes()
    known_good = _promoter(live, tmp_path).current_revision()

    record = _promoter(live, tmp_path).promote(
        candidate, changed_files=["app.py"], health_check=_failing_check(), commit_message="promote"
    )

    assert record.outcome is PromotionOutcome.ROLLED_BACK
    assert not record.success
    assert (live / "app.py").read_bytes() == original, "the file must be byte-identical to before"
    assert _promoter(live, tmp_path).current_revision() == known_good


def test_a_failing_verification_rolls_back_before_restarting(live, candidate, tmp_path):
    original = (live / "app.py").read_bytes()
    restarted = []

    promoter = _promoter(live, tmp_path, restart=lambda: (restarted.append(1), (True, "restarted"))[1])
    record = promoter.promote(
        candidate,
        changed_files=["app.py"],
        health_check=_passing_check(),
        verify=lambda repo: (False, "regression suite failed"),
    )

    assert record.outcome is PromotionOutcome.ROLLED_BACK
    assert restarted == [], "there is no point restarting into a change we already rejected"
    assert (live / "app.py").read_bytes() == original


def test_a_failing_restart_rolls_back(live, candidate, tmp_path):
    original = (live / "app.py").read_bytes()
    promoter = _promoter(live, tmp_path, restart=lambda: (False, "service did not come back"))
    record = promoter.promote(candidate, changed_files=["app.py"], health_check=_passing_check())
    assert record.outcome is PromotionOutcome.ROLLED_BACK
    assert (live / "app.py").read_bytes() == original


def test_rollback_restores_a_file_the_candidate_created(live, tmp_path):
    """A new file must be removed again, not merely reverted."""

    candidate = tmp_path / "adds"
    candidate.mkdir()
    (candidate / "app.py").write_bytes((live / "app.py").read_bytes())
    (candidate / "brand_new.py").write_bytes(b"NEW = True\n")

    record = _promoter(live, tmp_path).promote(
        candidate, changed_files=["app.py", "brand_new.py"], health_check=_failing_check()
    )
    assert record.outcome is PromotionOutcome.ROLLED_BACK
    assert not (live / "brand_new.py").exists()


def test_rollback_restores_a_file_the_candidate_deleted(live, tmp_path):
    candidate = tmp_path / "deletes"
    candidate.mkdir()
    (candidate / "app.py").write_bytes((live / "app.py").read_bytes())
    # README.md deliberately absent from the candidate: a deletion.

    record = _promoter(live, tmp_path).promote(
        candidate, changed_files=["app.py", "README.md"], health_check=_failing_check()
    )
    assert record.outcome is PromotionOutcome.ROLLED_BACK
    assert (live / "README.md").read_text(encoding="utf-8") == "live\n"


def test_a_failed_rollback_is_reported_as_needing_a_human(live, candidate, tmp_path, monkeypatch):
    """The one outcome that must never be quietly swallowed."""

    promoter = _promoter(live, tmp_path)
    monkeypatch.setattr(promoter, "_rollback", lambda snapshot, known_good: (False, "disk is read-only"))

    record = promoter.promote(candidate, changed_files=["app.py"], health_check=_failing_check())

    assert record.outcome is PromotionOutcome.ROLLBACK_FAILED
    assert record.needs_human
    assert "ROLLBACK FAILED" in record.reason


def test_manual_rollback_returns_to_a_named_revision(live, candidate, tmp_path):
    promoter = _promoter(live, tmp_path)
    known_good = promoter.current_revision()
    promoter.promote(candidate, changed_files=["app.py"], health_check=_passing_check(), commit_message="promote")
    assert (live / "app.py").read_text(encoding="utf-8").startswith("VERSION = 2")

    record = promoter.rollback_to(known_good)

    assert record.outcome is PromotionOutcome.ROLLED_BACK
    assert (live / "app.py").read_text(encoding="utf-8").startswith("VERSION = 1")
    assert promoter.current_revision() == known_good


# ------------------------------------------------------------------ pieces


def test_health_check_requires_expected_output_not_just_exit_zero():
    """A command that exits zero without doing anything is not health."""

    check = HealthCheck(command=[sys.executable, "-c", "pass"], expect_output="JARVIS_HEALTH_OK")
    ok, detail = check.run(Path.cwd())
    assert not ok
    assert "expected" in detail


def test_health_check_times_out_rather_than_hanging(tmp_path):
    check = HealthCheck(command=[sys.executable, "-c", "import time; time.sleep(30)"], timeout_seconds=2)
    ok, detail = check.run(tmp_path)
    assert not ok and "timed out" in detail


def test_the_default_health_check_exercises_the_real_kernel():
    """Importing the package is not enough; the kernel must assemble."""

    check = default_health_check()
    ok, detail = check.run(Path(__file__).resolve().parent.parent)
    assert ok, detail


def test_snapshot_round_trips_content_and_absence(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "kept.py").write_bytes(b"before\n")

    snapshot = Snapshot.create(root, tmp_path / "snap", ["kept.py", "not_yet.py"])

    (root / "kept.py").write_bytes(b"after\n")
    (root / "not_yet.py").write_bytes(b"created\n")
    snapshot.restore(root)

    assert (root / "kept.py").read_bytes() == b"before\n"
    assert not (root / "not_yet.py").exists()


def test_audit_reports_the_last_known_good_revision(live, candidate, tmp_path):
    audit = PromotionAudit(tmp_path / "promotions.jsonl")
    promoter = Promoter(live, audit=audit, snapshot_root=tmp_path / "snapshots")
    promoter.promote(candidate, changed_files=["app.py"], health_check=_passing_check(), commit_message="promote")
    assert audit.last_known_good() == promoter.current_revision()


def test_snapshots_are_pruned(live, candidate, tmp_path):
    promoter = _promoter(live, tmp_path)
    for index in range(4):
        (candidate / "app.py").write_bytes(f"VERSION = {index + 10}\n".encode())
        promoter.promote(
            candidate, changed_files=["app.py"], health_check=_passing_check(), commit_message=f"promote {index}"
        )
    assert promoter.prune_snapshots(keep=2) == 2
    assert len(list(promoter.snapshot_root.iterdir())) == 2
