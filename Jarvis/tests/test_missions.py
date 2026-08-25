"""An interrupted mission is not a wasted one.

A capability acquisition runs for one to two hours. Anything that stopped it --
a crash, a reboot, an expert quota expiring, ZEUS being restarted to pick up a
fix -- threw the whole thing away, and the next attempt re-derived evidence
that had already been paid for.

That is worse than slow. The performance ledger then records a fresh set of
local failures for work the model was never asked to redo, and the escalation
controller reasons from those counts -- so a restart does not merely cost time,
it manufactures evidence.

The tests below are mostly about refusing to resume. A wrong resume skips work
on the strength of evidence gathered about a different question, which is a
worse failure than starting again.
"""

from __future__ import annotations

import time

import pytest

from runtime.missions import (
    MAX_AGE_SECONDS,
    Attempt,
    MissionCheckpoint,
    MissionStore,
    fingerprint,
)


@pytest.fixture
def store(tmp_path):
    return MissionStore(tmp_path / "missions")


def _checkpoint(goal="build a provider", capability="cap.x", defect="", attempts=1):
    return MissionCheckpoint(
        capability_id=capability,
        goal_fingerprint=fingerprint(goal),
        defect=defect,
        attempts=[Attempt(tier="build_local", evidence=f"attempt {i}") for i in range(attempts)],
    )


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

def test_a_checkpoint_survives_the_process(store):
    store.save(_checkpoint(attempts=2))

    reloaded = store.load("cap.x")

    assert reloaded is not None
    assert reloaded.local_attempts == 2


def test_the_evidence_is_what_is_stored_not_just_a_count(store):
    """Resuming means not repeating an attempt whose answer is known, so the
    answer has to be there."""

    checkpoint = _checkpoint(attempts=0)
    checkpoint.attempts.append(
        Attempt(tier="build_local", hypothesis="the limit is wrong",
                evidence="Spotify replied 400 Invalid limit", succeeded=False)
    )
    store.save(checkpoint)

    reloaded = store.load("cap.x")

    assert reloaded.attempts[0].hypothesis == "the limit is wrong"
    assert "Invalid limit" in reloaded.attempts[0].evidence


def test_a_corrupt_checkpoint_is_ignored_rather_than_fatal(store):
    store.root.mkdir(parents=True, exist_ok=True)
    store.path_for("cap.x").write_text("{ not json", encoding="utf-8")

    assert store.load("cap.x") is None
    assert store.resumable("cap.x", "build a provider") is None


# --------------------------------------------------------------------------
# Refusing to resume, which is most of the job
# --------------------------------------------------------------------------

def test_a_different_question_is_not_resumable(store):
    store.save(_checkpoint(goal="build a music provider"))

    assert store.resumable("cap.x", "build a completely different thing") is None


def test_a_different_defect_is_not_resumable(store):
    """Evidence about one defect says nothing about another, even in the same
    capability."""

    store.save(_checkpoint(defect="search returns 400"))

    assert store.resumable("cap.x", "build a provider", "playback does not switch tracks") is None
    assert store.resumable("cap.x", "build a provider", "search returns 400") is not None


def test_a_stale_checkpoint_is_not_resumable(store):
    """It describes a machine that may no longer exist."""

    checkpoint = _checkpoint()
    checkpoint.updated_at = time.time() - MAX_AGE_SECONDS - 60
    store.save(checkpoint)
    # save() refreshes updated_at, so age it on disk instead.
    stale = store.load("cap.x")
    stale.updated_at = time.time() - MAX_AGE_SECONDS - 60
    store.path_for("cap.x").write_text(
        __import__("json").dumps(stale.to_dict()), encoding="utf-8"
    )

    assert store.resumable("cap.x", "build a provider") is None


def test_a_finished_mission_is_not_resumable(store):
    checkpoint = _checkpoint()
    checkpoint.acquired = True
    store.save(checkpoint)

    assert store.resumable("cap.x", "build a provider") is None


def test_an_unknown_capability_is_not_resumable(store):
    assert store.resumable("never.seen", "build a provider") is None


# --------------------------------------------------------------------------
# The fingerprint identifies the question, not its wording
# --------------------------------------------------------------------------

def test_whitespace_does_not_change_the_question():
    assert fingerprint("build   a\n provider") == fingerprint("build a provider")


def test_a_recalled_lesson_prepended_to_a_repair_does_not_change_the_question():
    """Lessons are prepended to the goal and differ between runs. A fingerprint
    that moved every time would make every checkpoint unresumable, which looks
    exactly like the feature not working."""

    core = "REPAIR an existing, working implementation. This is NOT a rebuild.\n\nTHE DEFECT: x"
    with_lesson = "A previous task of this kind was solved as follows...\n\n" + core

    assert fingerprint(with_lesson) == fingerprint(core)


def test_a_different_defect_does_change_the_question():
    a = "REPAIR an existing, working implementation.\n\nTHE DEFECT: search fails"
    b = "REPAIR an existing, working implementation.\n\nTHE DEFECT: playback fails"

    assert fingerprint(a) != fingerprint(b)


# --------------------------------------------------------------------------
# The mission actually uses it
# --------------------------------------------------------------------------

def test_a_resumed_mission_does_not_repeat_a_finished_attempt(tmp_path):
    from test_acquisition import StubGateway, StubKernel, StubLedger, StubService

    from service.acquisition import AcquisitionMission

    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    store = MissionStore(tmp_path / "missions")
    service = StubService(succeed_on=None)

    class Controller:
        def decide(self, signals):
            return type("D", (), {"escalate": False, "reason": "no", "evidence": []})()

    mission = AcquisitionMission(
        service=service, kernel=StubKernel(tmp_path), gateway=StubGateway(),
        ledger=StubLedger(), controller=Controller(), missions=store,
    )

    first = mission.run("build a provider", capability_id="cap.x")
    calls_after_first = service.calls

    second = mission.run("build a provider", capability_id="cap.x")

    assert calls_after_first == AcquisitionMission.MAX_LOCAL_ATTEMPTS
    assert service.calls == calls_after_first, "a known answer must not be paid for twice"
    assert any(step.stage == "resume" for step in second.steps)


def test_a_successful_mission_leaves_nothing_to_resume(tmp_path):
    from test_acquisition import StubGateway, StubKernel, StubLedger, StubService

    from service.acquisition import AcquisitionMission

    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    store = MissionStore(tmp_path / "missions")
    mission = AcquisitionMission(
        service=StubService(succeed_on=1), kernel=StubKernel(tmp_path),
        gateway=StubGateway(), ledger=StubLedger(), missions=store,
    )

    mission.run("build a provider", capability_id="cap.x")

    assert store.load("cap.x") is None
