"""The durable Mission Engine and the evidence core."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.evidence import Evidence, EvidenceKind, Verifier, claim, from_receipt, observation, verdict
from runtime.mission_engine import Mission, MissionCancelled, MissionEngine, MissionEngineStore, MissionPaused


@pytest.fixture
def engine(tmp_path: Path):
    events = []
    eng = MissionEngine(MissionEngineStore(tmp_path / "missions"), emit=lambda k, p: events.append((k, p)))
    eng.events = events  # type: ignore[attr-defined]
    return eng


def test_a_mission_is_one_file_that_a_new_process_can_resume_from(engine, tmp_path):
    m = engine.create("Build a study routine", kind="complex", interpretation="compose existing capabilities",
                      constraints=["no paid APIs"], acceptance=["timer starts", "music plays"])
    engine.transition(m, "INVESTIGATE", "looking at capabilities")
    engine.add_hypothesis(m, "music.play and timer.start exist")
    t1 = engine.add_task(m, "start timer")
    t2 = engine.add_task(m, "play music", depends_on=[t1.task_id])
    engine.set_next(m, "start the timer")
    # a new engine, from disk only
    fresh = MissionEngine(MissionEngineStore(tmp_path / "missions"))
    loaded = fresh.store.load(m.mission_id)
    assert loaded.phase == "INVESTIGATE" and loaded.next_action == "start the timer"
    assert loaded.acceptance_criteria == ["timer starts", "music plays"] and loaded.constraints == ["no paid APIs"]
    assert [t["task_id"] for t in fresh.ready_tasks(loaded)] == [t1.task_id]
    assert fresh.blocked_tasks(loaded)[0][1] == [t1.task_id]
    brief = fresh.brief(loaded)
    assert brief["tasks"] == {"done": 0, "total": 2, "next": ["start timer", "play music"]}
    assert len(json.dumps(brief)) < 2000, "the brief is what replaces a transcript; it must stay small"


def test_complete_needs_proof_and_phases_are_checked(engine):
    m = engine.create("do a thing")
    with pytest.raises(ValueError):
        engine.transition(m, "COMPLETE")
    with pytest.raises(ValueError):
        engine.transition(m, "VERIFY")  # UNDERSTAND -> VERIFY is not a transition
    engine.transition(m, "PLAN")
    engine.transition(m, "EXECUTE")
    engine.add_evidence(m, claim("I did it", source="FAST_LOCAL"))
    engine.transition(m, "VERIFY")
    with pytest.raises(ValueError):
        engine.transition(m, "COMPLETE")  # a claim is not proof
    receipt = {"id": "rcpt_1", "kind": "file.write", "executor": "files", "ok": True, "verified": True,
               "detail": "wrote notes.txt", "verifications": [{"check": "read back", "passed": True, "observed": "12 bytes"}]}
    engine.add_evidence(m, from_receipt(receipt))
    engine.transition(m, "COMPLETE", "done")
    assert m.finished and m.outcome == "complete"
    assert any(p.get("phase") == "COMPLETE" for _, p in engine.events)


def test_pause_cancel_and_resume_are_honoured_at_the_next_step(engine):
    m = engine.create("long work")
    engine.transition(m, "PLAN")
    engine.request_pause(m.mission_id)
    with pytest.raises(MissionPaused):
        engine.transition(m, "EXECUTE")
    engine.settle(m, MissionPaused(m.mission_id))
    assert m.phase == "PAUSED" and m.previous_phase == "PLAN" and not m.finished
    assert engine.resume(m.mission_id)["phase"] == "PLAN"
    m = engine.store.load(m.mission_id)
    engine.request_cancel(m.mission_id)
    with pytest.raises(MissionCancelled):
        engine.transition(m, "EXECUTE")
    engine.settle(m, MissionCancelled(m.mission_id))
    assert m.phase == "CANCELLED" and m.finished


def test_a_handler_that_raises_becomes_a_failed_record_not_a_crash(engine):
    def handler(eng, mission):
        eng.transition(mission, "PLAN")
        raise RuntimeError("the tool exploded")

    engine.register("complex", handler)
    m = engine.run(engine.create("fragile"))
    assert m.phase == "FAILED" and "exploded" in m.reason
    assert engine.store.load(m.mission_id).phase == "FAILED"
    assert engine.run(engine.create("no kind", kind="unknown")).phase == "BLOCKED"


def test_failed_approaches_and_interrupted_missions_survive(engine):
    m = engine.create("stubborn")
    engine.fail_approach(m, "regex on the whole file", "anchor not found")
    engine.transition(m, "PLAN")
    interrupted = engine.mark_interrupted()
    assert [x.mission_id for x in interrupted] == [m.mission_id]
    again = engine.store.load(m.mission_id)
    assert again.failed_approaches[0]["approach"] == "regex on the whole file"
    assert again.history[-1]["event"] == "interrupted" and "resume from PLAN" in again.next_action


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------

def test_the_writer_may_not_verify_its_own_effect():
    ev = observation("notes.txt exists", tool="files")
    with pytest.raises(ValueError):
        Verifier("files").confirm(ev, lambda: (True, "seen"))
    fact = Verifier("shell").confirm(ev, lambda: (True, "dir listing shows notes.txt"))
    assert fact.kind is EvidenceKind.VERIFIED_FACT and fact.execution_verified and fact.verifier == "shell"
    refuted = Verifier("shell").confirm(ev, lambda: (False, "not there"))
    assert refuted.kind is EvidenceKind.TOOL_OBSERVATION and not refuted.execution_verified and refuted.data["refuted_by"] == "shell"


def test_execution_verified_is_not_goal_satisfied():
    receipt = {"id": "r", "kind": "file.write", "executor": "files", "ok": True, "verified": True, "detail": "wrote",
               "verifications": [{"check": "read back", "passed": True, "observed": "ok"}]}
    ev = from_receipt(receipt)
    v = verdict([ev])
    assert v.execution_verified and v.goal_satisfied is None, "nobody has said the owner's intent was met"
    v2 = verdict([ev], goal_check=lambda: False)
    assert v2.execution_verified and v2.goal_satisfied is False
    unverified = from_receipt({**receipt, "verified": False, "verifications": []})
    assert unverified.kind is EvidenceKind.CLAIM and not verdict([unverified]).execution_verified


def test_evidence_round_trips_through_json():
    ev = Evidence(EvidenceKind.EXTERNAL_SOURCE, "Ollama 0.30.10 works on the GTX 1070", source="docs.example", url="https://docs.example/x")
    again = Evidence.from_dict(json.loads(json.dumps(ev.to_dict())))
    assert again.kind is EvidenceKind.EXTERNAL_SOURCE and again.url == ev.url and again.evidence_id == ev.evidence_id
