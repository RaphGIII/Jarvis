"""WorkItems: the live now-object every long action must have."""

from __future__ import annotations

from service.jobs import JobBoard


def _board(tmp_path, events):
    return JobBoard(tmp_path / "jobs.jsonl", emit=events.append)


def test_lifecycle_emits_and_persists(tmp_path):
    events: list = []
    board = _board(tmp_path, events)
    job = board.create("Bild: Universum", kind="image", cancellable=True)
    board.phase(job.job_id, "Modell lädt", progress=0.15)
    board.phase(job.job_id, "Generiere", progress=0.55)
    board.complete(job.job_id, {"file": "x.png"})
    kinds = [e["event"] for e in events]
    assert kinds == ["created", "phase", "phase", "completed"]
    assert events[-1]["result"]["file"] == "x.png"
    assert events[-1]["timings"]["Generiere"] >= 0
    # a fresh board reads the finished job back from disk
    fresh = _board(tmp_path, [])
    recent = fresh.recent()
    assert recent and recent[0]["job_id"] == job.job_id and recent[0]["state"] == "COMPLETED"


def test_active_and_cancel(tmp_path):
    events: list = []
    board = _board(tmp_path, events)
    job = board.create("Wissen indexieren", kind="index", cancellable=True)
    assert [j["job_id"] for j in board.active()] == [job.job_id]
    assert board.cancel(job.job_id) is True
    assert board.cancelled(job.job_id) is True
    assert board.active() == []
    assert board.recent()[0]["state"] == "CANCELLED"


def test_uncancellable_job_only_sets_the_event(tmp_path):
    board = _board(tmp_path, [])
    job = board.create("SelfDev", kind="selfdev", cancellable=False)
    assert board.cancel(job.job_id) is False
    assert board.cancelled(job.job_id) is True   # the runner may still honour it
    assert board.active()  # state unchanged until the runner reacts


def test_fail_records_the_reason(tmp_path):
    board = _board(tmp_path, [])
    job = board.create("Artikel", kind="web")
    board.fail(job.job_id, "keine Quelle lesbar")
    assert board.recent()[0]["error"].startswith("keine Quelle")


def test_intent_layer_cancels_jobs():
    from service.intents import TopIntent, understand

    u = understand("Stopp die Bilderzeugung.")
    assert u.top is TopIntent.SYSTEM_CONTROL
    assert u.action is not None and u.action.operation == "job.cancel" and u.action.target == "image"
    u2 = understand("Stopp.")
    assert u2.action is not None and u2.action.operation == "system.stop"
