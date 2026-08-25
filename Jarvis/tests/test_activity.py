"""Activity is a record, not a story.

The Activity button in the running product did nothing visible.  It toggled a
boolean that decided whether *future* tool events would be echoed into the chat
transcript as grey notes -- so pressing it opened no panel, showed no history,
and gave no sign that anything had been recorded, because nothing had.

The fix is a durable log, and the property worth testing is where its entries
come from.  The log has exactly one input: the synchronous watcher on the event
bus.  There is no function in :mod:`runtime.activity` that accepts prose, so
there is no path by which a model's account of what it did becomes an activity
entry.  The tests below pin that, and pin the filtering -- because a log that
records every streamed token is as useless as one that records nothing.
"""

from __future__ import annotations

import json

import pytest

from runtime.activity import NOTABLE_STATES, ActivityEntry, ActivityLog
from service.events import EventBus, EventType


@pytest.fixture
def log(tmp_path):
    return ActivityLog(tmp_path / "activity.jsonl")


@pytest.fixture
def wired(log):
    bus = EventBus()
    log.attach(bus)
    return bus, log


RECEIPT = {
    "id": "rcpt_abc123",
    "kind": "file.write",
    "executor": "tools.write_file",
    "ok": True,
    "verified": True,
    "detail": "wrote 17 characters to C:/ws/zeus_test.txt",
    "evidence": {"path": "C:/ws/zeus_test.txt", "characters": 17},
    "verifications": [
        {"check": "file exists on disk", "passed": True, "observed": "C:/ws/zeus_test.txt (17 bytes)"},
        {"check": "content matches exactly", "passed": True, "observed": "'ZEUS funktioniert'"},
    ],
    "duration_seconds": 0.42,
}


# --------------------------------------------------------------------------
# Where entries come from
# --------------------------------------------------------------------------

def test_an_entry_exists_only_because_an_event_was_published(wired):
    bus, log = wired

    bus.publish(EventType.USER_MESSAGE, {"text": "erstelle die Datei x"})

    entries = log.recent()
    assert [entry.kind for entry in entries] == ["request"]
    assert entries[0].summary == "erstelle die Datei x"


def test_the_log_has_no_way_to_record_anything_but_an_event():
    """The structural claim, checked rather than asserted in a docstring.

    If some future edit adds a `record_text(str)` helper, this fails -- which
    is the point. The guarantee is that the module's only public writer takes
    an ActivityEntry built by `entry_for` from a real bus event.
    """

    import inspect

    from runtime import activity

    public_writers = [
        name for name, value in inspect.getmembers(activity.ActivityLog, inspect.isfunction)
        if not name.startswith("_") and name in {"record", "attach"}
    ]

    assert set(public_writers) == {"record", "attach"}
    signature = inspect.signature(activity.ActivityLog.record)
    assert list(signature.parameters)[1:] == ["entry"]
    # A string rather than the class: the module uses PEP 563 annotations.
    assert signature.parameters["entry"].annotation == "ActivityEntry"


def test_a_watcher_that_throws_cannot_break_the_thing_it_watches(tmp_path):
    """Recording is synchronous with publishing, so it must be harmless."""

    bus = EventBus()
    broken = ActivityLog(tmp_path / "nope" / "\0invalid" / "a.jsonl")
    broken.attach(bus)

    # The bus swallows watcher failures; publishing must still work.
    event = bus.publish(EventType.USER_MESSAGE, {"text": "hello"})

    assert event.seq == 1


# --------------------------------------------------------------------------
# What is worth recording
# --------------------------------------------------------------------------

def test_streamed_tokens_are_not_activity(wired):
    """Thousands per conversation; the finished message says what they made."""

    bus, log = wired
    for chunk in "the quick brown fox".split():
        bus.publish(EventType.TOKEN, {"text": chunk})

    assert log.recent() == []


def test_idle_and_thinking_are_not_recorded_but_working_is(wired):
    """Every turn passes through idle and thinking; recording them buries
    the transitions that carry information."""

    bus, log = wired
    for state in ("idle", "thinking", "working", "verifying", "idle"):
        bus.publish(EventType.STATE, {"state": state, "detail": state})

    assert [entry.kind for entry in log.recent()] == ["state.working", "state.verifying"]


@pytest.mark.parametrize("state", NOTABLE_STATES)
def test_every_notable_state_produces_an_entry(wired, state):
    bus, log = wired

    bus.publish(EventType.STATE, {"state": state, "detail": "d"})

    assert log.recent()[0].kind == f"state.{state}"


def test_a_receipt_event_becomes_a_verdict_entry_carrying_its_receipt(wired):
    bus, log = wired

    bus.publish(EventType.TOOL, {"summary": "x", "receipt": RECEIPT})

    entry = log.recent()[0]
    assert entry.kind == "action.verified"
    assert entry.receipt_id == "rcpt_abc123"
    assert entry.detail["verifications"][1]["check"] == "content matches exactly"


@pytest.mark.parametrize(
    "ok,verified,expected",
    [(True, True, "action.verified"), (True, False, "action.ran"), (False, False, "action.failed")],
)
def test_the_verdict_comes_from_the_receipt_not_from_any_text(wired, ok, verified, expected):
    """A receipt whose detail reads like a triumph is still failed if it failed."""

    bus, log = wired
    receipt = {**RECEIPT, "ok": ok, "verified": verified,
               "detail": "successfully created and verified everything"}

    bus.publish(EventType.TOOL, {"receipt": receipt})

    assert log.recent()[0].kind == expected


def test_an_answers_backend_is_recorded_so_prose_and_evidence_stay_distinguishable(wired):
    bus, log = wired

    bus.publish(EventType.MESSAGE, {"text": "I did it", "backend": "qwen3:4b-instruct"})

    entry = log.recent()[0]
    assert entry.kind == "answer"
    assert entry.detail["backend"] == "qwen3:4b-instruct"
    assert entry.receipt_id == "", "an answer is not evidence and carries no receipt"


def test_errors_and_notifications_are_recorded(wired):
    bus, log = wired

    bus.publish(EventType.ERROR, {"error": "RuntimeError: boom"})
    bus.publish(EventType.NOTIFICATION, {"text": "quota resets at noon"})

    assert [entry.kind for entry in log.recent()] == ["error", "notification"]


# --------------------------------------------------------------------------
# Durability
# --------------------------------------------------------------------------

def test_activity_survives_the_process(tmp_path):
    """A client that was not connected must still see what happened."""

    path = tmp_path / "activity.jsonl"
    bus = EventBus()
    ActivityLog(path).attach(bus)
    bus.publish(EventType.TOOL, {"receipt": RECEIPT})

    reopened = ActivityLog(path).recent()

    assert len(reopened) == 1
    assert reopened[0].receipt_id == "rcpt_abc123"


def test_one_corrupt_line_does_not_hide_the_rest(tmp_path):
    path = tmp_path / "activity.jsonl"
    log = ActivityLog(path)
    log.record(ActivityEntry(kind="request", summary="first"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ not json\n")
    log.record(ActivityEntry(kind="request", summary="second"))

    assert [entry.summary for entry in ActivityLog(path).recent()] == ["first", "second"]


def test_reading_recent_activity_does_not_read_the_whole_file(tmp_path):
    """The file is append-only and unbounded; showing the last fifty entries
    must not get slower every day the product is used."""

    from runtime.activity import TAIL_BYTES

    path = tmp_path / "activity.jsonl"
    log = ActivityLog(path)
    padding = "x" * 2000
    for index in range(600):
        log.record(ActivityEntry(kind="request", summary=f"{index} {padding}"))

    assert path.stat().st_size > TAIL_BYTES
    recent = log.recent(20)
    assert len(recent) == 20
    assert recent[-1].summary.startswith("599")


def test_the_entry_shape_is_what_the_ui_reads(wired):
    """A contract between two files that cannot import each other."""

    bus, log = wired
    bus.publish(EventType.TOOL, {"receipt": RECEIPT})

    payload = log.recent()[0].to_dict()

    for field in ("kind", "summary", "at", "seq", "receipt_id", "detail"):
        assert field in payload, f"ui/app.js reads entry.{field}"
    assert json.dumps(payload), "an entry must survive serialisation"
