"""The local-first calendar: parsing, persistence, reminders, .ics round trip."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone as _tz

import pytest

from service.calendar import CalendarStore, parse_event

TZ = _tz(timedelta(hours=2))
NOW = datetime(2026, 9, 2, 19, 0, tzinfo=TZ)  # a Wednesday


# -- parsing ---------------------------------------------------------------

def test_the_owner_sentence_parses_completely():
    out = parse_event("Trag morgen um 14 Uhr Lernen ein.", now=NOW)
    assert out["missing"] == []
    assert out["title"] == "Lernen"
    assert out["start"].startswith("2026-09-03T14:00")
    assert out["duration_minutes"] == 60


def test_duration_and_minutes():
    out = parse_event("Trag morgen um 14 Uhr einen Testtermin für 30 Minuten ein.", now=NOW)
    assert out["missing"] == []
    assert out["start"].startswith("2026-09-03T14:00")
    assert out["end"].startswith("2026-09-03T14:30")
    assert "Testtermin" in out["title"]


def test_explicit_date_and_clock_minutes():
    out = parse_event("Termin am 05.10. um 9:15 Zahnarzt eintragen", now=NOW)
    assert out["start"].startswith("2026-10-05T09:15")
    assert "Zahnarzt" in out["title"]


def test_weekday_resolves_forward():
    out = parse_event("Trag am Montag um 8 Uhr Vorlesung ein", now=NOW)
    assert out["start"].startswith("2026-09-07T08:00")  # next Monday


def test_missing_time_is_reported_not_guessed():
    out = parse_event("Trag morgen Lernen ein", now=NOW)
    assert "time" in out["missing"] and out["start"] is None


def test_hours_duration_in_words():
    out = parse_event("Leg übermorgen um 15 Uhr für zwei Stunden Physikum ein", now=NOW)
    assert out["start"].startswith("2026-09-04T15:00")
    assert out["end"].startswith("2026-09-04T17:00")
    assert "Physikum" in out["title"]


# -- the store -------------------------------------------------------------

def _store(tmp_path) -> CalendarStore:
    return CalendarStore(tmp_path / "events.json")


def test_create_persists_across_instances(tmp_path):
    store = _store(tmp_path)
    event = store.create(title="Testtermin", start=NOW.isoformat())
    again = CalendarStore(tmp_path / "events.json")
    stored = again.get(event["id"])
    assert stored and stored["title"] == "Testtermin"
    assert stored["end"] > stored["start"]


def test_list_filters_by_range_and_query(tmp_path):
    store = _store(tmp_path)
    store.create(title="Früh", start=NOW.isoformat())
    store.create(title="Spät", start=(NOW + timedelta(days=10)).isoformat())
    week = store.list(start=NOW.isoformat(), end=(NOW + timedelta(days=7)).isoformat())
    assert [e["title"] for e in week] == ["Früh"]
    assert [e["title"] for e in store.list(query="spät")] == ["Spät"]


def test_update_and_delete(tmp_path):
    store = _store(tmp_path)
    event = store.create(title="Alt", start=NOW.isoformat())
    updated = store.update(event["id"], title="Neu", location="Bibliothek")
    assert updated["title"] == "Neu" and updated["location"] == "Bibliothek"
    assert store.delete(event["id"]) is True
    assert store.get(event["id"]) is None
    assert store.delete(event["id"]) is False


def test_create_without_title_refuses(tmp_path):
    with pytest.raises(ValueError):
        _store(tmp_path).create(title="  ", start=NOW.isoformat())


def test_due_reminders_fire_once(tmp_path):
    store = _store(tmp_path)
    start = NOW + timedelta(minutes=10)
    store.create(title="Bald", start=start.isoformat(), reminder_minutes=15)
    due = store.due_reminders(now=NOW)
    assert [e["title"] for e in due] == ["Bald"]
    assert store.due_reminders(now=NOW) == []  # announced exactly once


def test_ics_round_trip(tmp_path):
    store = _store(tmp_path)
    store.create(title="Export mich", start=NOW.isoformat(), location="Uni", notes="Zeile 1")
    text = store.export_ics()
    assert "BEGIN:VEVENT" in text and "SUMMARY:Export mich" in text
    other = CalendarStore(tmp_path / "other.json")
    assert other.import_ics(text) == 1
    assert other.list()[0]["title"] == "Export mich"


# -- the deterministic intent ----------------------------------------------

def test_intent_layer_types_the_owner_sentence():
    from service.intents import TopIntent, understand

    u = understand("Trag morgen um 14 Uhr einen Testtermin für 30 Minuten ein.")
    assert u.top is TopIntent.SYSTEM_CONTROL
    assert u.action is not None and u.action.operation == "calendar.create"


def test_intent_layer_queries_and_opening_stay_apart():
    from service.intents import understand

    q = understand("Welche Termine habe ich morgen?")
    assert q.action is not None and q.action.operation == "calendar.query"
    o = understand("Öffne den Kalender.")
    assert o.action is not None and o.action.operation == "system.open_view" and o.action.target == "calendar"
