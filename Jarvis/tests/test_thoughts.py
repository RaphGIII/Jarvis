"""Proactive thoughts: real evidence in, bounded and deduplicated observations out."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from runtime.thoughts import (IMPORTANCE, Thought, ThoughtEngine, ThoughtStore, capability_degradation, project_inactivity,
                              repeated_corrections, repeated_failures, speak_policy)


def failed(i, reason="capability:learned.ausgabe_dateipfad_zeilen: File not found"):
    return {"id": f"m_{i}", "title": f"Mission {i}", "goal": f"goal {i}", "state": "failed", "reason": reason, "blockers": []}


def test_a_repeated_real_failure_becomes_one_insight_with_evidence():
    thoughts = repeated_failures([failed(1), failed(2), failed(3), {"id": "m_9", "state": "completed", "reason": ""}])
    assert len(thoughts) == 1
    t = thoughts[0]
    assert t.type == "INSIGHT" and t.importance == "HIGH" and len(t.evidence) == 3
    assert all(e["kind"] == "mission" and e["ref"].startswith("m_") for e in t.evidence)
    assert "derselben Ursache" in t.title


def test_a_single_failure_is_not_a_pattern():
    assert repeated_failures([failed(1)]) == []


def test_the_same_insight_is_not_spammed(tmp_path):
    store = ThoughtStore(tmp_path / "thoughts.json")
    first, outcome1 = store.offer(repeated_failures([failed(1), failed(2)])[0])
    second, outcome2 = store.offer(repeated_failures([failed(1), failed(2), failed(3)])[0])
    assert outcome1 == "new" and outcome2 == "cooling"
    assert second.thought_id == first.thought_id and second.count == 2
    assert len(store.list()) == 1
    assert ThoughtStore(tmp_path / "thoughts.json").list()[0].count == 2, "persisted"


def test_low_value_thoughts_stay_silent_and_high_ones_are_said_once(tmp_path):
    store = ThoughtStore(tmp_path / "t.json")
    old = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    facts = {"missions": [failed(1), failed(2), failed(3)],
             "projects": [{"id": "p1", "title": "Voice", "state": "executing", "updated_at": old, "origin": "owner"}],
             "corrections": [], "capabilities": []}
    notes = []
    engine = ThoughtEngine(store, facts=lambda: facts, proactivity=lambda: 50, emit=lambda k, p: notes.append((k, p)))
    record = engine.tick("manual", force=True)
    assert record["new"] == 2
    assert not [n for n in notes if n[0] == "notification"], "HIGH is said at the next natural moment, not as an interrupt"
    nxt = engine.next_to_say()
    assert nxt is not None and nxt.type == "INSIGHT"
    store.mark_delivered(nxt.thought_id, "spoken")
    assert engine.next_to_say() is None, "the LOW reminder stays in the inbox"
    assert store.get(nxt.thought_id).status == "IMPORTANT"


def test_the_cooldown_prevents_re_running_every_second(tmp_path):
    engine = ThoughtEngine(ThoughtStore(tmp_path / "t.json"), facts=lambda: {"missions": [], "projects": [], "corrections": [], "capabilities": []})
    assert engine.tick("mission_finished", force=True)["ran"]
    assert engine.tick("mission_finished")["ran"] is False


def test_dismissing_a_type_three_times_demotes_it(tmp_path):
    store = ThoughtStore(tmp_path / "t.json")
    for i in range(3):
        t, _ = store.offer(Thought(type="REMINDER", title=f"r{i}", text="x", why_it_matters="y", evidence=[{"kind": "project", "ref": str(i), "summary": ""}], importance="MEDIUM"))
        store.set_status(t.thought_id, "DISMISSED")
    demoted, _ = store.offer(Thought(type="REMINDER", title="r9", text="x", why_it_matters="y", evidence=[{"kind": "project", "ref": "9", "summary": ""}], importance="MEDIUM"))
    assert demoted.importance == "LOW"
    store.mute("REMINDER")
    assert store.offer(Thought(type="REMINDER", title="r10", text="x", why_it_matters="y", evidence=[{"kind": "project", "ref": "10", "summary": ""}]))[1] == "muted"


def test_a_dismissed_thought_does_not_come_back(tmp_path):
    store = ThoughtStore(tmp_path / "t.json")
    t, _ = store.offer(repeated_failures([failed(1), failed(2)])[0])
    store.set_status(t.thought_id, "DISMISSED")
    assert store.offer(repeated_failures([failed(1), failed(2)])[0])[1] == "dismissed"


def test_a_thought_needs_evidence():
    with pytest.raises(ValueError):
        Thought(type="IDEA", title="x", text="y", why_it_matters="z")


def test_other_detectors_read_real_records():
    old = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
    idle = project_inactivity([{"id": "p1", "title": "Voice", "state": "executing", "updated_at": old, "origin": "owner"},
                               {"id": "p2", "title": "Spotify", "state": "paused", "updated_at": old, "origin": "acquisition"}])
    assert len(idle) == 1 and idle[0].context["project_id"] == "p1"
    corr = repeated_corrections([{"correction_id": f"c{i}", "classification": "ENTITY_RESOLUTION_ERROR", "entities": {"provider": "spotify"}, "what_was_wrong": "wrong track"} for i in range(3)])
    assert len(corr) == 1 and corr[0].type == "OPTIMIZATION"
    warn = capability_degradation([{"capability_id": "x.y", "health": {"state": "failing", "last_error": "File not found", "consecutive_failures": 2}}])
    assert len(warn) == 1 and warn[0].importance == "HIGH"


def test_the_owners_dial_sets_the_policy():
    assert speak_policy(10) == {"speak_at": "URGENT", "interrupt_at": "URGENT"}
    assert speak_policy(50)["speak_at"] == "HIGH"
    assert speak_policy(90)["speak_at"] == "MEDIUM"


def test_the_core_can_save_a_thought_to_knowledge_and_create_a_mission(tmp_path):
    from service.core import JarvisCore

    class Kernel:
        def __init__(self):
            self.state_root = tmp_path
            self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

        def provider(self, tier):
            class P:
                def generate_stream(self, prompt, **_):
                    yield "ok"
            return P()

    core = JarvisCore(kernel=Kernel())
    t, _ = core.thoughts.store.offer(repeated_failures([failed(1), failed(2)])[0])
    saved = core.thought_action(t.thought_id, "save_knowledge")
    assert saved["ok"] and core.knowledge_read(t.title)["ok"]
    assert core.thoughts.store.get(t.thought_id).status == "SAVED"
    made = core.thought_action(t.thought_id, "create_mission")
    assert made["ok"] and core.list_missions()["count"] == 1
    assert core.list_thoughts()["counts"]["ACTED_ON"] == 1
