"""Feedback must change future behaviour -- scoped, bounded, reversible, inspectable."""

from __future__ import annotations

import json

import pytest

from runtime.adaptation import AdaptiveOwnerModel, AdaptiveRule, classify_context


@pytest.fixture
def model(tmp_path):
    insights = []
    m = AdaptiveOwnerModel(tmp_path / "adaptive.json", on_insight=lambda t, p: insights.append((t, p)))
    m._insights = insights  # type: ignore[attr-defined]
    return m


TECH = {"kind": "technical_explanation"}
CONFIRM = {"kind": "action_confirmation"}


def test_three_too_short_ratings_make_technical_answers_longer_but_not_confirmations(model):
    for _ in range(3):
        model.record_response_feedback(rating="down", category="TOO_SHORT", context=TECH, request="Erkläre die Gluconeogenese")
    lines = model.guidance(TECH)
    assert any("ausführlicher" in l for l in lines), lines
    assert model.guidance(CONFIRM) == [], "the nudge is scoped: confirmations stay concise"


def test_one_rating_is_not_yet_a_rule(model):
    model.record_response_feedback(rating="down", category="TOO_SHORT", context=TECH)
    assert model.guidance(TECH) == [], "one accidental thumb must not change behaviour"
    rule = model.rules[0]
    assert abs(rule.weight) < 0.5 and rule.reversible


def test_positive_feedback_confirms_and_strengthens(model):
    for _ in range(2):
        model.record_response_feedback(rating="down", category="TOO_SHORT", context=TECH)
    conf_before = model.rules[0].confidence
    model.record_response_feedback(rating="up", context=TECH)
    assert model.rules[0].confidence >= conf_before


def test_weights_are_bounded_and_opposing_feedback_reverses(model):
    for _ in range(20):
        model.record_response_feedback(rating="down", category="TOO_SHORT", context=TECH)
    assert model.rules[0].weight <= 2.0
    for _ in range(20):
        model.record_response_feedback(rating="down", category="TOO_LONG", context=TECH)
    assert model.rules[0].weight >= -2.0
    assert len([r for r in model.rules if r.domain == "ANSWER_LENGTH"]) == 1, "one rule per scope, moved -- not a pile"


def test_persistence_survives_restart_and_deletion_reverts(model, tmp_path):
    for _ in range(3):
        model.record_response_feedback(rating="down", category="TOO_SHORT", context=TECH)
    again = AdaptiveOwnerModel(tmp_path / "adaptive.json")
    assert any("ausführlicher" in l for l in again.guidance(TECH))
    rid = again.rules[0].rule_id
    assert again.delete_rule(rid) is True
    assert again.guidance(TECH) == []
    third = AdaptiveOwnerModel(tmp_path / "adaptive.json")
    assert third.guidance(TECH) == [], "deletion is persistent"


def test_owner_rules_outrank_learned_nudges_and_are_editable(model):
    model.add_owner_rule("Bei medizinischen Erklärungen ausführlicher antworten.", domain="ANSWER_LENGTH", scope=TECH)
    for _ in range(3):
        model.record_response_feedback(rating="down", category="TOO_LONG", context=TECH)
    lines = model.guidance(TECH)
    assert lines[0] == "Bei medizinischen Erklärungen ausführlicher antworten."
    rule = next(r for r in model.rules if r.source == "OWNER_RULE")
    model.update_rule(rule.rule_id, {"enabled": False})
    assert "Bei medizinischen" not in " ".join(model.guidance(TECH))


def test_unconfirmed_learning_decays(model):
    for _ in range(3):
        model.record_response_feedback(rating="down", category="TOO_SHORT", context=TECH)
    rule = model.rules[0]
    rule.last_confirmed = "2025-01-01T00:00:00+00:00"  # long ago
    assert rule.effective_confidence() < 0.1
    assert model.guidance(TECH) == [], "stale learning stops steering"


def test_verifier_overruled_three_times_raises_an_insight(model):
    for i in range(3):
        out = model.record_action_feedback(kind="music.play", verdict="RESULT_WAS_SUCCESSFUL", receipt_id=f"r{i}",
                                           request="Spiel Rammstein")
    assert out["insight"] and "music.play" in out["insight"]
    assert model._insights and model._insights[-1][0] == "verifier"
    assert model.lessons["music.play"][0]["receipt_id"] == "r0"


def test_context_classification_is_deterministic():
    assert classify_context(request="Erkläre den Algorithmus", intent="conversation")["kind"] == "technical_explanation"
    assert classify_context(backend="projects.store")["kind"] == "action_confirmation"
    assert classify_context(backend="personality")["kind"] == "small_talk"
    assert classify_context(request="Wie war dein Tag")["kind"] == "conversation"


def test_the_store_is_json_and_carries_provenance(model, tmp_path):
    model.record_response_feedback(rating="down", category="TOO_SHORT", context=TECH, text="zu kurz", request_id="req9")
    data = json.loads((tmp_path / "adaptive.json").read_text(encoding="utf-8"))
    rule = data["rules"][0]
    assert rule["source"] == "EXPLICIT_RATING" and rule["reversible"] and rule["evidence"][0]["request_id"] == "req9"
    assert data["feedback"][0]["text"] == "zu kurz"
