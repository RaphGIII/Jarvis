"""Capability composition: reuse before development, typed steps before shell."""

from __future__ import annotations

import json
from types import SimpleNamespace

from service.composer import Composer, Plan, Step, looks_compound, primitive_from_manifest


class Model:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt, **_):
        self.prompts.append(prompt)
        return json.dumps(self.answer) if not isinstance(self.answer, str) else self.answer


def test_a_study_session_is_composed_from_existing_primitives_not_developed():
    composer = Composer(capabilities=[{"capability_id": "system.screen.capture", "status": "active", "description": "Take a screenshot.",
                                       "input_schema": "{'properties': {'path': {'type': 'string'}}}"}])
    model = Model({"mode": "doing", "steps": [
        {"step": "file.open", "path": "Biologie.pdf"},
        {"step": "timer.start", "minutes": 25, "label": "study"},
        {"step": "music.play", "query": "lofi"},
    ]})
    plan = composer.plan("Beginne meine Lernsession: öffne Biologie.pdf, stell einen Timer auf 25 Minuten und spiel lofi.", model)
    assert plan.mode == "doing" and plan.executable and not plan.missing
    assert [s.step for s in plan.steps] == ["file.open", "timer.start", "music.play"]
    assert plan.steps[1].arguments == {"minutes": 25, "label": "study"}
    assert "capability:system.screen.capture" in model.prompts[0], "registered capabilities are on the menu"
    assert "shell" not in model.prompts[0].lower()


def test_only_the_missing_primitive_is_named_as_a_gap():
    composer = Composer()
    plan = composer.parse("dim the lights and play music", json.dumps({"mode": "doing", "steps": [
        {"step": "MISSING", "primitive": "lights.dim", "purpose": "dim the room lights"},
        {"step": "music.play", "query": "jazz"},
    ]}))
    assert plan.mode == "doing" and not plan.executable
    assert plan.missing == ["lights.dim"]
    assert [s.status for s in plan.steps] == ["missing", "planned"]


def test_a_step_the_menu_never_offered_is_rejected_not_run():
    plan = Composer().parse("x", json.dumps({"mode": "doing", "steps": [{"step": "shell.run", "command": "rm -rf /"}]}))
    assert not plan.executable and plan.missing == ["shell.run"] and plan.mode == "learning"


def test_device_requirements_are_checked_against_the_context():
    composer = Composer(context_requirements=["screen"])  # no speaker here
    plan = composer.parse("play music", json.dumps({"mode": "doing", "steps": [{"step": "music.play", "query": "jazz"}]}))
    assert not plan.executable and "speaker" in plan.missing[0]


def test_execution_stops_at_the_first_failed_step_and_keeps_receipts():
    plan = Composer().parse("x", json.dumps({"mode": "doing", "steps": [
        {"step": "file.write", "path": "a.txt", "content": "1"}, {"step": "file.read", "path": "missing.txt"}, {"step": "say", "text": "done"}]}))
    calls = []

    def run(step):
        calls.append(step.step)
        ok = step.step != "file.read"
        return SimpleNamespace(ok=ok, id=f"r_{step.step}", detail="ok" if ok else "no such file")

    receipts = Composer().execute(plan, run)
    # file.read is an observation (optional by default): its failure is kept
    # on record and the plan goes on to the step that carries the goal.
    assert calls == ["file.write", "file.read", "say"]
    assert [s.status for s in plan.steps] == ["done", "failed", "done"]
    assert [s.role for s in plan.steps] == ["required", "optional", "optional"]
    assert plan.steps[1].detail == "no such file" and len(receipts) == 3


def test_a_failed_required_step_stops_the_rest_unless_a_replan_takes_over():
    plan = Composer().parse("x", json.dumps({"mode": "doing", "steps": [
        {"step": "file.write", "path": "a.txt", "content": "1"}, {"step": "project.create", "name": "p"}, {"step": "say", "text": "done"}]}))
    calls = []

    def run(step):
        calls.append(step.step)
        ok = step.step != "file.write"
        return SimpleNamespace(ok=ok, id=f"r_{step.step}", detail="ok" if ok else "disk full")

    Composer().execute(plan, run)
    assert calls == ["file.write"]
    assert [s.status for s in plan.steps] == ["failed", "skipped", "skipped"]

    # With a replan the remainder is replaced and executed.
    plan = Composer().parse("x", json.dumps({"mode": "doing", "steps": [
        {"step": "file.write", "path": "a.txt", "content": "1"}, {"step": "say", "text": "done"}]}))
    calls.clear()

    def replan(current, failed):
        fresh = Composer().parse("x", json.dumps({"mode": "doing", "steps": [{"step": "note.create", "title": "a", "text": "1"}, {"step": "say", "text": "done"}]}))
        fresh.replans = 1
        return fresh

    Composer().execute(plan, run, replan=replan)
    assert calls == ["file.write", "note.create", "say"]
    assert [s.status for s in plan.steps] == ["failed", "skipped", "done", "done"] and plan.replans == 1


def test_questions_are_answering_and_compound_detection():
    assert Composer().parse("what is x", json.dumps({"mode": "answering"})).mode == "answering"
    assert looks_compound("Öffne die Datei und dann spiel Musik")
    assert looks_compound("Start a timer, open notes.txt and play jazz")
    assert not looks_compound("Spiel Lose Yourself von Eminem")
    assert not looks_compound("Wer bist du?")


def test_manifests_become_primitives_with_their_inputs():
    prim = primitive_from_manifest({"capability_id": "archive.zip.create", "status": "active", "description": "Package a folder into a zip.\nmore",
                                    "input_schema": "{'properties': {'path': {'type': 'string'}, 'client_secret': {'type': 'string'}}}"})
    assert prim.name == "capability:archive.zip.create" and prim.inputs == {"path": "<string>"} and prim.purpose.startswith("Package")
    assert primitive_from_manifest({"capability_id": "x", "status": "disabled"}) is None
