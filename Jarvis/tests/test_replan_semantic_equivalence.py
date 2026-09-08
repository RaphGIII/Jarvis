"""A replacement plan has to be about the same thing the owner asked for.

Reproduced live on 2026-09-08: the owner asked for the SHA-256 checksum of a
file, the checksum capability was failing, the composer replanned around it
with ``file.read`` plus a line/word counter, both steps ran and verified, and
ZEUS reported *"Ziel erreicht (2 Schritte, verifiziert)"* with
``GOAL_SATISFIED=True``. Two true statements -- the steps ran, the steps were
verified -- were used to justify a third that was false.

``EXECUTION_VERIFIED`` answers "did what we ran work". Nothing answered "is what
we ran the thing that was asked for", and after a replan those are different
questions. This module pins the second one.
"""

from __future__ import annotations

from typing import Any

import pytest

from capabilities.models import GoalEnvelope
from service.composer import GoalEvaluation, Plan, Step, evaluate_goal


class _Receipt:
    """A receipt that says the step worked, because in the live case they did."""

    def __init__(self, receipt_id: str, kind: str, ok: bool = True, verified: bool = True) -> None:
        self.id = receipt_id
        self.kind = kind
        self.ok = ok
        self.verified = verified


def _plan(goal: str, steps: list[Step], *, replans: int = 1) -> Plan:
    plan = Plan(goal=goal, mode="doing", steps=steps)
    plan.replans = replans
    return plan


# --------------------------------------------------------------------------
# The live false positive, exactly
# --------------------------------------------------------------------------


def test_the_live_checksum_false_positive_is_not_goal_satisfied() -> None:
    """The reproduction. Before the fix this returned GOAL_SATISFIED=True."""

    goal = "Zeus, wie lautet die sha256 Pruefsumme von zeus_acceptance.txt?"
    steps = [
        Step(step="capability:local.berechne.sha.256_pruefsumme", status="replanned", role="required",
             detail="AttributeError: '_hashlib.HASH' object has no attribute 'hex_digest'"),
        Step(step="file.read", status="done", role="optional", receipt_id="r1",
             arguments={"path": "zeus_acceptance.txt"}),
        Step(step="capability:learned.ausgabe_dateipfad_zeilen", status="done", role="required", receipt_id="r2",
             arguments={"file_path": "zeus_acceptance.txt"}),
    ]
    receipts = [_Receipt("r1", "file.read"), _Receipt("r2", "capability.learned.ausgabe_dateipfad_zeilen")]

    evaluation = evaluate_goal(_plan(goal, steps), receipts)

    assert evaluation.executed is True
    assert evaluation.execution_verified is True, "the steps really did run and verify"
    assert evaluation.goal_satisfied is False, "but a line count is not a checksum"
    assert any("equival" in reason.lower() or "checksum" in reason.lower() or "FILE_CHECKSUM" in reason
               for reason in evaluation.reasons), evaluation.reasons


def test_the_same_plan_without_a_replan_is_left_alone() -> None:
    """The check is about replacements. A first plan is the planner's own reading."""

    goal = "Lies mir zeus_acceptance.txt vor und zaehle die Zeilen"
    steps = [
        Step(step="file.read", status="done", role="required", receipt_id="r1"),
        Step(step="capability:learned.ausgabe_dateipfad_zeilen", status="done", role="required", receipt_id="r2"),
    ]
    receipts = [_Receipt("r1", "file.read"), _Receipt("r2", "capability.learned")]

    evaluation = evaluate_goal(_plan(goal, steps, replans=0), receipts)

    assert evaluation.goal_satisfied is True


def test_a_replan_that_stays_on_the_same_goal_still_satisfies_it() -> None:
    """A replacement is not automatically wrong -- only an unrelated one is."""

    goal = "Zeus, berechne mir die SHA-256-Pruefsumme der Datei zeus_acceptance.txt"
    steps = [
        Step(step="capability:local.berechne.sha.256_pruefsumme", status="replanned", role="required",
             detail="timed out"),
        Step(step="capability:files.sha256.fallback", status="done", role="required", receipt_id="r1",
             arguments={"file_path": "zeus_acceptance.txt"},
             purpose="compute the sha256 checksum of the file"),
    ]
    receipts = [_Receipt("r1", "capability.files.sha256.fallback")]

    evaluation = evaluate_goal(_plan(goal, steps), receipts)

    assert evaluation.goal_satisfied is True, evaluation.reasons


# --------------------------------------------------------------------------
# Held-out: one goal family per row, and the substitutions that must not pass
# --------------------------------------------------------------------------


HELD_OUT = [
    pytest.param(
        "Zeus, berechne mir die SHA-256-Pruefsumme der Datei bericht.txt",
        "capability:local.berechne.sha.256_pruefsumme",
        ["file.read", "capability:learned.ausgabe_dateipfad_zeilen"],
        id="checksum-replaced-by-read-and-line-count",
    ),
    pytest.param(
        "Zeus, welcher Ordner auf D: ist am groessten?",
        "fs.largest",
        ["fs.list"],
        id="largest-folder-replaced-by-a-listing",
    ),
    pytest.param(
        "Zeus, spiel Rammstein",
        "music.play",
        ["web.search", "say"],
        id="playback-replaced-by-a-search",
    ),
    pytest.param(
        "Zeus, oeffne Spotify",
        "app.open",
        ["web.open"],
        id="app-replaced-by-a-web-page",
    ),
    pytest.param(
        "Zeus, lies mir die Startseite von example.com vor",
        "web.read_summary",
        ["knowledge.search"],
        id="web-fetch-replaced-by-memory",
    ),
    pytest.param(
        "Zeus, trag mir morgen um 10 Uhr Zahnarzt in den Kalender ein",
        "calendar.create",
        ["note.create"],
        id="calendar-entry-replaced-by-a-note",
    ),
    pytest.param(
        "Zeus, erstelle ein Bild vom Universum mit Galaxien",
        "image.generate",
        ["web.search", "file.write"],
        id="image-generation-replaced-by-a-search",
    ),
]


@pytest.mark.parametrize("goal,failed_step,replacement", HELD_OUT)
def test_an_unrelated_replacement_never_satisfies_the_goal(goal: str, failed_step: str, replacement: list[str]) -> None:
    steps = [Step(step=failed_step, status="replanned", role="required", detail="it failed")]
    receipts = []
    for index, name in enumerate(replacement):
        steps.append(Step(step=name, status="done", role="required", receipt_id=f"r{index}"))
        receipts.append(_Receipt(f"r{index}", name))

    evaluation = evaluate_goal(_plan(goal, steps), receipts)

    assert evaluation.execution_verified is True, "every replacement step ran and verified"
    assert evaluation.goal_satisfied is False, f"{replacement} is not {goal!r}"
    assert evaluation.reasons


@pytest.mark.parametrize("goal,failed_step,replacement", HELD_OUT)
def test_the_goal_envelope_of_each_family_is_recognised(goal: str, failed_step: str, replacement: list[str]) -> None:
    """The check needs to know what the request was about at all."""

    from service.composer import goal_family

    family = goal_family(goal, failed_step)

    assert family, f"no family recognised for {goal!r} / {failed_step}"
    assert all(not _serves(family, name) for name in replacement), family


def _serves(family: str, step: str) -> bool:
    from service.composer import step_serves_family

    return step_serves_family(family, step)


def test_a_goal_family_nothing_is_known_about_is_not_blocked() -> None:
    """An unrecognised goal keeps the old behaviour rather than failing shut.

    The check can only speak about families it knows. Refusing everything it
    cannot classify would turn a semantic guard into an outage.
    """

    goal = "Zeus, mach irgendetwas Undefiniertes mit dem Ding"
    steps = [
        Step(step="some.primitive", status="replanned", role="required", detail="failed"),
        Step(step="other.primitive", status="done", role="required", receipt_id="r1"),
    ]

    evaluation = evaluate_goal(_plan(goal, steps), [_Receipt("r1", "other.primitive")])

    assert evaluation.goal_satisfied is True


def test_the_envelope_carries_the_target_so_a_different_file_is_not_the_goal() -> None:
    """Right operation, wrong object, is still not the goal."""

    goal = "Zeus, berechne mir die SHA-256-Pruefsumme der Datei bericht.txt"
    steps = [
        Step(step="capability:local.berechne.sha.256_pruefsumme", status="replanned", role="required", detail="failed"),
        Step(step="capability:local.berechne.sha.256_pruefsumme", status="done", role="required", receipt_id="r1",
             arguments={"file_path": "C:/tmp/urlaub.txt"}),
    ]

    evaluation = evaluate_goal(_plan(goal, steps), [_Receipt("r1", "capability.local.berechne.sha.256_pruefsumme")])

    assert evaluation.goal_satisfied is False
    assert any("target" in reason.lower() for reason in evaluation.reasons), evaluation.reasons


def test_goal_envelope_typing_is_unchanged_by_this_check() -> None:
    """The envelope is the existing contract; this module only reads it."""

    assert GoalEnvelope.from_text("Spiel Rammstein").goal_type == "PLAY_MEDIA"
    assert GoalEnvelope.from_text("Wie spaet ist es?").goal_type == "TIME_QUERY"


def test_a_named_primitive_that_matches_no_family_is_not_classified_by_words() -> None:
    """The step is the better signal, and it has already spoken.

    "Starte ein Projekt namens Schach" reads as APP_OPEN to a keyword table,
    which would then reject its own `project.create` replacement. A primitive
    with a real name that matches no family means this goal has no family.
    """

    from service.composer import goal_family

    assert goal_family("Zeus, starte ein Projekt namens Schach", "project.create") == ""
    assert goal_family("Zeus, oeffne Spotify", "app.open") == "APP_OPEN"
    # An opaque capability id still falls back to the words, which is what the
    # fallback is for.
    assert goal_family("Zeus, berechne die sha256 pruefsumme",
                       "capability:learned.irgendwas") == "FILE_CHECKSUM"


def test_a_project_replan_is_not_blocked_by_the_equivalence_check() -> None:
    goal = "Zeus, starte ein Projekt namens Schach und leg eine Notiz an"
    steps = [
        Step(step="project.create", status="replanned", role="required", detail="store was locked"),
        Step(step="project.create", status="done", role="required", receipt_id="r1"),
        Step(step="note.create", status="done", role="required", receipt_id="r2"),
    ]
    receipts = [_Receipt("r1", "project.create"), _Receipt("r2", "note.create")]

    assert evaluate_goal(_plan(goal, steps), receipts).goal_satisfied is True
