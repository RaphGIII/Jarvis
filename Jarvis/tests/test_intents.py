"""Top-level intent and the natural project-creation contract.

Every paraphrase the owner actually uses must become a typed ActionIntent,
and questions/self-development must never reach the project executor.
"""

from __future__ import annotations

import pytest

from service.intents import TopIntent, is_action_request, parse_project_operation, understand

TITLES = ["Stockfish", "ZEUS", "Biochemie", "Test Alpha"]


@pytest.mark.parametrize("text,title", [
    ("Mach mir ein Projekt für M1.", "M1"),
    ("Lege ein neues Projekt Biochemie an.", "Biochemie"),
    ("Ich möchte ein neues Projekt für meinen Hausbau.", "Hausbau"),
    ("Erstelle ein neues Projekt für meine Prüfungsvorbereitung.", "Prüfungsvorbereitung"),
    ("Zeus, erstelle ein neues Projekt namens Biochemie.", "Biochemie"),
    ("Zeus, mach ein Projekt für meine M1 Vorbereitung.", "M1 Vorbereitung"),
    ("Erstelle ein neues Projekt „ZEUS Voice“.", "ZEUS Voice"),
    ("Neues Projekt: Physikum", "Physikum"),
    ("Kannst du mir ein Projekt für Biochemie anlegen?", "Biochemie"),
    ("Erstelle ein neues Projekt namens Sprachtest.", "Sprachtest"),
    ("Erstelle ein Projekt mit dem Namen Zeus Testprojekt", "Zeus Testprojekt"),
    ("Create a new project called Exam Prep.", "Exam Prep"),
])
def test_project_creation_paraphrases_yield_the_title(text, title):
    u = understand(text, project_titles=TITLES)
    assert u.top is TopIntent.PROJECT_OPERATION, u.to_dict()
    assert u.action.operation == "project.create" and u.action.arguments["title"] == title


def test_tasks_parent_importance_and_deadline_are_typed_arguments():
    u = understand("Erstelle ein Projekt M1 und leg drei Aufgaben an: Biochemie wiederholen, Anatomie kreuzen, Physik Formeln.", project_titles=TITLES)
    assert u.action.arguments["tasks"] == ["Biochemie wiederholen", "Anatomie kreuzen", "Physik Formeln"]
    u = understand("Erstelle ein neues Projekt namens Test Alpha und leg die Aufgaben Eins, Zwei und Drei an.")
    assert u.action.arguments["title"] == "Test Alpha" and u.action.arguments["tasks"] == ["Eins", "Zwei", "Drei"]
    u = understand("Erstell unter ZEUS ein Teilprojekt Voice.", project_titles=TITLES)
    assert u.action.arguments["title"] == "Voice" and u.action.arguments["parent"] == "ZEUS"
    u = understand("Erstelle ein wichtiges Projekt Hausbau bis Freitag.")
    assert u.action.arguments["importance"] == "FOCUS" and u.action.arguments["deadline"] == "Freitag" and u.action.arguments["title"] == "Hausbau"


def test_a_missing_title_is_a_clarification_not_a_guess():
    u = understand("Erstelle ein neues Projekt.")
    assert u.top is TopIntent.CLARIFICATION and "title" in u.action.missing and "heißen" in u.question


@pytest.mark.parametrize("text,top", [
    ("Was ist ein Projekt?", TopIntent.CONVERSATION),
    ("Verbessere deine Projektansicht.", TopIntent.CONVERSATION),  # the router decides self-development; the project parser stays out
    ("Zeus, wie geht es dir?", TopIntent.CONVERSATION),
    ("Nein, ich meinte Stockfish.", TopIntent.CORRECTION),
    ("Du hast mich bei Projekterstellung gerade falsch verstanden. Finde den Fehler und repariere dich.", TopIntent.SELF_DEVELOPMENT),
    ("Zeus, öffne Mission Control.", TopIntent.SYSTEM_CONTROL),
    ("Zeus, speichere das in Knowledge.", TopIntent.KNOWLEDGE_OPERATION),
    ("Zeus, mach einen Screenshot.", TopIntent.ACTION),
    ("Zeus, spiel Rammstein.", TopIntent.ACTION),
])
def test_top_level_intent_is_semantic(text, top):
    assert understand(text, project_titles=TITLES).top is top


def test_self_development_route_is_honoured_over_the_project_parser():
    from service.routing import route

    text = "Verbessere deine Projektansicht."
    u = understand(text, route=route(text), project_titles=TITLES)
    assert u.top is TopIntent.SELF_DEVELOPMENT


def test_project_read_open_rename_delete_archive_add_tasks():
    assert understand("Zeig mir meine Projekte.").action.operation == "project.list"
    assert understand("Zeus, öffne meine Projekte.").action.operation == "project.list"
    assert understand("Welche Projekte habe ich?").action.operation == "project.list"
    a = understand("Zeus, öffne das Stockfish-Projekt.", project_titles=TITLES).action
    assert a.operation == "project.open" and a.target == "Stockfish"
    assert understand("Zeus, öffne dieses Projekt.").action.target == "__last__"
    a = understand("Das Projekt heißt nicht Bio sondern Biochemie.").action
    assert a.operation == "project.rename" and a.target == "Bio" and a.arguments["title"] == "Biochemie"
    a = understand("Lösche das Projekt Test Alpha.", project_titles=TITLES).action
    assert a.operation == "project.delete" and a.target == "Test Alpha" and a.consequence.value == "irreversible"
    assert understand("Archiviere das Projekt Test Alpha.", project_titles=TITLES).action.operation == "project.archive"
    a = understand("Füge dem Projekt Biochemie die Aufgabe Glykolyse lernen hinzu.", project_titles=TITLES).action
    assert a.operation == "project.add_tasks" and a.target == "Biochemie" and a.arguments["tasks"] == ["Glykolyse lernen"]


@pytest.mark.parametrize("text,expected", [
    ("Erstelle eine Notiz über Mitochondrien.", True),
    ("Öffne Activity.", True),
    ("Kannst du bitte die Musik pausieren?", True),
    ("Ich möchte ein neues Projekt für Physik.", True),
    ("Was kannst du alles?", False),
    ("Wie funktioniert dein Router?", False),
    ("Danke dir.", False),
])
def test_action_requests_are_recognised_as_such(text, expected):
    assert is_action_request(text) is expected
