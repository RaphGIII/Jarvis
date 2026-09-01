"""Action must act: through the real request path, an actionable request ends
in an executed action, a mission, one concise question, or a plain "cannot"
-- never in advisory prose.  Project creation end to end, with cleanup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from service.core import JarvisCore
from service.events import EventType


class Provider:
    """FAST_LOCAL stand-in: prose for conversation, a declined plan for actions."""

    def __init__(self, plan=None):
        self.plan = plan or {"action": "none", "reason": "not a real-world action"}
        self.prompts: list[str] = []

    def generate(self, prompt, **_):
        self.prompts.append(prompt)
        return json.dumps(self.plan)

    def generate_stream(self, prompt, **_):
        self.prompts.append(prompt)
        yield "Klar, du könntest ein Projekt erstellen, indem du ..."


class Kernel:
    def __init__(self, root: Path, provider) -> None:
        self.state_root = root
        self._provider = provider
        self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

    def provider(self, tier):
        return self._provider


def make(tmp_path, provider=None):
    from core.kernel import JarvisKernel, KernelConfig

    kernel = JarvisKernel(KernelConfig(state_root=tmp_path / "state"))
    provider = provider or Provider()
    kernel.provider = lambda tier: provider  # type: ignore[assignment]
    kernel.catalog.get = lambda tier: type("S", (), {"model": "stub"})()  # type: ignore[assignment]
    core = JarvisCore(kernel=kernel)
    core.language = "de"
    return core, provider


def ask(core, text, *, wait=20.0, meta=None):
    """Send and wait for the answer.  Generous: under a loaded CPU (the full
    suite, a model probe) the answer thread can take seconds to be scheduled."""

    with core.bus.subscribe(replay=False) as sub:
        core.send_message(text, meta=meta)
        deadline = time.time() + wait
        events = []
        while time.time() < deadline:
            events.extend(sub.drain())
            if any(e.type is EventType.MESSAGE for e in events):
                break
            time.sleep(0.05)
    answers = [e.payload for e in events if e.type is EventType.MESSAGE]
    return (answers[-1]["text"] if answers else ""), events


# --------------------------------------------------------------------------
# project create, end to end
# --------------------------------------------------------------------------

def test_project_create_end_to_end_with_three_tasks_and_cleanup(tmp_path):
    core, provider = make(tmp_path)
    answer, events = ask(core, "Erstelle ein neues Projekt namens Test Alpha und leg die Aufgaben Eins, Zwei und Drei an.")

    understood = [e.payload for e in events if e.type is EventType.TOOL and e.payload.get("understanding")]
    assert understood and understood[0]["understanding"]["top"] == "project_operation"
    receipts = [e.payload["receipt"] for e in events if e.type is EventType.TOOL and e.payload.get("receipt")]
    assert receipts and receipts[0]["kind"] == "project.create" and receipts[0]["verified"], receipts
    assert receipts[0]["evidence"]["title"] == "Test Alpha" and receipts[0]["evidence"]["tasks"] == ["Eins", "Zwei", "Drei"]
    goal = [e.payload for e in events if e.type is EventType.TOOL and str(e.payload.get("summary", "")).startswith("goal:")]
    assert goal and goal[0]["goal"]["GOAL_SATISFIED"] is True
    # persisted and visible through the same API the UI uses
    listed = [p for p in core.list_projects() if p["title"] == "Test Alpha"]
    assert listed and listed[0]["tasks"] == 3
    # concise, natural response; no log dump
    assert answer.startswith("Erledigt.") and "Test Alpha" in answer and "drei Aufgaben" in answer and "rcpt_" not in answer
    assert not provider.prompts, "a deterministic contract never needed the model"
    # controlled cleanup: a TEST-titled record hides from the default galaxy and can be archived
    assert listed[0]["importance"] == "TEST"
    assert all(n["id"] != listed[0]["id"] for n in core.project_graph()["nodes"])
    assert any(n["id"] == listed[0]["id"] for n in core.project_graph(everything=True)["nodes"])
    core.kernel.projects.delete(listed[0]["id"], remove_workspace=True)
    assert not any(p["title"] == "Test Alpha" for p in core.list_projects())


@pytest.mark.parametrize("text", [
    "Mach mir ein Projekt für M1.",
    "Ich möchte ein neues Projekt für meinen Hausbau.",
    "Zeus, erstelle ein neues Projekt für meine Physikumsvorbereitung.",
])
def test_paraphrases_without_project_create_wording_still_create(tmp_path, text):
    core, _ = make(tmp_path)
    answer, events = ask(core, text)
    receipts = [e.payload["receipt"] for e in events if e.type is EventType.TOOL and e.payload.get("receipt")]
    assert receipts and receipts[0]["kind"] == "project.create" and receipts[0]["verified"]
    assert answer.startswith("Erledigt")
    assert "könntest" not in answer


def test_a_second_project_with_the_same_title_is_refused_not_duplicated(tmp_path):
    core, _ = make(tmp_path)
    ask(core, "Erstelle ein Projekt Biochemie.")
    answer, _ = ask(core, "Erstelle ein Projekt Biochemie.")
    assert "gibt es schon" in answer
    assert len([p for p in core.list_projects() if p["title"] == "Biochemie"]) == 1


def test_missing_title_asks_one_question_and_the_answer_completes_it(tmp_path):
    core, _ = make(tmp_path)
    answer, _ = ask(core, "Erstelle ein neues Projekt.")
    assert "heißen" in answer
    answer, events = ask(core, "Sprachtest")
    receipts = [e.payload["receipt"] for e in events if e.type is EventType.TOOL and e.payload.get("receipt")]
    assert receipts and receipts[0]["evidence"]["title"] == "Sprachtest"
    assert answer.startswith("Erledigt")


def test_open_and_list_projects_open_the_view(tmp_path):
    core, _ = make(tmp_path)
    ask(core, "Erstelle ein Projekt Stockfish.")
    answer, events = ask(core, "Zeus, öffne das Stockfish-Projekt.")
    opened = [e.payload for e in events if e.type is EventType.NOTIFICATION and e.payload.get("kind") == "open_view"]
    assert opened and opened[0]["view"] == "projects" and opened[0]["params"]["id"]
    assert "Stockfish" in answer
    answer, events = ask(core, "Zeus, öffne meine Projekte.")
    opened = [e.payload for e in events if e.type is EventType.NOTIFICATION and e.payload.get("kind") == "open_view"]
    assert opened and "Stockfish" in answer


def test_rename_correction_updates_the_project(tmp_path):
    core, _ = make(tmp_path)
    ask(core, "Erstelle ein Projekt Bio.")
    answer, _ = ask(core, "Das Projekt heißt nicht Bio sondern Biochemie.")
    assert "Biochemie" in answer and any(p["title"] == "Biochemie" for p in core.list_projects())
    assert not any(p["title"] == "Bio" for p in core.list_projects())


def test_delete_asks_for_confirmation_and_no_declines(tmp_path):
    core, _ = make(tmp_path)
    ask(core, "Erstelle ein Projekt Alt.")
    answer, _ = ask(core, "Lösche das Projekt Alt.")
    assert "wirklich" in answer and any(p["title"] == "Alt" for p in core.list_projects())
    answer, _ = ask(core, "Nein.")
    assert "nichts gemacht" in answer.lower() and any(p["title"] == "Alt" for p in core.list_projects())
    ask(core, "Lösche das Projekt Alt.")
    answer, _ = ask(core, "Ja.")
    assert "gelöscht" in answer and not any(p["title"] == "Alt" for p in core.list_projects())


# --------------------------------------------------------------------------
# action must act: no advisory prose
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Erstelle eine Notiz über Mitochondrien.",
    "Zeus, mach einen Screenshot.",
    "Zeus, speichere das in Knowledge.",
    "Starte eine Mission für die Prüfungsvorbereitung.",
    "Erstelle ein neues Projekt.",
    "Zeus, öffne meine Projekte.",
])
def test_an_action_request_never_degrades_into_prose(tmp_path, text):
    core, provider = make(tmp_path)
    answer, events = ask(core, text, wait=8.0)
    assert answer, "an action request must get an answer"
    assert "könntest" not in answer, answer
    acted = any(e.type is EventType.TOOL and (e.payload.get("receipt") or e.payload.get("action") or "executing" in str(e.payload.get("summary", ""))) for e in events)
    acted = acted or any(e.type is EventType.NOTIFICATION and e.payload.get("kind") == "open_view" for e in events)
    asked = any(w in answer for w in ("?", "fehlt", "Sag mir", "heißen"))
    mission = core.list_missions()["count"] > 0
    refused = "nicht ausführen" in answer or "kann ich" in answer
    assert acted or asked or mission or refused, (answer, [e.payload.get("summary") for e in events if e.type is EventType.TOOL])


def test_low_speech_confidence_asks_before_acting(tmp_path):
    core, _ = make(tmp_path)
    answer, _ = ask(core, "Lege ein neues Projekt Biochemie an.", meta={"source": "microphone", "speech_level": "low", "speech_confidence": 0.3, "normalized": "Lege ein neues Projekt Biochemie an."})
    assert "nicht sicher" in answer and not any(p["title"] == "Biochemie" for p in core.list_projects())
    answer, _ = ask(core, "Ja.")
    assert answer.startswith("Erledigt") and any(p["title"] == "Biochemie" for p in core.list_projects())


def test_a_self_repair_request_becomes_self_development_not_an_action(tmp_path):
    core, _ = make(tmp_path)
    calls = []
    core._answer_by_self_development = lambda text, scope, classification=None: calls.append(text)  # type: ignore[assignment]
    core._answer("Du hast mich bei Projekterstellung gerade falsch verstanden. Finde den Fehler und repariere dich.", "")
    assert calls


# --------------------------------------------------------------------------
# corrections through the chat
# --------------------------------------------------------------------------

def test_no_i_meant_after_a_spoken_request_learns_vocabulary_and_reruns(tmp_path):
    core, _ = make(tmp_path)
    ask(core, "Öffne das Starkfisch-Projekt.", meta={"source": "microphone", "raw_transcript": "Öffne das Starkfisch-Projekt.", "normalized": "Öffne das Starkfisch-Projekt."})
    answer, events = ask(core, "Nein, ich meinte Stockfish.")
    messages = [e.payload["text"] for e in events if e.type is EventType.MESSAGE]
    assert any("Verstanden" in m and "Stockfish" in m for m in messages), messages
    assert any(row["meant"] == "Stockfish" and row["heard"] == "starkfisch" for row in core.voice.vocabulary.list())
    reruns = [e.payload for e in events if e.type is EventType.USER_MESSAGE and (e.payload.get("meta") or {}).get("source") == "correction_rerun"]
    assert reruns and "Stockfish" in reruns[0]["text"]
    # and the normaliser applies it from now on
    assert core._normalizer().apply("Öffne das Starkfisch Projekt").text.startswith("Öffne das Stockfish")


# --------------------------------------------------------------------------
# personality
# --------------------------------------------------------------------------

def test_who_are_you_is_answered_as_zeus_not_as_a_system_description(tmp_path):
    core, provider = make(tmp_path)
    answer, _ = ask(core, "Zeus, wer bist du?")
    assert answer.startswith("Ich bin Zeus") or answer.startswith("Zeus.")
    assert "Assistent" in answer and "Wahrnehmung" not in answer and "Gefühl" not in answer
    assert not provider.prompts


def test_conversation_sends_the_personality_as_the_system_message(tmp_path):
    core, provider = make(tmp_path)

    class Recording(Provider):
        def __init__(self):
            super().__init__()
            self.systems = []

        def generate_stream(self, prompt, **kw):
            self.systems.append(kw.get("system", ""))
            self.prompts.append(prompt)
            yield "Alles gut."

    rec = Recording()
    core.kernel.provider = lambda tier: rec  # type: ignore[assignment]
    ask(core, "Erzähl mir etwas über Sterne.")
    assert rec.systems and "Character:" in rec.systems[0] and "Your job is to" not in rec.systems[0]
    assert rec.prompts[0].strip().startswith(("user:", "Recent conversation:"))


def test_a_mishear_corrected_after_a_spoken_create_renames_instead_of_creating_twice(tmp_path):
    core, _ = make(tmp_path)
    ask(core, "Erstelle ein neues Projekt namens Sprachtist Audio.", meta={"source": "microphone", "raw_transcript": "Erstelle ein neues Projekt namens Sprachtist Audio.", "normalized": "Erstelle ein neues Projekt namens Sprachtist Audio."})
    assert any(p["title"] == "Sprachtist Audio" for p in core.list_projects())
    answer, events = ask(core, "Nein, ich meinte Sprachtest.")
    messages = [e.payload["text"] for e in events if e.type is EventType.MESSAGE]
    # the acknowledgement arrives first; the rename follows in the same turn
    deadline = time.time() + 10
    while time.time() < deadline and not any(p["title"] == "Sprachtest Audio" for p in core.list_projects()):
        time.sleep(0.1)
    titles = [p["title"] for p in core.list_projects()]
    assert "Sprachtest Audio" in titles and "Sprachtist Audio" not in titles, titles
    assert len([t for t in titles if t.startswith("Sprach")]) == 1
    assert any("benenne" in m or "heißt jetzt" in m for m in messages), messages
    assert any(row["meant"] == "Sprachtest" for row in core.voice.vocabulary.list())
