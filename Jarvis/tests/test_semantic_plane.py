"""The semantic control plane: goals from meaning, never from word overlap.

The founding incident: "Öffne Wikipedia, Glucoluse." was routed to
``learned.ausgabe_dateipfad_zeilen`` — a word counter — because the learned
capability had stored the German stopword "einer" as a keyword and one shared
stopword was a sufficient match.  These tests pin every layer of the fix:

1. German function words can no longer match a capability (registry).
2. A spoken site name resolves to its canonical URL deterministically.
3. Owner aliases ("Uni-Planer" → a real path) normalize, persist and match.
4. The FAST_LOCAL semantic planner is schema-constrained: a tool outside the
   closed operation set is unrepresentable, not merely discouraged.
"""

from __future__ import annotations

import json

import pytest

from capabilities.models import CapabilityManifest
from capabilities.registry import CapabilityRegistry
from service.aliases import AliasStore, classify_target, fold, normalize, parse_teach
from service.semantic import GOAL_SCHEMA, OPERATIONS, SemanticGoal, SemanticPlanner
from service.websearch import known_site, resolve_site

CONTRACT = (
    "run(payload) must accept payload['action'] and return a dict with 'ok' "
    "(bool) and 'error' (str when not ok)."
)


# --------------------------------------------------------------------------
# 1. the registry no longer matches on German stopwords
# --------------------------------------------------------------------------

def _word_counter(tmp_path) -> CapabilityRegistry:
    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(CapabilityManifest(
        capability_id="learned.ausgabe_dateipfad_zeilen",
        description="Count lines, words and characters of a text file. " + CONTRACT,
        creation_metadata={"keywords": [
            "anzahl", "ausgabe", "dateipfad", "einer", "eingabe", "ist", "lerne",
            "rter", "textdatei", "woerter", "zeichen", "zeilen", "zeus", "zaehlt"]},
        entrypoint="capability_modules.learned:run", version="1.0.0", status="active",
    ))
    return registry


def test_wikipedia_does_not_match_the_word_counter(tmp_path):
    """The live incident, verbatim: one German stopword must not be a match."""

    registry = _word_counter(tmp_path)
    goal = ("öffnen einer Webseite mit dem Namen 'Wikipedia' und der "
            "Suchanfrage 'Glucoluse' in einem Browser")
    assert registry.find(goal, limit=1) == []


def test_the_word_counter_is_still_findable_by_its_subject(tmp_path):
    registry = _word_counter(tmp_path)
    found = registry.find("Wie viele Zeilen hat die Textdatei plan.txt?", limit=1)
    assert [m.capability_id for m in found] == ["learned.ausgabe_dateipfad_zeilen"]


def test_pure_function_word_requests_match_nothing(tmp_path):
    registry = _word_counter(tmp_path)
    assert registry.find("Kannst du mir bitte mal eben etwas zeigen", limit=1) == []


# --------------------------------------------------------------------------
# 2. canonical site resolution
# --------------------------------------------------------------------------

def test_wikipedia_resolves_to_the_german_wikipedia():
    url, how = resolve_site("Wikipedia")
    assert url == "https://de.wikipedia.org" and how == "known"


def test_a_bare_domain_is_treated_as_an_address():
    url, how = resolve_site("heise.de")
    assert url == "https://heise.de" and how == "url"


def test_an_unknown_name_becomes_a_results_page_never_a_dead_end():
    url, how = resolve_site("VölligUnbekannteSeite")
    assert how == "search" and url.startswith("https://www.bing.com/search")


def test_known_site_is_true_only_for_resolvable_names():
    assert known_site("wikipedia") and known_site("github.com")
    assert not known_site("VölligUnbekannteSeite")


# --------------------------------------------------------------------------
# 3. owner aliases
# --------------------------------------------------------------------------

def test_alias_normalization_strips_possessives_and_umlauts():
    assert normalize("meinen Uni-Planer") == "uni planer"
    assert normalize("Mein Übungs-Ordner") == "uebungs ordner"


def test_alias_learn_get_and_match(tmp_path):
    store = AliasStore(tmp_path / "aliases.json")
    store.learn("Uni-Planer", "file", r"D:\Studium\Planer.xlsx")
    entry = store.get("meinen Uni-Planer")
    assert entry and entry["kind"] == "file" and entry["value"].endswith("Planer.xlsx")
    # persisted: a fresh store reads the same fact
    again = AliasStore(tmp_path / "aliases.json").get("uni planer")
    assert again and again["value"].endswith("Planer.xlsx")
    assert store.matches("Öffne bitte meinen Uni-Planer")


def test_alias_rejects_unknown_kind(tmp_path):
    store = AliasStore(tmp_path / "aliases.json")
    with pytest.raises(ValueError):
        store.learn("X", "banana", "y")


def test_classify_target_paths_and_urls(tmp_path):
    d = tmp_path / "ordner"
    d.mkdir()
    kind, value = classify_target(str(d))
    assert kind == "folder"
    kind, value = classify_target("www.uni-heidelberg.de/planer")
    assert kind == "url" and value.startswith("https://")
    kind, _ = classify_target("irgendein Name")
    assert kind == ""


def test_parse_teach_understands_the_owner_phrasing():
    pair = parse_teach("Zeus, wenn ich 'Lernplan' sage, meine ich D:\\Studium\\plan.xlsx")
    assert pair == ("Lernplan", "D:\\Studium\\plan.xlsx")
    pair = parse_teach("Merk dir: Uni-Planer ist D:\\Studium\\Planer.xlsx")
    assert pair == ("Uni-Planer", "D:\\Studium\\Planer.xlsx")
    assert parse_teach("Wie spät ist es?") is None


# --------------------------------------------------------------------------
# 4. the schema-constrained planner
# --------------------------------------------------------------------------

class _StructuredProvider:
    """A provider that honours the schema the way Ollama structured output does."""

    def __init__(self, payload):
        self.payload = payload
        self.saw_schema = None
        self.prompt = ""

    def generate_structured(self, prompt, schema, **kwargs):
        self.saw_schema = schema
        self.prompt = prompt
        return json.dumps(self.payload)


def test_planner_returns_a_typed_goal_and_passes_the_schema():
    provider = _StructuredProvider({"operation": "web.open", "target": "Wikipedia",
                                    "confidence": 0.95, "reason": "eine bekannte Website"})
    goal = SemanticPlanner().plan("Öffne Wikipedia.", provider)
    assert isinstance(goal, SemanticGoal)
    assert goal.operation == "web.open" and goal.target == "Wikipedia"
    assert provider.saw_schema is GOAL_SCHEMA
    assert provider.saw_schema["properties"]["operation"]["enum"] == list(OPERATIONS)


def test_planner_rejects_an_operation_outside_the_closed_set():
    provider = _StructuredProvider({"operation": "rm.rf", "target": "/", "confidence": 1.0, "reason": "x"})
    assert SemanticPlanner().plan("Öffne Wikipedia.", provider) is None


def test_planner_context_carries_apps_projects_and_aliases():
    provider = _StructuredProvider({"operation": "app.open", "target": "Spotify",
                                    "confidence": 0.9, "reason": "installierte App"})
    SemanticPlanner().plan("Mach Spotify auf.", provider,
                           apps=["Spotify"], projects=["M1 Physikum"],
                           aliases=[{"name": "Uni-Planer", "kind": "file", "value": "D:\\x.xlsx"}])
    assert "Spotify" in provider.prompt
    assert "M1 Physikum" in provider.prompt
    assert "Uni-Planer" in provider.prompt


def test_planner_survives_a_provider_without_structured_support():
    class Legacy:
        def generate(self, prompt, **kwargs):
            return '{"operation": "music.control", "target": "Rammstein abspielen", "confidence": 0.8, "reason": "Musik"}'

    goal = SemanticPlanner().plan("Spiel Rammstein.", Legacy())
    assert goal is not None and goal.operation == "music.control"


def test_planner_returns_none_on_garbage():
    class Broken:
        def generate_structured(self, prompt, schema, **kwargs):
            return "keine Ahnung"

    assert SemanticPlanner().plan("Öffne Wikipedia.", Broken()) is None


def test_fold_is_stable_for_german():
    assert fold("Übungs-Ördner ß") == "uebungs-oerdner ss"
