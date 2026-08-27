"""Top-level routing: what a request *is* is decided before any domain parser runs.

The live failure these pin: a long German request to repair ZEUS's own
desktop lifecycle contained "Wenn ich ZEUS.exe starte", the music parser saw
"start", and the paragraph went to Spotify as a search query.
"""

from __future__ import annotations

import pytest

from service.intent import Intent, classify
from service.music import prose_reason, understand
from service.routing import TopLevelIntent, looks_like_prose, read, route

LIFECYCLE_REQUEST = (
    "Zeus, repariere deinen Desktop- und Supervisor-Lifecycle. Wenn ich dein Fenster mit X schließe, darf kein "
    "undefinierter Zustand entstehen. Ich möchte langfristig, dass das Schließen des Fensters nur deine Oberfläche "
    "ausblendet, während Core und Wakeword weiterlaufen. Wenn ich ZEUS.exe danach erneut starte, soll innerhalb von "
    "höchstens zwei Sekunden wieder dein vorhandenes Fenster erscheinen und dieselbe Core-Instanz verwenden. Es dürfen "
    "niemals unnötig doppelte Core-, speech.listener- oder speech.worker-Prozesse entstehen. Füge außerdem eine klare "
    "Möglichkeit „ZEUS vollständig beenden“ hinzu, die Supervisor, Core, Listener und Worker sauber beendet. Teste "
    "Fenster schließen → erneut öffnen, vollständiges Beenden, Self-Update → Restart und Prozessanzahlen real. "
    "Entwickle, verifiziere und übernimm die Änderung selbst."
)
STARTUP_REQUEST = (
    "Zeus, ändere deinen eigenen Startvorgang: Wenn ich ZEUS.exe starte, soll deine Oberfläche künftig in einem "
    "eigenen Desktop-Fenster erscheinen und nicht in meinem normalen Browser. Entwickle und teste diese Änderung selbst."
)


def test_the_lifecycle_paragraph_is_self_development_not_a_song():
    decision = classify(LIFECYCLE_REQUEST)
    assert decision.intent is Intent.SELF_DEVELOPMENT
    assert decision.top_level == "self_development" and decision.confidence == "high"
    assert understand(LIFECYCLE_REQUEST) is None, "the music parser must not claim a paragraph"


def test_the_startup_paragraph_is_self_development_not_a_song():
    decision = classify(STARTUP_REQUEST)
    assert decision.intent is Intent.SELF_DEVELOPMENT
    assert "zeus.exe" in decision.route.reading.self_refs


@pytest.mark.parametrize(
    "text,expected,top",
    [
        # OBJECT vs OPERATION: the same nouns, different targets.
        ("Play a song.", Intent.MUSIC, TopLevelIntent.REAL_WORLD_ACTION),
        ("Improve how you choose songs.", Intent.SELF_DEVELOPMENT, TopLevelIntent.SELF_DEVELOPMENT),
        ("Verbessere, wie du Lieder auswählst.", Intent.SELF_DEVELOPMENT, TopLevelIntent.SELF_DEVELOPMENT),
        ("Take a screenshot.", Intent.ACTION, TopLevelIntent.REAL_WORLD_ACTION),
        ("Mach einen Screenshot.", Intent.ACTION, TopLevelIntent.REAL_WORLD_ACTION),
        ("Improve your screenshot function.", Intent.SELF_DEVELOPMENT, TopLevelIntent.SELF_DEVELOPMENT),
        ("Repariere deine Screenshot-Funktion, sie ist kaputt.", Intent.SELF_DEVELOPMENT, TopLevelIntent.CAPABILITY_REPAIR),
        ("Open Activity.", Intent.ACTION, TopLevelIntent.REAL_WORLD_ACTION),
        ("Change your Activity view so that it shows durations.", Intent.SELF_DEVELOPMENT, TopLevelIntent.SELF_DEVELOPMENT),
        ("Ändere deine Activity-Ansicht so, dass sie Dauer anzeigt.", Intent.SELF_DEVELOPMENT, TopLevelIntent.SELF_DEVELOPMENT),
        ("Change your core personality to be more formal.", Intent.OWNER_CONFIG, TopLevelIntent.OWNER_CONFIG_CHANGE),
        ("Ändere deine Kernpersönlichkeit: sei förmlicher.", Intent.OWNER_CONFIG, TopLevelIntent.OWNER_CONFIG_CHANGE),
        # Transport commands still work bare.
        ("Pause.", Intent.MUSIC, TopLevelIntent.CONVERSATION),
        ("Weiter.", Intent.MUSIC, TopLevelIntent.CONVERSATION),
        ("Spiel Lose Yourself von Eminem auf Spotify.", Intent.MUSIC, TopLevelIntent.REAL_WORLD_ACTION),
        ("Spiel dein Lieblingslied.", Intent.MUSIC, TopLevelIntent.REAL_WORLD_ACTION),
        ("Zeus, starte die Playlist Focus.", Intent.MUSIC, TopLevelIntent.REAL_WORLD_ACTION),
        # Behaviour changes and placements are self-development.
        ("Ich möchte, dass du künftig kürzer antwortest.", Intent.SELF_DEVELOPMENT, TopLevelIntent.SELF_DEVELOPMENT),
        ("Zeus, show my current GPU utilization subtly next to your eye.", Intent.SELF_DEVELOPMENT, TopLevelIntent.SELF_DEVELOPMENT),
        ("Zeus, füge in deiner Kopfzeile eine kleine Uhr hinzu.", Intent.SELF_DEVELOPMENT, TopLevelIntent.SELF_DEVELOPMENT),
        ("Zeus, mach dein Auge im Leerlauf größer.", Intent.SELF_DEVELOPMENT, TopLevelIntent.SELF_DEVELOPMENT),
        # ...but a request that acts on the world with a self-reference in it is an action.
        ("Ich möchte, dass du mir eine Datei anlegst mit dem Namen test.txt.", Intent.ACTION, TopLevelIntent.REAL_WORLD_ACTION),
        ("Zeus, öffne die Datei bericht.md.", Intent.ACTION, TopLevelIntent.REAL_WORLD_ACTION),
        # Questions are answered, never started as missions.
        ("Kannst du dein Auge größer machen?", Intent.CONVERSATION, TopLevelIntent.CONVERSATION),
        ("Wie geht es mit meinem Projekt weiter?", Intent.CONVERSATION, TopLevelIntent.CONVERSATION),
        ("Wer bist du?", Intent.CONVERSATION, TopLevelIntent.CONVERSATION),
        # The other top-level kinds.
        ("lerne wie man Musik abspielt", Intent.CAPABILITY, TopLevelIntent.CAPABILITY_ACQUISITION),
        ("implementiere einen Sortieralgorithmus", Intent.PROJECT, TopLevelIntent.PROJECT),
        ("Nein, das war falsch, ich meinte die Notiz.", Intent.CORRECTION, TopLevelIntent.OWNER_CORRECTION),
        ("Recherchiere den aktuellen Stand zu Ollama auf der GTX 1070.", Intent.CONVERSATION, TopLevelIntent.RESEARCH),
        ("Was kannst du alles?", Intent.READ, TopLevelIntent.CONVERSATION),
    ],
)
def test_object_and_operation_decide_the_route(text, expected, top):
    decision = classify(text)
    assert decision.intent is expected, decision.reason
    assert decision.route.intent is top, decision.route.to_dict()


def test_a_self_target_records_the_overruled_world_objects_as_a_conflict():
    decision = route("Verbessere, wie du Lieder auswählst.")
    assert decision.intent is TopLevelIntent.SELF_DEVELOPMENT
    assert any("lieder" in c for c in decision.conflicts)


def test_an_owner_correction_forces_the_route_before_it_is_chosen():
    class Row:
        correction_id = "corr_1"
        then = {"overrides": {"intent": "self_development"}}

    decision = route("Spiel bitte die Nummer mit dem Fenster.", corrections=[Row()])
    assert decision.intent is TopLevelIntent.SELF_DEVELOPMENT and decision.forced_by_owner
    assert decision.corrections == ("corr_1",)
    assert classify("Spiel bitte die Nummer mit dem Fenster.", corrections=[Row()]).intent is Intent.SELF_DEVELOPMENT


def test_the_registry_names_make_a_capability_repair_findable():
    decision = route("Repariere bitte dein archive.zip.create, es zählt die Dateien falsch.",
                     capability_names=["archive.zip.create", "music.provider.spotify"])
    assert decision.intent is TopLevelIntent.CAPABILITY_REPAIR
    assert "create" in decision.reading.capability_terms or "zip" in decision.reading.capability_terms


# --------------------------------------------------------------------------
# The provider-level guard
# --------------------------------------------------------------------------

def test_prose_is_never_a_search_query():
    assert looks_like_prose("Lose Yourself Eminem") == ""
    assert looks_like_prose("Bohemian Rhapsody Queen live at Wembley 1986") == ""
    assert looks_like_prose(LIFECYCLE_REQUEST)
    assert looks_like_prose("Wenn ich ZEUS.exe starte, soll dein Fenster erscheinen")
    assert looks_like_prose("one. two. three sentences")
    assert prose_reason("Wenn ich ZEUS.exe starte, soll dein Fenster erscheinen")


def test_the_music_resolver_refuses_prose_before_touching_a_provider(tmp_path):
    from service.music import MusicRequest, MusicService as MusicResolver

    class Registry:
        def all(self):
            return [type("M", (), {"capability_id": "music.provider.spotify", "status": "active"})()]

    resolver = MusicResolver(preferences={"music.default_provider": "spotify"}, capabilities=Registry(), session=object())
    resolver.provider_capability = lambda: type("M", (), {"capability_id": "music.provider.spotify"})()
    resolver.credentials = lambda: (_ for _ in ()).throw(AssertionError("credentials must not be read for prose"))
    outcome = resolver.run(MusicRequest("play", query=LIFECYCLE_REQUEST))
    assert not outcome.receipt.ok and "refused to search spotify" in outcome.receipt.detail
    assert outcome.receipt.evidence.get("guard") == "prose"


def test_start_only_means_play_next_to_a_music_word():
    assert understand("Wenn ich ZEUS.exe starte, soll dein Fenster erscheinen") is None
    assert understand("Starte den Rechner neu") is None
    assert understand("Starte die Playlist Focus").action == "play"


def test_reading_exposes_object_and_operation():
    reading = read("Zeus, spiel Lose Yourself von Eminem.")
    # "yourself" is a track title here; the operation still decides.
    assert reading.operation == "act" and reading.object == "world"
    assert route("Zeus, spiel Lose Yourself von Eminem.").intent is TopLevelIntent.REAL_WORLD_ACTION
    reading = read("Zeus, ändere deinen eigenen Startvorgang.")
    assert reading.operation == "modify" and reading.object == "self" and reading.self_score >= 2
