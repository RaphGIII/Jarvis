"""Music: generic above the provider, verified below it.

Two things are being defended here.

*The conversational layer must never learn the word Spotify.*  The forbidden
shape is ``if "music" in text: open_spotify()`` -- it makes one provider the
only provider, buries the user's preference in a conditional, and puts
provider-specific behaviour where nothing can verify it.  So the tests check
that the routing is driven by a stored preference and a registered capability,
and that swapping the preference swaps the provider with no code change.

*A provider must not be able to report its own success.*  This project has
already registered a music capability that returned ``{"message": "Dry run:
..."}`` from every branch and played nothing.  So the provider's output is
never the verdict: Windows is asked what is actually playing, and the receipt is
built from that.  The adversarial tests below give the provider every
opportunity to lie -- ok=True, a plausible track name, a confident message --
and check that the receipt still says what the operating system said.
"""

from __future__ import annotations

import pytest

from runtime.preferences import Preferences
from runtime.secrets import SecretStore
from service.intent import Intent, classify
from service.music import (
    MusicRequest,
    MusicService,
    compose,
    extract_query,
    provider_acceptance,
    provider_constraints,
    provider_goal,
    provider_keywords,
    understand,
)
from tools.media_session import MediaState


# --------------------------------------------------------------------------
# Fakes: a Windows that says what we tell it, a provider that does what we say
# --------------------------------------------------------------------------

class FakeSession:
    """Stands in for the operating system's media session."""

    def __init__(self, *states):
        self.states = list(states) or [MediaState(ok=False, error="no session")]
        self.reads = 0

    def read(self, *, app=""):
        self.reads += 1
        return self.states[min(self.reads - 1, len(self.states) - 1)]


class FakeManifest:
    def __init__(self, capability_id, status="active"):
        self.capability_id = capability_id
        self.status = status


class FakeExecution:
    def __init__(self, ok=True, output=None, error=""):
        self.ok = ok
        self.output = output or {}
        self.error = error


class FakeCapabilities:
    """A capability registry plus executor, with no subprocesses involved."""

    def __init__(self, manifests=(), execution=None):
        self._manifests = list(manifests)
        self._execution = execution or FakeExecution()
        self.calls = []

    def list(self):
        return list(self._manifests)

    def execute(self, capability_id, payload=None):
        self.calls.append((capability_id, dict(payload or {})))
        return self._execution


PLAYING = MediaState(ok=True, app="Spotify.exe", title="Lose Yourself", artist="Eminem",
                     status="Playing", position_seconds=3.0, duration_seconds=326.0)
PAUSED = MediaState(ok=True, app="Spotify.exe", title="Lose Yourself", artist="Eminem",
                    status="Paused", position_seconds=3.0)
WRONG_TRACK = MediaState(ok=True, app="Spotify.exe", title="Never Gonna Give You Up",
                         artist="Rick Astley", status="Playing")
NOTHING = MediaState(ok=False, error="no active media session")


def build(tmp_path, *, session, manifests=(("music.provider.spotify",)), execution=None,
          provider="spotify", credentials=True):
    prefs = Preferences(tmp_path / "preferences.json")
    prefs.set("music.default_provider", provider)
    secrets_root = tmp_path / "secrets"
    if credentials:
        secrets_root.mkdir(parents=True, exist_ok=True)
        (secrets_root / "spotify.json").write_text(
            '{"client_id": "abc", "client_secret": "xyz"}', encoding="utf-8"
        )
    return MusicService(
        preferences=prefs,
        capabilities=FakeCapabilities(
            [FakeManifest(name) for name in (manifests[0] if isinstance(manifests[0], tuple) else manifests)]
            if manifests else [],
            execution,
        ),
        secrets=SecretStore(secrets_root),
        session=session,
    )


# --------------------------------------------------------------------------
# The conversational layer knows nothing about providers
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,action,query",
    [
        ("Zeus, spiel Lose Yourself von Eminem auf Spotify.", "play", "Lose Yourself Eminem"),
        ("Spiel Bohemian Rhapsody von Queen", "play", "Bohemian Rhapsody Queen"),
        ("spiel mir Feeling Good von Nina Simone", "play", "Feeling Good Nina Simone"),
        ("Play Get Lucky by Daft Punk", "play", "Get Lucky Daft Punk"),
        ("Spiel Du Hast von Rammstein", "play", "Du Hast Rammstein"),
        ("Pause.", "pause", ""),
        ("Stoppe die Musik", "pause", ""),
        ("Weiter.", "resume", ""),
        ("Naechstes Lied.", "next", ""),
        ("Nächstes Lied bitte", "next", ""),
        ("Vorheriges Lied", "previous", ""),
        ("Was laeuft gerade?", "current", ""),
        ("Was läuft gerade?", "current", ""),
    ],
)
def test_a_music_sentence_becomes_a_provider_independent_request(text, action, query):
    request = understand(text)

    assert request is not None, f"{text!r} was not recognised as music"
    assert request.action == action
    assert request.query == query
    assert "spotify" not in request.to_dict()["action"]


@pytest.mark.parametrize(
    "text",
    [
        "Wie geht es mit meinem Projekt weiter?",   # contains "weiter"
        "Erklaer mir ein Beispiel fuer Rekursion",  # contains "spiel"
        "Erstelle die Datei zeus_test.txt mit dem Inhalt X",
        "Wer bist du?",
        "Welche Faehigkeiten sind verifiziert?",
    ],
)
def test_sentences_that_merely_contain_a_music_word_are_not_music(text):
    """"weiter" ends a German sentence about a project as readily as it
    resumes a track, and "Beispiel" contains "spiel"."""

    assert understand(text) is None
    assert classify(text).intent is not Intent.MUSIC


def test_music_requests_are_classified_as_music():
    assert classify("Spiel Lose Yourself von Eminem").intent is Intent.MUSIC
    assert classify("Pause.").intent is Intent.MUSIC


def test_asking_for_music_with_no_track_named_is_a_resume_not_a_search():
    assert understand("Mach Musik an").action == "resume"
    assert extract_query("Mach Musik an") == ""


# --------------------------------------------------------------------------
# Provider selection comes from a preference
# --------------------------------------------------------------------------

def test_the_provider_comes_from_a_stored_preference(tmp_path):
    service = build(tmp_path, session=FakeSession(PLAYING))

    assert service.provider == "spotify"
    assert service.output == "this_pc"


def test_changing_the_preference_changes_the_provider_with_no_code_change(tmp_path):
    service = build(tmp_path, session=FakeSession(PLAYING),
                    manifests=["music.provider.spotify", "music.provider.deezer"],
                    provider="deezer")

    manifest = service.provider_capability()

    assert service.provider == "deezer"
    assert manifest.capability_id == "music.provider.deezer"


def test_an_unregistered_provider_is_a_gap_not_a_substitution(tmp_path):
    """The failure that matters: no Spotify capability must never become
    'here is a YouTube link instead'."""

    service = build(tmp_path, session=FakeSession(NOTHING), manifests=[])

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    assert outcome.gap is True
    assert outcome.receipt.ok is False
    assert "spotify" in outcome.receipt.detail.lower()
    text = compose(outcome).lower()
    for substitute in ("youtube", "browser", "instead", "stattdessen"):
        assert substitute not in text


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------

def test_playing_by_name_without_credentials_fails_with_the_exact_requirement(tmp_path):
    service = build(tmp_path, session=FakeSession(NOTHING), credentials=False)

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    assert outcome.receipt.ok is False
    assert outcome.requirement is not None
    assert "developer.spotify.com" in outcome.receipt.detail
    assert "SPOTIFY_CLIENT_ID" in str(outcome.requirement.env_vars)


def test_transport_still_works_without_credentials(tmp_path):
    """Pausing does not need a search, so it must not need a search's secret."""

    service = build(tmp_path, session=FakeSession(PAUSED), credentials=False,
                    execution=FakeExecution(ok=True, output={"ok": True}))

    outcome = service.run(MusicRequest("pause"))

    assert outcome.requirement is None
    assert outcome.receipt.verified is True


def test_credentials_never_appear_in_the_receipt(tmp_path):
    service = build(tmp_path, session=FakeSession(PLAYING),
                    execution=FakeExecution(ok=True, output={"ok": True, "client_secret": "xyz"}))

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    assert "xyz" not in str(outcome.receipt.to_dict())


def test_a_partial_credential_counts_as_absent(tmp_path):
    """A client id with no secret cannot authenticate; reporting it as present
    turns a clear 'not configured' into a confusing 'unauthorised' later."""

    root = tmp_path / "secrets"
    root.mkdir(parents=True)
    (root / "spotify.json").write_text('{"client_id": "abc", "client_secret": ""}', encoding="utf-8")

    secret = SecretStore(root).read("spotify", ("client_id", "client_secret"), env_prefix="SPOTIFY")

    assert secret.present is False


# --------------------------------------------------------------------------
# The provider does not get to say whether it worked
# --------------------------------------------------------------------------

def test_a_provider_reporting_success_while_nothing_plays_is_not_verified(tmp_path):
    """The exact shape of the earlier fake music capability."""

    service = build(
        tmp_path,
        session=FakeSession(NOTHING),
        execution=FakeExecution(ok=True, output={"ok": True, "message": "Now playing Lose Yourself"}),
    )

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    assert outcome.receipt.ok is False
    assert outcome.receipt.verified is False
    assert "not treating" in compose(outcome).lower() or "failed" in compose(outcome).lower()


def test_a_provider_that_plays_the_wrong_track_is_not_verified(tmp_path):
    service = build(tmp_path, session=FakeSession(WRONG_TRACK),
                    execution=FakeExecution(ok=True, output={"ok": True, "title": "Lose Yourself"}))

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    assert outcome.receipt.verified is False
    failed_checks = [check.check for check in outcome.receipt.failures]
    assert "the requested track is playing" in failed_checks


def test_a_provider_claiming_to_pause_while_playback_continues_is_not_verified(tmp_path):
    service = build(tmp_path, session=FakeSession(PLAYING),
                    execution=FakeExecution(ok=True, output={"ok": True}))

    outcome = service.run(MusicRequest("pause"))

    assert outcome.receipt.verified is False
    assert any("paused" in check.check for check in outcome.receipt.failures)


def test_playing_the_right_track_verifies(tmp_path):
    service = build(tmp_path, session=FakeSession(PLAYING),
                    execution=FakeExecution(ok=True, output={"ok": True}))

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    assert outcome.receipt.verified is True
    assert outcome.receipt.evidence["track"] == "Lose Yourself - Eminem"
    assert "Lose Yourself" in compose(outcome)


def test_a_subtitled_track_still_counts_as_the_requested_one(tmp_path):
    """Spotify returns "Lose Yourself - From '8 Mile' Soundtrack"; calling that
    a mismatch would fail a request that succeeded."""

    subtitled = MediaState(ok=True, app="Spotify.exe", status="Playing",
                           title="Lose Yourself - From \"8 Mile\" Soundtrack", artist="Eminem")
    service = build(tmp_path, session=FakeSession(subtitled),
                    execution=FakeExecution(ok=True, output={"ok": True}))

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    assert outcome.receipt.verified is True


def test_a_player_that_is_not_the_chosen_provider_fails_the_check(tmp_path):
    other = MediaState(ok=True, app="chrome.exe", title="Lose Yourself", artist="Eminem",
                       status="Playing")
    service = build(tmp_path, session=FakeSession(other),
                    execution=FakeExecution(ok=True, output={"ok": True}))

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    assert outcome.receipt.verified is False
    assert any("Spotify" in check.check for check in outcome.receipt.failures)


def test_next_requires_the_track_to_actually_change(tmp_path):
    same = MediaState(ok=True, app="Spotify.exe", title="Lose Yourself", artist="Eminem",
                      status="Playing")
    service = build(tmp_path, session=FakeSession(same, same, same),
                    execution=FakeExecution(ok=True, output={"ok": True}))

    outcome = service.run(MusicRequest("next"))

    assert outcome.receipt.verified is False
    assert any("changed" in check.check for check in outcome.receipt.failures)


def test_a_provider_that_fails_is_reported_honestly(tmp_path):
    service = build(tmp_path, session=FakeSession(NOTHING),
                    execution=FakeExecution(ok=False, error="track not found"))

    outcome = service.run(MusicRequest("play", query="Nonexistent Song"))

    assert outcome.receipt.ok is False
    assert "track not found" in outcome.receipt.detail
    assert "receipt" in compose(outcome)


def test_every_verification_records_what_was_observed(tmp_path):
    service = build(tmp_path, session=FakeSession(PLAYING),
                    execution=FakeExecution(ok=True, output={"ok": True}))

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    for check in outcome.receipt.verifications:
        assert check.observed, f"{check.check} recorded no observation"


# --------------------------------------------------------------------------
# The acquisition brief must not contain the demo
# --------------------------------------------------------------------------

#: The tracks live acceptance uses. Nothing the builder is shown may name them.
ACCEPTANCE_TRACKS = ("lose yourself", "eminem", "bohemian rhapsody", "queen",
                     "du hast", "rammstein", "get lucky", "daft punk",
                     "feeling good", "nina simone")


@pytest.mark.parametrize("track", ACCEPTANCE_TRACKS)
def test_nothing_the_builder_is_shown_names_an_acceptance_track(track):
    """An example in a brief is the first thing a struggling implementation
    special-cases. If the acceptance track appears anywhere the builder can
    read it, a passing capability might only be recognising a string."""

    shown = " ".join([
        provider_goal("spotify"),
        " ".join(provider_constraints("spotify")),
        " ".join(provider_keywords("spotify")),
        " ".join(" ".join(command) for _text, command in provider_acceptance()),
    ]).lower()

    assert track not in shown


def test_the_provider_goal_forbids_substitution():
    goal = provider_goal("spotify").lower()

    assert "never substitute" in goal
    for substitute in ("youtube", "browser", "generated tone"):
        assert substitute in goal, "the goal must name what it is forbidding"


def test_the_provider_goal_carries_the_facts_that_sank_earlier_attempts():
    goal = provider_goal("spotify")

    assert "winrt" in goal and "winsdk" in goal, "packages that are not installed"
    assert "standard library" in goal.lower() or "PowerShell" in goal
    assert "client_id" in goal and "client_secret" in goal


def test_the_constraints_repeat_the_lessons_of_six_failed_attempts():
    constraints = " ".join(provider_constraints("spotify")).lower()

    assert "not\nimportable" in constraints or "importable" in constraints
    assert "dry_run" in constraints
    assert ".get()" in constraints
    assert "standard library" in constraints


def test_the_local_build_is_gated_on_playback_too():
    """The four standard capability checks -- tests pass, run() returns a dict,
    the marker is gone, no undefined names -- are all satisfiable by a
    capability that does nothing at all. That is how this project once
    registered a music capability that returned "Dry run:" from every branch."""

    from service.music import provider_extra_checks

    checks = provider_extra_checks()
    playback = [check for check in checks if check.name == "playback"]

    assert playback, "the local build must be gated on real playback"
    command = " ".join(playback[0].command)
    assert "media_session" in command, "the gate must read the OS, not the capability"
    assert "resume" in command and "pause" in command


def test_acceptance_commands_check_behaviour_rather_than_wording():
    commands = provider_acceptance()

    assert len(commands) >= 3
    joined = " ".join(" ".join(command) for _text, command in commands)
    assert "INPUT_SCHEMA" in joined
    assert "teleport" in joined, "an unsupported action must be exercised"
    assert "dry_run" in joined


def test_provider_keywords_bridge_the_words_a_user_says_to_the_provider_name():
    """Lexical matching cannot get from "spiel was von den Beatles" to a
    capability called music.provider.spotify. The keywords are the bridge."""

    keywords = provider_keywords("spotify")

    for spoken in ("musik", "lied", "song", "spielen"):
        assert spoken in keywords
    assert "spotify" in keywords


# --------------------------------------------------------------------------
# A broken provider is a defect, not a gap and not the user's fault
# --------------------------------------------------------------------------

def test_a_provider_that_fails_is_marked_as_a_defect(tmp_path):
    """Observed live: a registered, verified Spotify provider whose search was
    rejected by Spotify for sending limit=20, which the API documents as legal
    and rejects anyway. Only real execution finds that."""

    service = build(tmp_path, session=FakeSession(NOTHING),
                    execution=FakeExecution(ok=False, error="Spotify replied 400 Invalid limit"))

    outcome = service.run(MusicRequest("play", query="something"))

    assert outcome.defect
    assert "Invalid limit" in outcome.defect
    assert outcome.gap is False, "the capability exists; rebuilding from scratch is not the fix"


def test_a_provider_contradicted_by_windows_is_a_defect(tmp_path):
    service = build(tmp_path, session=FakeSession(NOTHING),
                    execution=FakeExecution(ok=True, output={"ok": True}))

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    assert outcome.defect
    assert "Windows disagrees" in outcome.defect


def test_a_missing_credential_is_never_treated_as_a_defect(tmp_path):
    """Rebuilding the capability would not supply a credential the user has
    not given it, so calling that a defect would send ZEUS off to fix the
    wrong thing for half an hour."""

    service = build(tmp_path, session=FakeSession(NOTHING), credentials=False)

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    assert outcome.defect == ""
    assert outcome.requirement is not None


def test_a_missing_capability_is_a_gap_not_a_defect(tmp_path):
    service = build(tmp_path, session=FakeSession(NOTHING), manifests=[])

    outcome = service.run(MusicRequest("play", query="x"))

    assert outcome.gap is True
    assert outcome.defect == ""


def test_a_verified_action_reports_no_defect(tmp_path):
    service = build(tmp_path, session=FakeSession(PLAYING),
                    execution=FakeExecution(ok=True, output={"ok": True}))

    outcome = service.run(MusicRequest("play", query="Lose Yourself Eminem"))

    assert outcome.defect == ""


def test_the_search_gate_would_have_caught_the_shipped_bug():
    """Its absence is why the bug shipped: the playback gate proved the
    capability could drive a player, the acceptance commands proved it handled
    contracts honestly, and nothing exercised name-to-track resolution at all."""

    from service.music import provider_extra_checks

    names = [check.name for check in provider_extra_checks()]

    assert names == ["playback", "search", "switch"]
    command = " ".join(provider_extra_checks()[1].command)
    assert "'action':'search'" in command
    assert "track_id" in command, "a search that returns nothing must fail the gate"
    for track in ACCEPTANCE_TRACKS:
        assert track not in command.lower(), "the gate must not name a track"


def test_the_search_gate_reads_credentials_rather_than_carrying_them():
    """A secret in a command string reaches the project record, the logs and
    every process list on the machine."""

    from service.music import provider_extra_checks

    command = " ".join(provider_extra_checks()[1].command)

    assert "SecretStore" in command
    assert "s.get('client_secret')" in command


def test_a_disabled_version_is_reported_as_a_repair_not_a_first_build(tmp_path):
    """Saying "I have no capability yet, building one now" while rebuilding a
    disabled version is a small lie, and it misdescribes what will happen --
    the rebuild starts from the installed source, not from nothing."""

    class Registry:
        def __init__(self, status):
            self._status = status

        def get(self, capability_id):
            return FakeManifest(capability_id, status=self._status)

    service = build(tmp_path, session=FakeSession(NOTHING), manifests=[])
    service.capabilities.registry = Registry("disabled")

    retired = service.retired_capability()

    assert retired is not None
    assert service.capability_id == "music.provider.spotify"


def test_an_active_capability_is_not_reported_as_retired(tmp_path):
    class Registry:
        def get(self, capability_id):
            return FakeManifest(capability_id, status="active")

    service = build(tmp_path, session=FakeSession(NOTHING), manifests=[])
    service.capabilities.registry = Registry()

    assert service.retired_capability() is None


def test_a_provider_that_never_existed_is_not_reported_as_retired(tmp_path):
    class Registry:
        def get(self, capability_id):
            return None

    service = build(tmp_path, session=FakeSession(NOTHING), manifests=[])
    service.capabilities.registry = Registry()

    assert service.retired_capability() is None


# --------------------------------------------------------------------------
# A slow machine is not a broken capability
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reason",
    [
        "the capability did not finish within 120s",
        "could not reach Spotify: timed out",
        "connection reset by peer",
        "no active media session",
        "powershell is not on PATH",
    ],
)
def test_an_environmental_failure_is_not_a_defect(tmp_path, reason):
    """Measured: a verified Spotify provider was disabled, and half an hour
    spent rebuilding it, because one cold call -- PowerShell starting, a token
    fetched, a search over the network -- ran past a 120s budget on a machine
    that had just spent thirty minutes running a 7B model. The code was
    correct. The clock ran out."""

    service = build(tmp_path, session=FakeSession(NOTHING),
                    execution=FakeExecution(ok=False, error=reason))

    outcome = service.run(MusicRequest("play", query="something"))

    assert outcome.receipt.ok is False, "it still failed and must be reported as failing"
    assert outcome.defect == "", f"{reason!r} says nothing about the implementation"


@pytest.mark.parametrize(
    "reason",
    [
        "Spotify replied 400 Bad Request Invalid limit",
        "KeyError: 'tracks'",
        "the provider returned a string, not a dict",
    ],
)
def test_a_behavioural_failure_is_still_a_defect(tmp_path, reason):
    service = build(tmp_path, session=FakeSession(NOTHING),
                    execution=FakeExecution(ok=False, error=reason))

    outcome = service.run(MusicRequest("play", query="something"))

    assert outcome.defect, f"{reason!r} is the implementation's fault"


# --------------------------------------------------------------------------
# The transport commands have to actually be sent
# --------------------------------------------------------------------------

def test_a_transport_command_reaches_powershell(monkeypatch):
    """It did not, for the life of this module.

    The script declared `param($Command)` and the invocation ended
    `-Command <script> -Command <command>`. PowerShell takes the first
    -Command as the script and treats the rest as arguments to it, so
    $Command kept its default of "status": every pause and skip silently read
    the state, changed nothing, and returned the unchanged reading as though
    the player had refused. Reads were unaffected, which is exactly why it went
    unnoticed -- the module looked like it worked.
    """

    from tools import media_session

    captured = {}

    def fake_run(command, **kwargs):
        captured["argv"] = list(command)
        return type("R", (), {"stdout": '{"ok": true, "status": "Paused"}', "stderr": "",
                              "returncode": 0})()

    monkeypatch.setattr(media_session.subprocess, "run", fake_run)
    monkeypatch.setattr(media_session, "_powershell", lambda: "powershell")

    media_session.control("pause", app="spotify")

    argv = captured["argv"]
    # The script now runs from a file, so -Command carries the command itself
    # and PowerShell binds it to the script's param() block. The original defect
    # was a SECOND -Command: PowerShell reads that as an argument to the first.
    assert argv.count("-Command") == 1, "a second -Command becomes an argument, not a parameter"
    assert argv[argv.index("-Command") + 1] == "pause"
    assert argv[argv.index("-App") + 1] == "spotify"
    assert "-File" in argv, "a long script re-parsed on every call costs seconds"


@pytest.mark.parametrize("command", ["play", "pause", "next", "previous"])
def test_every_transport_command_is_passed_through(monkeypatch, command):
    from tools import media_session

    captured = {}
    monkeypatch.setattr(media_session, "_powershell", lambda: "powershell")
    monkeypatch.setattr(
        media_session.subprocess, "run",
        lambda cmd, **kw: captured.update(argv=list(cmd)) or type(
            "R", (), {"stdout": '{"ok": true}', "stderr": "", "returncode": 0})(),
    )

    media_session.control(command, app="")

    argv = captured["argv"]
    assert argv[argv.index("-Command") + 1] == command


def test_an_unsupported_transport_command_is_refused():
    from tools import media_session

    state = media_session.control("self_destruct")

    assert state.ok is False
    assert "unsupported" in state.error


def test_a_quote_cannot_escape_the_generated_script(monkeypatch):
    """Passed as arguments now, so there is no PowerShell syntax to escape into."""

    from tools import media_session

    captured = {}
    monkeypatch.setattr(media_session, "_powershell", lambda: "powershell")
    monkeypatch.setattr(
        media_session.subprocess, "run",
        lambda cmd, **kw: captured.update(argv=list(cmd)) or type(
            "R", (), {"stdout": '{"ok": true}', "stderr": "", "returncode": 0})(),
    )

    media_session.control("status", app="spot'; Remove-Item X -Recurse; '")

    argv = captured["argv"]
    hostile = argv[argv.index("-App") + 1]
    # It arrives as one argv entry, so no shell ever parses it as syntax. This
    # is the reason to pass arguments rather than interpolate into source:
    # there is nothing left to escape.
    assert hostile == "spot'; Remove-Item X -Recurse; '"
    assert not any("Remove-Item" in part for part in argv if part is not hostile)


def test_a_defect_report_carries_the_state_the_world_was_in(tmp_path):
    """"play returned ok=False" says what broke and nothing about when.

    The same action against the same code succeeds or fails depending on what
    the player was doing beforehand, which is exactly the variable a repair has
    to discover. The report states it rather than making the loop go and find
    it -- or worse, not think to look.
    """

    playing_before = MediaState(ok=True, app="Spotify.exe", title="Africa",
                                artist="TOTO", status="Playing")
    service = build(tmp_path, session=FakeSession(playing_before),
                    execution=FakeExecution(ok=False, error="started 'Africa' instead of 'Du Hast'"))

    outcome = service.run(MusicRequest("play", query="Du Hast Rammstein"))

    assert "player BEFORE" in outcome.defect
    assert "Africa" in outcome.defect
    assert "Playing" in outcome.defect
    assert "'query': 'Du Hast Rammstein'" in outcome.defect


def test_a_defect_report_states_observations_not_remedies(tmp_path):
    """A report that names the fix stops being evidence and becomes a patch
    written by whoever wrote the report."""

    service = build(tmp_path, session=FakeSession(PLAYING),
                    execution=FakeExecution(ok=False, error="wrong track"))

    defect = service.run(MusicRequest("play", query="x")).defect.lower()

    for prescription in ("pause first", "you should", "the fix is", "change the"):
        assert prescription not in defect
    assert "establish for yourself" in defect


def test_a_gate_covers_playing_a_track_while_something_else_plays():
    """The defect six gates could not see.

    `playback` proves resume and pause. `search` proves a name resolves. Neither
    proves the thing a user actually asks for: start THIS track, now, while
    something else is playing. A provider passed all six while being unable to
    do exactly that.
    """

    from service.music import provider_extra_checks

    switch = [c for c in provider_extra_checks() if c.name == "switch"]
    assert switch, "nothing gated the state the defect lives in"

    command = " ".join(switch[0].command)
    assert "action='resume'" in command or "'action': 'resume'" in command, (
        "the gate must start from a playing state"
    )
    assert "playing_before.playing" in command
    assert "media_session.read" in command, "the verdict must come from the OS"
    for track in ACCEPTANCE_TRACKS:
        assert track not in command.lower(), "the gate must not name a track"


def test_the_switch_gate_uses_the_providers_own_search_as_its_answer_key():
    """It cannot name a track without becoming hardcodeable, so it asks the
    provider what exists and then holds it to that."""

    from service.music import provider_extra_checks

    command = " ".join(
        part for check in provider_extra_checks() if check.name == "switch" for part in check.command
    )
    assert "action='search'" in command or "'action': 'search'" in command
    assert "wanted = found['title']" in command


def test_powershell_output_is_read_as_utf8(monkeypatch):
    """Every non-ASCII character came back as U+FFFD.

    PowerShell writes using the console's output encoding, which on this
    machine is a legacy code page; Python decodes as UTF-8. "Bück dich" was
    stored in a receipt as "B�ck dich" and then failed to match the title
    Windows had actually reported -- so a correct answer was recorded as a
    failed check. For a German user that is most track titles.
    """

    from tools import media_session

    captured = {}
    monkeypatch.setattr(media_session, "_powershell", lambda: "powershell")
    monkeypatch.setattr(
        media_session.subprocess, "run",
        lambda cmd, **kw: captured.update(argv=list(cmd), kwargs=kw) or type(
            "R", (), {"stdout": '{"ok": true}', "stderr": "", "returncode": 0})(),
    )

    media_session.read(app="spotify")

    from pathlib import Path

    script = Path(captured["argv"][captured["argv"].index("-File") + 1]).read_text(
        encoding="utf-8-sig"
    )
    assert "[Console]::OutputEncoding" in script, "stdout encoding must be pinned"
    assert "$OutputEncoding" in script, "the pipeline encoding must be pinned too"
    assert captured["kwargs"].get("encoding") == "utf-8"


def test_a_non_ascii_title_survives_the_round_trip():
    """Constructed rather than parsed: the failure was in decoding, so the test
    has to exercise the decoder with bytes that would break it."""

    from tools.media_session import _state

    state = _state({"ok": True, "app": "Spotify.exe", "title": "Bück dich",
                    "artist": "Rammstein", "status": "Playing"})

    assert state.title == "Bück dich"
    assert "�" not in state.describe()


# --------------------------------------------------------------------------
# A question is not a transport command, however short it is
# --------------------------------------------------------------------------
#
# Brevity was the whole guard, and the counter-example is four words long.
# "Wie geht es weiter?" is inside MAX_TRANSPORT_WORDS and is still a question.
# Observed live in the voice suite: it resumed playback, found no verified
# provider, and began acquiring a Spotify capability -- an autonomous build,
# started in answer to something nobody asked.


def test_a_four_word_question_is_not_a_resume():
    from service.music import understand

    assert understand("Wie geht es weiter?") is None
    assert understand("wie geht es weiter") is None


def test_the_bare_command_still_resumes():
    from service.music import understand

    assert understand("Weiter.").action == "resume"
    assert understand("weiter").action == "resume"


def test_a_question_about_something_else_entirely_is_left_alone():
    """"Naechste" in a question about steps is not a request for the next track."""

    from service.music import understand

    assert understand("Was ist der naechste Schritt?") is None


def test_a_question_that_names_music_is_still_a_music_request():
    """The interrogative rules out a bare verb, not the domain: someone asking
    "wie mache ich weiter mit der Musik" has said which weiter they mean."""

    from service.music import understand

    heard = understand("wie geht es weiter mit der musik")
    assert heard is not None and heard.action == "resume"


def test_asking_what_is_playing_still_works():
    """_CURRENT is matched before any transport verb, which is why asking a
    question about music is unaffected by a rule about questions."""

    from service.music import understand

    assert understand("Was laeuft gerade?").action == "current"
    assert understand("what's playing").action == "current"
