"""A capability must not be handed out for a question it has nothing to do with.

This file exists because of the worst false positive in the project. A mission
to acquire `system.screen.capture` -- capture the screen, save a PNG -- reported
success in zero seconds having built nothing, because resolution matched the
goal to `music.provider.spotify`.

Measured at the time: 39 of the goal's 90 terms matched, score 0.43. The shared
terms were *accept, actually, cannot, checked, current, error, exists, failure,
file, match, must, name, never, not*. Every one generic English; not one about
music or screens. Both capabilities are described by their contracts -- payload
keys, return shapes, failure rules -- written in the same house style, so what
they shared was the vocabulary of contracts.

Term overlap was standing in for "is about the same thing". It stopped being
that as soon as descriptions got long, and nothing noticed because the score
went *up* with description length.

Two defences, and the tests below are mostly about the second one being real:
match on what a capability declares itself to be for, and when a caller names
the capability it wants, do not consult similarity at all.
"""

from __future__ import annotations

import pytest

from capabilities.models import CapabilityManifest
from capabilities.registry import CapabilityRegistry

#: A contract, in the house style every capability's description is written in.
CONTRACT = """
run(payload) must accept payload['action'] and return a dict with 'ok' (bool)
and 'error' (str when not ok). Read every payload value with .get(); never
index a payload key directly. Honour payload.get('dry_run'): perform every
check and report what it would do without doing it. Never raise for an
expected failure. It must ACTUALLY WORK -- a function that returns a
description of the thing happening, or reports success without anything
happening, is a failure. It is checked from outside afterwards.
"""

SCREEN_GOAL = (
    "Capture what is currently displayed on this computer's screen and save it "
    "as a PNG file. " + CONTRACT
)


def _registry(tmp_path, *manifests) -> CapabilityRegistry:
    registry = CapabilityRegistry(tmp_path / "registry.json")
    for manifest in manifests:
        registry.register(manifest)
    return registry


def _music(description: str = "", **overrides) -> CapabilityManifest:
    return CapabilityManifest(
        capability_id="music.provider.spotify",
        description=description or ("Build a spotify music provider for Windows. " + CONTRACT),
        creation_metadata={"keywords": ["spotify", "music", "musik", "song", "lied",
                                        "track", "play", "playback", "audio"]},
        **overrides,
    )


def _capture(**overrides) -> CapabilityManifest:
    return CapabilityManifest(
        capability_id="system.screen.capture",
        description="Capture the screen and save it as a PNG. " + CONTRACT,
        creation_metadata={"keywords": ["screenshot", "screen", "capture", "bildschirm",
                                        "display", "png", "image"]},
        **overrides,
    )


# --------------------------------------------------------------------------
# The false positive itself
# --------------------------------------------------------------------------

def test_a_music_provider_is_not_returned_for_a_screen_capture_goal(tmp_path):
    """The exact match that made a mission report success having built nothing."""

    registry = _registry(tmp_path, _music())

    assert registry.find(SCREEN_GOAL) == []


def test_two_capabilities_sharing_only_a_contract_do_not_match_each_other(tmp_path):
    """The generalised form: house style is not subject matter."""

    registry = _registry(tmp_path, _music(), _capture())

    for manifest in registry.find(SCREEN_GOAL):
        assert manifest.capability_id == "system.screen.capture"


def test_a_longer_description_does_not_make_a_capability_match_more(tmp_path):
    """The scoring got *better* the more boilerplate a description carried,
    which is the property that let this go unnoticed."""

    short = _registry(tmp_path / "a", _music(description="Plays music on Spotify."))
    verbose = _registry(tmp_path / "b", _music())

    assert short.find(SCREEN_GOAL) == []
    assert verbose.find(SCREEN_GOAL) == []


# --------------------------------------------------------------------------
# It must still find the right thing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "spiel mir Musik von den Beatles",
        "play a song on spotify",
        "put some music on",
        "naechstes Lied",
    ],
)
def test_a_music_request_still_finds_the_music_provider(tmp_path, query):
    registry = _registry(tmp_path, _music(), _capture())

    found = registry.find(query, limit=1)

    assert found and found[0].capability_id == "music.provider.spotify"


@pytest.mark.parametrize(
    "query",
    ["take a screenshot", "capture my screen", "mach ein Bildschirmfoto", "save the display as a png"],
)
def test_a_capture_request_finds_the_capture_capability(tmp_path, query):
    registry = _registry(tmp_path, _music(), _capture())

    found = registry.find(query, limit=1)

    assert found and found[0].capability_id == "system.screen.capture"


def test_a_disabled_capability_is_never_offered(tmp_path):
    registry = _registry(tmp_path, _music())
    registry.disable("music.provider.spotify", reason="broken")

    assert registry.find("play a song on spotify") == []


def test_naming_the_capability_in_the_query_still_wins(tmp_path):
    registry = _registry(tmp_path, _music(), _capture())

    found = registry.find("run music.provider.spotify please", limit=1)

    assert found and found[0].capability_id == "music.provider.spotify"


# --------------------------------------------------------------------------
# A named capability is looked up by name
# --------------------------------------------------------------------------

def test_ensure_with_a_name_never_returns_a_different_capability(tmp_path, monkeypatch):
    """Even if resolution were still loose, naming what you want has to win.

    Two independent defences, because the consequence of getting this wrong is
    a mission that reports success in zero seconds having built nothing.
    """

    from capabilities.service import CapabilityService

    registry = _registry(tmp_path, _music())
    service = CapabilityService.__new__(CapabilityService)
    service.registry = registry
    # Force resolution to be as wrong as it originally was.
    service.resolve = lambda goal: registry.get("music.provider.spotify")

    acquired: list[str] = []
    service.acquire = lambda goal, **kwargs: acquired.append(kwargs.get("capability_id")) or type(
        "O", (), {"usable": False, "capability_id": "", "reason": "not built", "status": "failed",
                  "verification": {}})()

    outcome = CapabilityService.ensure(service, SCREEN_GOAL, capability_id="system.screen.capture")

    assert outcome.capability_id != "music.provider.spotify"
    assert acquired == ["system.screen.capture"], "it must go and build the thing that was named"


def test_ensure_with_a_name_reuses_that_capability_when_it_is_installed(tmp_path):
    from capabilities.service import CapabilityService

    registry = _registry(tmp_path, _music(), _capture())
    service = CapabilityService.__new__(CapabilityService)
    service.registry = registry
    service.resolve = lambda goal: None

    outcome = CapabilityService.ensure(service, SCREEN_GOAL, capability_id="system.screen.capture")

    assert outcome.status == "available"
    assert outcome.capability_id == "system.screen.capture"


def test_ensure_with_a_name_rebuilds_when_that_capability_is_disabled(tmp_path):
    from capabilities.service import CapabilityService

    registry = _registry(tmp_path, _capture())
    registry.disable("system.screen.capture", reason="a defect")
    service = CapabilityService.__new__(CapabilityService)
    service.registry = registry
    service.resolve = lambda goal: None

    built: list[str] = []
    service.acquire = lambda goal, **kwargs: built.append(kwargs.get("capability_id")) or type(
        "O", (), {"usable": False, "capability_id": "", "reason": "", "status": "failed",
                  "verification": {}})()

    CapabilityService.ensure(service, SCREEN_GOAL, capability_id="system.screen.capture")

    assert built == ["system.screen.capture"]


# --------------------------------------------------------------------------
# What a capability is indexed under, when the goal that produced it was a repair
# --------------------------------------------------------------------------
#
# Keywords are derived from the goal. The goal that creates a capability
# describes the capability; the goal that produces version 1.0.4 of one is a
# repair brief, which opens with the defect -- deliberately, so the planner
# plans a repair rather than a rebuild. So `music.provider.spotify` came to be
# indexed under *defect, existing, implementation, rebuild, repair, working*,
# and with two capabilities installed it answered "rebuild the existing
# implementation because of a defect and repair the working code" with a music
# player. Measured on the live registry, 2026-08-26.


REPAIR_GOAL = (
    "Rebuild the existing implementation because of a defect: the working "
    "playback path ignores a track URI while something is already playing. "
    "Repair it so a requested song replaces what is playing."
)


def test_a_repair_goal_does_not_index_a_capability_under_the_repair():
    from capabilities.service import _keywords_from

    derived = set(_keywords_from(REPAIR_GOAL))

    assert not derived & {"rebuild", "existing", "implementation", "defect", "repair", "working"}


def test_a_repair_goal_still_yields_its_subject():
    from capabilities.service import _keywords_from

    derived = set(_keywords_from(REPAIR_GOAL))

    assert {"playback", "track", "song", "playing"} & derived


def test_build_and_capability_vocabulary_is_not_a_subject():
    """Every capability's goal contains these, so a match on them means nothing."""

    from capabilities.service import _keywords_from

    derived = set(_keywords_from(
        "Build a reusable capability that can create and replace things, version 2"
    ))

    assert not derived & {"build", "reusable", "capability", "create", "replace", "version"}


def test_process_vocabulary_resolves_to_nothing_at_all(tmp_path):
    """The end of the defect, at the level a caller sees it.

    The keywords here are the ones the live registry actually held for
    music.provider.spotify v1.0.4 -- derived from its repair goal, and the
    reason a request about rebuilding an implementation resolved to a music
    player. A manifest that was never repaired would not reproduce it.
    """

    polluted = _music()
    polluted.creation_metadata["keywords"] = [
        "abspielen", "action", "audio", "defect", "existing", "implementation",
        "lied", "music", "musik", "pause", "play", "playback", "rebuild",
        "repair", "skip", "song", "spielen", "spotify", "titel", "track", "working",
    ]
    registry = _registry(tmp_path, polluted, _capture())

    assert registry.find(
        "rebuild the existing implementation because of a defect and repair the working code"
    ) == []


def test_the_two_installed_capabilities_do_not_answer_for_each_other(tmp_path):
    registry = _registry(tmp_path, _music(), _capture())

    music = [m.capability_id for m in registry.find("play Bohemian Rhapsody on spotify")]
    screen = [m.capability_id for m in registry.find("take a screenshot of my screen")]

    assert music[:1] == ["music.provider.spotify"]
    assert screen[:1] == ["system.screen.capture"]
    assert "music.provider.spotify" not in screen
    assert "system.screen.capture" not in music


# --------------------------------------------------------------------------
# A capability that declares no keywords must still be findable
# --------------------------------------------------------------------------
#
# Matching on the identifier and the declared keywords killed the false
# positive, and introduced a false negative underneath it. An identifier can be
# a code name: `custom.scale` shares no word with "double an integer from the
# request payload". A capability with no keywords therefore became invisible to
# resolution the moment it was registered, and the caller rebuilt from scratch
# what it had just finished building.
#
# Caught by three v04 tests -- the second call re-acquiring instead of reusing
# is exactly what they assert. They passed at `b4b27e2` and failed at
# `e26f723`; the whole of that commit is the difference.


def _unlabelled(tmp_path):
    """A manifest as the v04 runtime registers them: an opaque id, no keywords."""

    return _registry(tmp_path, CapabilityManifest(
        capability_id="custom.scale",
        description="Double an integer x from the request payload. " + CONTRACT,
    ))


def test_a_capability_with_no_keywords_is_found_by_its_description(tmp_path):
    registry = _unlabelled(tmp_path)

    found = registry.find("Double an integer x from the request payload.")

    assert [m.capability_id for m in found] == ["custom.scale"]


def test_the_description_fallback_does_not_bring_the_false_positive_back(tmp_path):
    """The music provider and the screen-capture goal shared only contract
    vocabulary, and every word of it is removed before this comparison. So the
    fallback is safe precisely because the boilerplate filter came first."""

    registry = _registry(tmp_path, CapabilityManifest(
        capability_id="music.provider.spotify",
        description="Build a spotify music provider for Windows. " + CONTRACT,
    ))

    assert registry.find(SCREEN_GOAL) == []


def test_a_capability_that_declares_keywords_is_still_judged_on_them(tmp_path):
    """The fallback is for capabilities that say nothing. One that has said what
    it is for is taken at its word, and its prose cannot widen that."""

    registry = _registry(tmp_path, _music(
        description="Build a spotify music provider that captures a screen png. " + CONTRACT,
    ))

    assert registry.find(SCREEN_GOAL) == []


def test_an_unrelated_request_still_matches_nothing(tmp_path):
    registry = _unlabelled(tmp_path)

    assert registry.find("what is the weather tomorrow") == []
