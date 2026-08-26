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
