"""The machine is probed, not guessed at, and remembered between runs.

Every capability acquisition used to begin by rediscovering the same handful of
facts -- which Python, whether PowerShell exists, what Ollama has pulled, which
packages import. Four tool calls per INVESTIGATE, once per attempt, six
attempts for one capability.

The cost was the smaller problem. Four of six earlier music attempts failed on
environment facts nobody had told the model: packages that were not installed,
ZEUS tools that are not importable from generated source. A brief that is
*wrong* about the machine is worse than one that says nothing, and one written
by hand goes stale the moment anything is installed.

So: deterministic probes, a fingerprint over the things that would make the
answers wrong, and provenance on every fact -- because a fact whose provenance
is unrecorded is indistinguishable from one somebody made up.
"""

from __future__ import annotations

import json

import pytest

from runtime.environment import (
    FINGERPRINT_KEYS,
    PROBES,
    EnvironmentCache,
    Fact,
    _fingerprint_of,
)


@pytest.fixture
def cache(tmp_path):
    return EnvironmentCache(tmp_path / "environment.json")


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------

def test_every_fact_records_how_it_was_determined(cache):
    """A fact with no provenance cannot be told apart from one that was made up."""

    facts = cache.probe()

    assert facts
    for key, fact in facts.items():
        assert fact.probe, f"{key} does not say how it was established"
        assert fact.at, f"{key} does not say when"


def test_the_probes_find_the_things_a_capability_needs_to_know(cache):
    facts = cache.probe()

    for key in ("os", "python.executable", "executables.available", "packages.importable"):
        assert key in facts, f"{key} was not probed"


def test_absent_packages_are_recorded_as_well_as_present_ones(cache):
    """"You may import X" is only half the brief. The failure that actually
    happened was `from playsound import playsound` for a package nobody had
    said was missing."""

    facts = cache.probe()

    assert "packages.absent" in facts
    importable = set(facts["packages.importable"].value)
    absent = set(facts["packages.absent"].value)
    assert not (importable & absent), "a package cannot be both"


def test_a_probe_that_raises_becomes_a_fact_rather_than_an_exception(cache, monkeypatch):
    """A broken probe must not abort the other six."""

    def explode():
        raise RuntimeError("nvidia-smi fell over")

    monkeypatch.setitem(PROBES, "gpu", explode)

    facts = cache.probe()

    assert "probe.gpu.failed" in facts
    assert "nvidia-smi fell over" in str(facts["probe.gpu.failed"].value)
    assert "os" in facts, "the other probes must still have run"


# --------------------------------------------------------------------------
# Caching and invalidation
# --------------------------------------------------------------------------

def test_the_picture_survives_the_process(tmp_path):
    first = EnvironmentCache(tmp_path / "env.json")
    first.probe()

    second = EnvironmentCache(tmp_path / "env.json")
    second.load()

    assert second.get("os") == first.get("os")
    assert second.get("python.executable") == first.get("python.executable")


def test_a_changed_machine_invalidates_the_cache(tmp_path):
    """The fingerprint covers the things whose change makes everything else
    suspect. If a package appears, the cached brief is now wrong."""

    cache = EnvironmentCache(tmp_path / "env.json")
    before = cache.probe()

    stale = dict(before)
    stale["packages.importable"] = Fact(
        "packages.importable", ["something-that-was-not-there"], "test", "now"
    )

    assert _fingerprint_of(stale) != _fingerprint_of(before)


def test_the_fingerprint_ignores_things_that_change_harmlessly(tmp_path):
    cache = EnvironmentCache(tmp_path / "env.json")
    facts = cache.probe()

    moved = dict(facts)
    moved["gpu"] = Fact("gpu", "a different card", "test", "now")

    assert _fingerprint_of(moved) == _fingerprint_of(facts), (
        "a GPU swap does not invalidate which packages import"
    )


def test_the_fingerprint_keys_are_all_actually_probed(cache):
    """A fingerprint over a key nothing produces is a fingerprint over nothing."""

    facts = cache.probe()

    for key in FINGERPRINT_KEYS:
        assert key in facts, f"{key} is in the fingerprint but never probed"


def test_reading_a_fresh_cache_does_not_re_probe_everything(tmp_path, monkeypatch):
    cache = EnvironmentCache(tmp_path / "env.json")
    cache.probe()

    calls = []
    original = PROBES["applications"]
    monkeypatch.setitem(PROBES, "applications", lambda: calls.append(1) or original())

    cache.facts()

    assert calls == [], "an unchanged machine must not re-run the expensive probes"


def test_a_corrupt_cache_file_is_re_probed_rather_than_fatal(tmp_path):
    path = tmp_path / "env.json"
    path.write_text("{ not json", encoding="utf-8")
    cache = EnvironmentCache(path)

    facts = cache.facts()

    assert "os" in facts


# --------------------------------------------------------------------------
# The briefing a capability actually reads
# --------------------------------------------------------------------------

def test_the_briefing_names_what_cannot_be_imported(cache):
    cache.probe()

    briefing = " ".join(cache.briefing())

    assert "CANNOT be imported" in briefing


def test_every_briefing_line_carries_its_provenance(cache):
    cache.probe()

    for line in cache.briefing():
        assert line.rstrip().endswith("]"), f"no provenance on: {line}"


def test_the_briefing_is_serialisable(cache):
    cache.probe()

    assert json.dumps(cache.to_dict(), default=str)


def test_capability_briefs_include_the_environment(tmp_path):
    """The whole point: this reaches the thing being built."""

    import capabilities.service as service

    service._ENVIRONMENT = ["Local models available: x.  [GET /api/tags]"]
    try:
        assert service._environment_briefing() == ["Local models available: x.  [GET /api/tags]"]
    finally:
        service._ENVIRONMENT = None


def test_a_broken_environment_cache_leaves_the_brief_silent_not_wrong(tmp_path, monkeypatch):
    """A brief that is wrong about the machine is worse than one that says
    nothing -- so the fallback is nothing, never a guess."""

    import capabilities.service as service

    service._ENVIRONMENT = None
    monkeypatch.setattr(
        "runtime.environment.EnvironmentCache.briefing",
        lambda self: (_ for _ in ()).throw(RuntimeError("unreadable")),
    )
    try:
        assert service._environment_briefing() == []
    finally:
        service._ENVIRONMENT = None
