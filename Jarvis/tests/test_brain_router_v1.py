"""Router contract.

Rewritten when the heavy tier moved from BUILD_REMOTE to BUILD_LOCAL.  The
invariants the earlier version protected are all still asserted here -- chat
goes to the fast model, development work goes to the heavy one, and a heavy
request never silently falls through to a different tier -- but the heavy tier
is now the local coder model, so Jarvis works with no remote compute at all.
"""

from __future__ import annotations

import pytest

from brain.router import BrainRouter, BrainTier, BrainUnavailable, RemoteBrainUnavailable
from brain.tiers import HealthState, ModelCatalog, ModelHealth, ModelProbe, ModelTier, default_catalog


class FakeBrain:
    """A provider that reports whatever model the spec asked for.

    Reporting a fixed name instead would make every probe fail with
    MODEL_MISSING before it ever reached the generation step, which hides the
    behaviour these tests are actually about.
    """

    provider_name = "fake"

    def __init__(self, spec=None):
        self.model_name = getattr(spec, "model", "fake-model")
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return f"ANSWER:{prompt}"

    def list_models(self):
        return [self.model_name]


def _router(*, provider_factory=None, **kwargs):
    catalog = ModelCatalog(specs=default_catalog())
    factory = provider_factory or (lambda spec: FakeBrain(spec))
    return BrainRouter(catalog=catalog, probe=ModelProbe(catalog, ttl_seconds=0.0, provider_factory=factory), **kwargs)


# ------------------------------------------------------------------ routing


@pytest.mark.parametrize(
    "message",
    [
        "Erkläre mir die Photosynthese.",
        "Hallo Jarvis, wie geht es dir?",
        "What is the capital of France?",
    ],
)
def test_conversation_routes_to_the_fast_local_model(message):
    assert _router().route(message).tier is BrainTier.FAST_LOCAL


@pytest.mark.parametrize(
    "message",
    [
        "Implementiere ein neues Feature in diesem Repository.",
        "Fix the bug in the parser.",
        "Build me a small script that renames files.",
        "Refactor this module.",
    ],
)
def test_development_work_routes_to_the_local_build_model(message):
    decision = _router().route(message)
    assert decision.tier is BrainTier.BUILD_LOCAL
    assert decision.is_project, "building software is a project, not a chat turn"


def test_capability_requests_route_to_the_build_tier_as_projects():
    decision = _router().route("Lerne, wie man Musik abspielt.")
    assert decision.tier is BrainTier.BUILD_LOCAL
    assert decision.is_project


def test_the_default_heavy_tier_is_local_and_free():
    """The point of the change: no credential, no remote host, still works."""

    catalog = ModelCatalog(specs=default_catalog())
    spec = catalog.get(BrainTier.BUILD_LOCAL.to_model_tier())
    assert spec.enabled and not spec.paid
    assert "127.0.0.1" in spec.base_url or "localhost" in spec.base_url


# ------------------------------------------------------------------ dispatch


def test_fast_requests_are_answered_by_the_fast_brain():
    fast = FakeBrain()
    router = _router(fast_brain=fast)
    answer, decision = router.respond("Hallo Jarvis")
    assert answer == "ANSWER:Hallo Jarvis"
    assert decision.tier is BrainTier.FAST_LOCAL
    assert len(fast.calls) == 1


def test_a_development_request_is_never_answered_as_chat():
    """It must become a project, not an essay about what Jarvis would do."""

    fast, build = FakeBrain(), FakeBrain()
    router = _router(fast_brain=fast, build_brain=build)

    with pytest.raises(BrainUnavailable):
        router.respond("Implementiere ein neues Feature in diesem Repository.")

    assert fast.calls == []
    assert build.calls == []


def test_injected_build_brain_is_used():
    build = FakeBrain()
    assert _router(build_brain=build).brain(BrainTier.BUILD_LOCAL) is build


def test_a_disabled_tier_is_refused_with_an_explanation():
    catalog = ModelCatalog(environ={"JARVIS_BUILD_LOCAL_ENABLED": "0"})
    router = BrainRouter(catalog=catalog, probe=ModelProbe(catalog, ttl_seconds=0.0, provider_factory=lambda spec: FakeBrain(spec)))
    with pytest.raises(BrainUnavailable, match="disabled"):
        router.brain(BrainTier.BUILD_LOCAL)


def test_the_old_exception_name_still_works():
    """Existing callers catch RemoteBrainUnavailable; keep that working."""

    assert RemoteBrainUnavailable is BrainUnavailable


# ------------------------------------------------------------------ honesty


def test_require_build_brain_refuses_a_tier_that_cannot_generate():
    """A tier is only usable once a real generation has succeeded."""

    class BrokenProvider(FakeBrain):
        def generate(self, prompt, **kwargs):
            raise RuntimeError("cuda out of memory")

    router = _router(provider_factory=lambda spec: BrokenProvider(spec))
    with pytest.raises(BrainUnavailable, match="GENERATION_FAILED"):
        router.require_build_brain()


def test_require_build_brain_refuses_a_missing_model():
    class NoModelProvider(FakeBrain):
        def list_models(self):
            return ["some-completely-different-model"]

    router = _router(provider_factory=lambda spec: NoModelProvider(spec))
    with pytest.raises(BrainUnavailable, match="MODEL_MISSING"):
        router.require_build_brain()


def test_status_reports_per_tier_health_and_paid_tiers():
    status = _router().status(force=True)
    assert status["build_tier"] == "BUILD_LOCAL"
    assert status["build_online"] is True
    assert status["paid_tiers_enabled"] == [], "no paid tier may be on by default"
    assert set(status["tiers"]) == {tier.value for tier in ModelTier}
