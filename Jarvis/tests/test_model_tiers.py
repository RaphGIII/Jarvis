from __future__ import annotations

import json

import pytest

from brain.tiers import (
    HealthState,
    ModelCatalog,
    ModelProbe,
    ModelSpec,
    ModelTier,
    default_catalog,
)


class StubProvider:
    """A provider whose three failure surfaces can be steered independently."""

    def __init__(self, *, models=None, list_error=None, generate_error=None, output="OK"):
        self._models = models if models is not None else ["qwen3:4b-instruct"]
        self._list_error = list_error
        self._generate_error = generate_error
        self._output = output
        self.generate_calls = 0

    def list_models(self):
        if self._list_error:
            raise self._list_error
        return list(self._models)

    def generate(self, prompt, *, max_tokens=8, temperature=0.0, top_p=None):
        self.generate_calls += 1
        if self._generate_error:
            raise self._generate_error
        return self._output


def _probe(spec_overrides=None, **provider_kwargs):
    catalog = ModelCatalog(specs=default_catalog())
    if spec_overrides:
        current = catalog.get(ModelTier.FAST_LOCAL)
        catalog.set(ModelTier.FAST_LOCAL, ModelSpec(**{**current.to_dict(), **spec_overrides, "tier": ModelTier.FAST_LOCAL}))
    provider = StubProvider(**provider_kwargs)
    return ModelProbe(catalog, ttl_seconds=0.0, provider_factory=lambda spec: provider), provider


# ------------------------------------------------------------------ catalog


def test_every_tier_has_a_spec():
    catalog = ModelCatalog(specs=default_catalog())
    assert set(catalog.tiers()) == set(ModelTier)


def test_paid_tiers_are_disabled_by_default():
    """Deleting cloud credentials must never break local development."""

    catalog = ModelCatalog(specs=default_catalog())
    assert catalog.paid_tiers_enabled() == []
    assert not catalog.get(ModelTier.EXPERT_CLOUD).enabled


def test_local_build_tier_is_enabled_by_default():
    catalog = ModelCatalog(specs=default_catalog())
    build = catalog.get(ModelTier.BUILD_LOCAL)
    assert build.enabled and build.configured and not build.paid


def test_env_layer_overrides_model_and_context():
    catalog = ModelCatalog(
        environ={"JARVIS_BUILD_LOCAL_MODEL": "some-30b", "JARVIS_BUILD_LOCAL_CONTEXT_WINDOW": "32768"}
    )
    spec = catalog.get(ModelTier.BUILD_LOCAL)
    assert spec.model == "some-30b"
    assert spec.context_window == 32768


def test_configuring_a_model_enables_a_disabled_tier():
    catalog = ModelCatalog(environ={"JARVIS_VISION_LOCAL_MODEL": "llava:7b"})
    assert catalog.get(ModelTier.VISION_LOCAL).enabled


def test_explicit_disable_beats_implicit_enable():
    catalog = ModelCatalog(
        environ={"JARVIS_VISION_LOCAL_MODEL": "llava:7b", "JARVIS_VISION_LOCAL_ENABLED": "0"}
    )
    assert not catalog.get(ModelTier.VISION_LOCAL).enabled


def test_file_layer_is_overridden_by_env_layer(tmp_path):
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps({"tiers": {"BUILD_LOCAL": {"model": "from-file", "context_window": 4096}}}),
        encoding="utf-8",
    )
    catalog = ModelCatalog(
        config_path=config,
        environ={"JARVIS_MODELS_CONFIG": str(config), "JARVIS_BUILD_LOCAL_MODEL": "from-env"},
    )
    spec = catalog.get(ModelTier.BUILD_LOCAL)
    assert spec.model == "from-env"
    assert spec.context_window == 4096  # file layer still applied where env is silent


def test_catalog_round_trips_through_save_and_load(tmp_path):
    config = tmp_path / "models.json"
    catalog = ModelCatalog(environ={})
    catalog.set(
        ModelTier.SELF_HOSTED,
        ModelSpec(tier=ModelTier.SELF_HOSTED, provider="openai_compatible", model="big", base_url="http://server:8000", enabled=True),
    )
    catalog.save(config)

    reloaded = ModelCatalog(config_path=config, environ={"JARVIS_MODELS_CONFIG": str(config)})
    spec = reloaded.get(ModelTier.SELF_HOSTED)
    assert spec.model == "big" and spec.base_url == "http://server:8000" and spec.enabled


def test_invalid_config_value_is_reported_not_swallowed(tmp_path):
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"tiers": {"BUILD_LOCAL": {"context_window": "not-a-number"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="context_window"):
        ModelCatalog(config_path=config, environ={"JARVIS_MODELS_CONFIG": str(config)})


# ------------------------------------------------------------------- health


def test_online_requires_a_real_generation():
    probe, provider = _probe()
    health = probe.probe(ModelTier.FAST_LOCAL)
    assert health.state is HealthState.ONLINE
    assert provider.generate_calls == 1, "a reachable server alone must not count as ONLINE"


def test_reachable_server_with_missing_model_is_not_online():
    """The exact failure the previous CLI reported as ONLINE."""

    probe, provider = _probe(models=["some-other-model"])
    health = probe.probe(ModelTier.FAST_LOCAL)
    assert health.state is HealthState.MODEL_MISSING
    assert not health.online
    assert provider.generate_calls == 0


def test_unreachable_provider_is_distinguished_from_missing_model():
    probe, _ = _probe(list_error=ConnectionError("refused"))
    assert probe.probe(ModelTier.FAST_LOCAL).state is HealthState.PROVIDER_UNREACHABLE


def test_failing_generation_is_reported_as_generation_failed():
    probe, _ = _probe(generate_error=RuntimeError("cuda error"))
    health = probe.probe(ModelTier.FAST_LOCAL)
    assert health.state is HealthState.GENERATION_FAILED
    assert "cuda error" in health.detail


def test_empty_completion_is_not_online():
    probe, _ = _probe(output="   ")
    assert probe.probe(ModelTier.FAST_LOCAL).state is HealthState.GENERATION_FAILED


def test_provider_without_model_listing_still_probes_generation():
    """An endpoint that cannot enumerate models is unknown, not empty."""

    probe, provider = _probe(models=[])
    assert probe.probe(ModelTier.FAST_LOCAL).state is HealthState.ONLINE
    assert provider.generate_calls == 1


def test_tag_variants_count_as_present():
    """``qwen3:latest`` on the server satisfies a ``qwen3:4b-instruct`` request.

    Ollama tags the same weights several ways; reporting MODEL_MISSING over a
    naming convention would be a different kind of dishonesty.
    """

    probe, provider = _probe(models=["qwen3:latest"])
    assert probe.catalog.get(ModelTier.FAST_LOCAL).model.startswith("qwen3:")
    assert probe.probe(ModelTier.FAST_LOCAL).state is HealthState.ONLINE
    assert provider.generate_calls == 1


def test_disabled_and_unconfigured_tiers_do_not_touch_the_network():
    catalog = ModelCatalog(specs=default_catalog())

    def explode(spec):
        raise AssertionError("a disabled tier must not build a provider")

    probe = ModelProbe(catalog, ttl_seconds=0.0, provider_factory=explode)
    assert probe.probe(ModelTier.EXPERT_CLOUD).state is HealthState.DISABLED
    assert probe.probe(ModelTier.VISION_LOCAL).state is HealthState.DISABLED


def test_probe_results_are_cached_until_invalidated():
    catalog = ModelCatalog(specs=default_catalog())
    provider = StubProvider()
    probe = ModelProbe(catalog, ttl_seconds=300.0, provider_factory=lambda spec: provider)

    probe.probe(ModelTier.FAST_LOCAL)
    probe.probe(ModelTier.FAST_LOCAL)
    assert provider.generate_calls == 1

    probe.invalidate(ModelTier.FAST_LOCAL)
    probe.probe(ModelTier.FAST_LOCAL)
    assert provider.generate_calls == 2
