from __future__ import annotations

import json
import subprocess

from brain.resources import (
    ContextBenchmark,
    ContextMeasurement,
    GpuInfo,
    GpuUsageMonitor,
    HostInfo,
    HostProbe,
    ResourcePolicy,
    ResourcePolicyStore,
    ResourceTuner,
)
from brain.tiers import ModelCatalog, ModelTier, default_catalog


def _completed(stdout: str, returncode: int = 0):
    return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr="")


def _host(total=8192, used=610, free=7582, gpus=1):
    return HostInfo(
        platform="Windows 10",
        cpu_count=8,
        total_ram_mib=16384,
        gpus=[GpuInfo(name="NVIDIA GeForce GTX 1070", total_mib=total, used_mib=used, free_mib=free)] * gpus,
    )


class StubProbe(HostProbe):
    def __init__(self, host: HostInfo):
        self._host = host

    def detect(self):
        return self._host

    def detect_gpus(self):
        return list(self._host.gpus)


class StubBenchmark(ContextBenchmark):
    """Replays a scripted sweep instead of running real generations."""

    def __init__(self, results: dict[int, ContextMeasurement]):
        self._results = results
        self.measured: list[int] = []

    def measure(self, spec, context_window, *, max_tokens=96):
        self.measured.append(int(context_window))
        return self._results.get(
            int(context_window),
            ContextMeasurement(context_window=int(context_window), ok=False, error="out of memory"),
        )


# ------------------------------------------------------------------ host


def test_gpu_inventory_is_parsed_from_nvidia_smi():
    probe = HostProbe(runner=lambda *a, **k: _completed("NVIDIA GeForce GTX 1070, 8192, 610, 7582\n"))
    gpus = probe.detect_gpus()
    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce GTX 1070"
    assert gpus[0].total_mib == 8192 and gpus[0].free_mib == 7582


def test_missing_or_failing_nvidia_smi_is_not_fatal():
    probe = HostProbe(runner=lambda *a, **k: _completed("", returncode=9))
    assert probe.detect_gpus() == []


def test_detect_returns_a_usable_host_record_on_this_machine():
    info = HostProbe().detect()
    assert info.platform
    assert info.cpu_count >= 1


def test_gpu_utilisation_is_read_alongside_memory():
    probe = HostProbe(runner=lambda *a, **k: _completed("NVIDIA GeForce GTX 1070, 8192, 610, 7582, 37\n"))
    gpu = probe.detect_gpus()[0]
    assert gpu.utilization_percent == 37
    assert gpu.free_mib == 7582  # the memory reading is unchanged


def test_an_unreadable_utilisation_does_not_discard_the_memory_reading():
    """Some drivers report ``[N/A]``; that is not a reason to report no GPU."""

    probe = HostProbe(runner=lambda *a, **k: _completed("NVIDIA GeForce GTX 1070, 8192, 610, 7582, [N/A]\n"))
    gpus = probe.detect_gpus()
    assert len(gpus) == 1
    assert gpus[0].utilization_percent == 0
    assert gpus[0].total_mib == 8192


# ------------------------------------------------------------------ gpu load


def test_gpu_usage_reports_load_and_memory():
    monitor = GpuUsageMonitor(host_probe=StubProbe(_host(total=8192, used=4096, free=4096)))
    monitor.host_probe._host.gpus[0].utilization_percent = 62
    sample = monitor.refresh()
    assert sample["available"] is True
    assert sample["utilization_percent"] == 62
    assert sample["memory_percent"] == 50


def test_a_host_with_no_gpu_reports_nothing_to_show():
    """Not an error: the interface hides the readout rather than showing 0%."""

    monitor = GpuUsageMonitor(host_probe=StubProbe(HostInfo()))
    sample = monitor.refresh()
    assert sample["measured"] is True
    assert sample["available"] is False


def test_gpu_usage_never_blocks_before_a_reading_exists():
    """snapshot() answers from cache, so it can sit on the request path."""

    monitor = GpuUsageMonitor(host_probe=StubProbe(_host()))
    assert monitor.snapshot() == {"measured": False, "available": False}


# ------------------------------------------------------------------ policy


def test_default_policy_is_conservative_on_an_8gb_card():
    policy = ResourcePolicy.default_for(_host())
    assert policy.max_concurrent_generations == 1
    assert policy.context_windows[ModelTier.BUILD_LOCAL.value] <= 12288
    assert policy.reserved_vram_mib >= 1024


def test_vram_reserve_scales_with_the_card():
    from brain.resources import reserved_vram_mib

    small = reserved_vram_mib(_host(total=8192))
    large = reserved_vram_mib(_host(total=49152, used=0, free=49152))
    assert small >= 1500, "an 8 GiB card must keep more than a token allowance for the desktop"
    assert large > small
    assert reserved_vram_mib(HostInfo()) == 0  # CPU-only host reserves nothing


def test_throughput_collapse_is_rejected_even_above_the_absolute_floor():
    """The real GTX 1070 case: 56 tok/s up to 24k, 31 tok/s at 32k.

    31 tok/s clears any sane absolute floor, but losing 45% of throughput is
    memory pressure, and paying for it in desktop responsiveness to gain
    nominal context is the wrong trade.
    """

    benchmark = StubBenchmark(
        {
            16384: _measurement(16384, 56.3, 2581),
            24576: _measurement(24576, 56.7, 2581),
            32768: _measurement(32768, 31.2, 2581),
        }
    )
    tuner = ResourceTuner(ModelCatalog(specs=default_catalog()), benchmark=benchmark, host_probe=StubProbe(_host()))
    policy = tuner.tune(tiers=[ModelTier.BUILD_LOCAL], candidates=[16384, 24576, 32768])
    assert policy.context_windows[ModelTier.BUILD_LOCAL.value] == 24576
    assert any("rejected 32768" in note for note in policy.notes)


def test_uniform_throughput_lets_the_largest_context_win():
    """No collapse means no reason to leave context on the table."""

    benchmark = StubBenchmark(
        {
            8192: _measurement(8192, 32.4, 4000),
            16384: _measurement(16384, 32.7, 3500),
            24576: _measurement(24576, 32.6, 3000),
        }
    )
    tuner = ResourceTuner(ModelCatalog(specs=default_catalog()), benchmark=benchmark, host_probe=StubProbe(_host()))
    policy = tuner.tune(tiers=[ModelTier.BUILD_LOCAL], candidates=[8192, 16384, 24576])
    assert policy.context_windows[ModelTier.BUILD_LOCAL.value] == 24576


def test_default_policy_scales_up_with_vram():
    big = ResourcePolicy.default_for(_host(total=49152, used=0, free=49152))
    assert big.context_windows[ModelTier.BUILD_LOCAL.value] >= 32768


def test_policy_applies_to_a_catalog():
    catalog = ModelCatalog(specs=default_catalog())
    policy = ResourcePolicy(
        context_windows={ModelTier.BUILD_LOCAL.value: 6144},
        keep_alive={ModelTier.BUILD_LOCAL.value: "30m"},
    )
    policy.apply_to(catalog)
    spec = catalog.get(ModelTier.BUILD_LOCAL)
    assert spec.context_window == 6144
    assert spec.keep_alive == "30m"


def test_policy_round_trips_through_the_store(tmp_path):
    store = ResourcePolicyStore(tmp_path / "resources.json")
    policy = ResourcePolicy(max_concurrent_generations=1, context_windows={"BUILD_LOCAL": 8192}, notes=["measured"])
    store.save(policy)
    reloaded = store.load()
    assert reloaded is not None
    assert reloaded.context_windows["BUILD_LOCAL"] == 8192
    assert reloaded.notes == ["measured"]


def test_corrupt_policy_file_falls_back_instead_of_crashing(tmp_path):
    path = tmp_path / "resources.json"
    path.write_text("{not json", encoding="utf-8")
    store = ResourcePolicyStore(path)
    assert store.load() is None
    assert store.load_or_default(_host()).context_windows


# ------------------------------------------------------------------ tuning


def _measurement(window, tok_s, free_mib, ok=True):
    return ContextMeasurement(
        context_window=window, ok=ok, tokens_per_second=tok_s, vram_free_mib=free_mib, generated_tokens=96
    )


def test_tuner_picks_the_largest_context_that_stays_fast_and_leaves_headroom():
    benchmark = StubBenchmark(
        {
            4096: _measurement(4096, 30.0, 4000),
            8192: _measurement(8192, 29.0, 2500),
            12288: _measurement(12288, 28.0, 2000),
            16384: _measurement(16384, 27.0, 300),  # too little headroom for the desktop
        }
    )
    tuner = ResourceTuner(ModelCatalog(specs=default_catalog()), benchmark=benchmark, host_probe=StubProbe(_host()))
    policy = tuner.tune(tiers=[ModelTier.BUILD_LOCAL], candidates=[4096, 8192, 12288, 16384])
    assert policy.context_windows[ModelTier.BUILD_LOCAL.value] == 12288


def test_tuner_rejects_a_context_that_is_too_slow_even_if_it_fits():
    benchmark = StubBenchmark(
        {
            4096: _measurement(4096, 30.0, 4000),
            8192: _measurement(8192, 1.5, 3000),  # spilling to CPU
        }
    )
    tuner = ResourceTuner(ModelCatalog(specs=default_catalog()), benchmark=benchmark, host_probe=StubProbe(_host()))
    policy = tuner.tune(tiers=[ModelTier.BUILD_LOCAL], candidates=[4096, 8192], min_tokens_per_second=3.0)
    assert policy.context_windows[ModelTier.BUILD_LOCAL.value] == 4096


def test_sweep_stops_after_the_first_failure():
    """Each candidate costs a model load; there is no point probing past OOM."""

    benchmark = StubBenchmark({4096: _measurement(4096, 30.0, 4000)})
    tuner = ResourceTuner(ModelCatalog(specs=default_catalog()), benchmark=benchmark, host_probe=StubProbe(_host()))
    tuner.tune(tiers=[ModelTier.BUILD_LOCAL], candidates=[4096, 8192, 16384, 32768])
    assert benchmark.measured == [4096, 8192]


def test_tuner_falls_back_when_nothing_completes():
    benchmark = StubBenchmark({})
    tuner = ResourceTuner(ModelCatalog(specs=default_catalog()), benchmark=benchmark, host_probe=StubProbe(_host()))
    policy = tuner.tune(tiers=[ModelTier.BUILD_LOCAL], candidates=[8192])
    assert policy.context_windows[ModelTier.BUILD_LOCAL.value] > 0
    assert any("no context size completed" in note for note in policy.notes)


def test_single_consumer_gpu_stays_at_one_concurrent_generation():
    tuner = ResourceTuner(ModelCatalog(specs=default_catalog()), benchmark=StubBenchmark({}), host_probe=StubProbe(_host()))
    policy = tuner.tune(tiers=[], candidates=[4096])
    assert policy.max_concurrent_generations == 1


def test_large_gpu_allows_more_concurrency():
    host = _host(total=49152, used=0, free=49152)
    tuner = ResourceTuner(ModelCatalog(specs=default_catalog()), benchmark=StubBenchmark({}), host_probe=StubProbe(host))
    assert tuner.tune(tiers=[], candidates=[4096]).max_concurrent_generations == 2


def test_disabled_tiers_are_skipped_and_reported():
    tuner = ResourceTuner(ModelCatalog(specs=default_catalog()), benchmark=StubBenchmark({}), host_probe=StubProbe(_host()))
    policy = tuner.tune(tiers=[ModelTier.VISION_LOCAL], candidates=[4096])
    assert any("VISION_LOCAL: skipped" in note for note in policy.notes)


def test_tuned_policy_is_json_serialisable(tmp_path):
    benchmark = StubBenchmark({4096: _measurement(4096, 30.0, 4000)})
    tuner = ResourceTuner(ModelCatalog(specs=default_catalog()), benchmark=benchmark, host_probe=StubProbe(_host()))
    policy = tuner.tune(tiers=[ModelTier.BUILD_LOCAL], candidates=[4096])
    assert json.loads(json.dumps(policy.to_dict()))["measurements"]["BUILD_LOCAL"]
