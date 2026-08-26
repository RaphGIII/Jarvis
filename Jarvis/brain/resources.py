"""Measure the host, then choose inference settings that fit it.

The mission constraint is blunt: the computer has to stay usable while Jarvis
works.  On a GTX 1070 that is decided almost entirely by two numbers -- how much
VRAM the model plus its KV cache occupy, and how many generations run at once.
Guessing them from a spec sheet does not work, because whether a given
``num_ctx`` spills to system RAM depends on the quantisation, the driver, and
whatever else already holds VRAM (a browser, a game, the compositor).

So this module measures instead of assuming:

* :class:`HostProbe` reads the actual GPU inventory and free VRAM.
* :class:`ContextBenchmark` runs real generations at candidate context sizes and
  records throughput, latency and VRAM headroom for each.
* :class:`ResourcePolicy` turns those measurements into the settings the tiers
  actually run with, and is what gets persisted.

The output is deliberately conservative.  A configuration that is 15% faster but
leaves 200 MB of VRAM headroom is the wrong trade for a machine someone is also
using.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from brain.tiers import ModelCatalog, ModelSpec, ModelTier


# --------------------------------------------------------------------------
# Host inventory
# --------------------------------------------------------------------------

def _percent(raw: Any) -> int:
    """A 0-100 integer, or 0 for anything unreadable."""

    try:
        return max(0, min(100, int(float(str(raw).strip().rstrip("%")))))
    except (TypeError, ValueError):
        return 0


@dataclass
class GpuInfo:
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    #: How busy the card's compute units are right now, 0-100.  Defaults to 0
    #: so an older reading -- or a driver that reports ``[N/A]`` -- still
    #: produces a usable record rather than none.
    utilization_percent: int = 0

    @property
    def free_fraction(self) -> float:
        return self.free_mib / self.total_mib if self.total_mib else 0.0

    @property
    def used_fraction(self) -> float:
        return self.used_mib / self.total_mib if self.total_mib else 0.0


@dataclass
class HostInfo:
    platform: str = ""
    cpu_count: int = 0
    total_ram_mib: int = 0
    gpus: list[GpuInfo] = field(default_factory=list)
    detected_at: str = ""

    @property
    def primary_gpu(self) -> GpuInfo | None:
        return self.gpus[0] if self.gpus else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "total_ram_mib": self.total_ram_mib,
            "gpus": [asdict(gpu) for gpu in self.gpus],
            "detected_at": self.detected_at,
        }


class HostProbe:
    """Reads what hardware is actually present, degrading gracefully."""

    def __init__(self, *, runner: Callable[..., subprocess.CompletedProcess] | None = None) -> None:
        self._run = runner or subprocess.run

    def detect(self) -> HostInfo:
        return HostInfo(
            platform=f"{platform.system()} {platform.release()}",
            cpu_count=os.cpu_count() or 0,
            total_ram_mib=self._total_ram_mib(),
            gpus=self.detect_gpus(),
            detected_at=datetime.now(timezone.utc).isoformat(),
        )

    def detect_gpus(self) -> list[GpuInfo]:
        if not shutil.which("nvidia-smi"):
            return []
        try:
            completed = self._run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if completed.returncode != 0:
            return []

        gpus: list[GpuInfo] = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            # Four columns is the memory-only form this probe used to ask for.
            # Accepting it too means a stubbed runner, a recorded output or an
            # older driver that drops the utilisation column still parses.
            if len(parts) not in (4, 5):
                continue
            try:
                memory = (int(parts[1]), int(parts[2]), int(parts[3]))
            except ValueError:
                continue
            gpus.append(
                GpuInfo(
                    name=parts[0],
                    total_mib=memory[0],
                    used_mib=memory[1],
                    free_mib=memory[2],
                    # Some drivers report "[N/A]" here; an unreadable load is
                    # not a reason to discard a perfectly good memory reading.
                    utilization_percent=_percent(parts[4]) if len(parts) == 5 else 0,
                )
            )
        return gpus

    def _total_ram_mib(self) -> int:
        try:
            if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
                return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
        except (OSError, ValueError):
            pass
        if platform.system() == "Windows":
            try:
                import ctypes

                class _MemoryStatusEx(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                status = _MemoryStatusEx()
                status.dwLength = ctypes.sizeof(_MemoryStatusEx)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    return int(status.ullTotalPhys / (1024 * 1024))
            except Exception:
                return 0
        return 0


class GpuUsageMonitor:
    """What the GPU is doing *right now*, cheap enough to put on a UI poll.

    Every reading costs an ``nvidia-smi`` launch -- tens of milliseconds of a
    process spawn, not a generation, but the interface asks every few seconds
    and several clients may be watching at once.  So a reading is taken on a
    background thread and cached, and :meth:`snapshot` returns the last one it
    got.  That keeps two properties that matter on this machine: the request
    path never waits on a subprocess, and N watchers cost the same as one.

    A host with no NVIDIA card is the ordinary case, not an error: the probe
    returns nothing, ``available`` stays false, and the caller shows nothing.
    """

    #: How stale a reading may be.  Short enough that the number tracks a
    #: generation starting, long enough that the probe is not the load.
    DEFAULT_TTL_SECONDS = 3.0

    def __init__(self, *, host_probe: HostProbe | None = None, ttl_seconds: float | None = None) -> None:
        self.host_probe = host_probe or HostProbe()
        self.ttl_seconds = float(self.DEFAULT_TTL_SECONDS if ttl_seconds is None else ttl_seconds)
        self._sample: dict[str, Any] = {"measured": False, "available": False}
        self._sampled_at = 0.0
        self._running = threading.Event()

    def snapshot(self) -> dict[str, Any]:
        """The last reading, refreshing in the background when it is stale."""

        # Read first, then decide whether to refresh: the caller is promised the
        # reading that existed when it asked, not one that may or may not have
        # landed by the time this returns.
        sample = dict(self._sample)
        if time.time() - self._sampled_at >= self.ttl_seconds and not self._running.is_set():
            self._running.set()
            threading.Thread(target=self._refresh, daemon=True, name="jarvis-gpu-usage").start()
        return sample

    def refresh(self) -> dict[str, Any]:
        """Take a reading now, on this thread.  For callers that can wait."""

        self._refresh()
        return dict(self._sample)

    def _refresh(self) -> None:
        try:
            gpus = self.host_probe.detect_gpus()
        except Exception:
            gpus = []
        try:
            self._sample = self._describe(gpus)
        finally:
            self._sampled_at = time.time()
            self._running.clear()

    @staticmethod
    def _describe(gpus: list[GpuInfo]) -> dict[str, Any]:
        if not gpus:
            # "Measured, and there is no NVIDIA GPU here" -- distinct from the
            # not-yet-measured state, so a client can tell them apart.
            return {"measured": True, "available": False}
        gpu = gpus[0]
        return {
            "measured": True,
            "available": True,
            "name": gpu.name,
            "utilization_percent": _percent(gpu.utilization_percent),
            "memory_percent": round(gpu.used_fraction * 100),
            "memory_used_mib": gpu.used_mib,
            "memory_total_mib": gpu.total_mib,
            "gpus": len(gpus),
        }


# --------------------------------------------------------------------------
# Benchmarking
# --------------------------------------------------------------------------

@dataclass
class ContextMeasurement:
    """One real generation at one candidate context size."""

    context_window: int
    ok: bool
    tokens_per_second: float = 0.0
    load_seconds: float = 0.0
    total_seconds: float = 0.0
    generated_tokens: int = 0
    vram_used_mib: int = 0
    vram_free_mib: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: The prompt is long enough that the KV cache is genuinely exercised, and
#: mechanical enough that any model can answer it, so the measurement reflects
#: throughput rather than reasoning difficulty.
_BENCHMARK_PROMPT = (
    "You are benchmarking a local inference server. "
    "Write a short Python function called `slugify` that lowercases a string, "
    "replaces every run of non-alphanumeric characters with a single hyphen, "
    "and strips leading and trailing hyphens. Return only the code."
)


class ContextBenchmark:
    """Runs real generations at candidate context sizes and records the cost."""

    def __init__(
        self,
        *,
        host_probe: HostProbe | None = None,
        provider_factory: Callable[[ModelSpec], Any] | None = None,
    ) -> None:
        self.host_probe = host_probe or HostProbe()
        if provider_factory is None:
            from brain.providers import provider_for_spec as provider_factory
        self._provider_factory = provider_factory

    def measure(self, spec: ModelSpec, context_window: int, *, max_tokens: int = 96) -> ContextMeasurement:
        from dataclasses import replace as _replace

        candidate = _replace(spec, context_window=int(context_window))
        started = time.perf_counter()
        try:
            provider = self._provider_factory(candidate)
            text = provider.generate(_BENCHMARK_PROMPT, max_tokens=max_tokens, temperature=0.0)
        except Exception as exc:
            return ContextMeasurement(
                context_window=int(context_window),
                ok=False,
                total_seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}"[:400],
            )

        total = time.perf_counter() - started
        metadata = dict(getattr(provider, "last_metadata", {}) or {})
        gpu = (self.host_probe.detect_gpus() or [None])[0]

        generated = int(metadata.get("generated_tokens") or 0)
        eval_seconds = float(metadata.get("eval_seconds") or 0.0)
        throughput = float(metadata.get("tokens_per_second") or 0.0)
        if not throughput and generated and eval_seconds:
            throughput = generated / eval_seconds

        return ContextMeasurement(
            context_window=int(context_window),
            ok=bool(str(text).strip()),
            tokens_per_second=round(throughput, 2),
            load_seconds=float(metadata.get("load_seconds") or 0.0),
            total_seconds=round(total, 2),
            generated_tokens=generated,
            vram_used_mib=gpu.used_mib if gpu else 0,
            vram_free_mib=gpu.free_mib if gpu else 0,
            error="" if str(text).strip() else "empty completion",
        )

    def sweep(self, spec: ModelSpec, candidates: list[int], *, max_tokens: int = 96) -> list[ContextMeasurement]:
        """Measure each candidate, smallest first, stopping after a failure.

        Ascending order matters: once a context size fails to load, every larger
        one will too, and each attempt costs a model load.  Stopping early keeps
        a tuning run to a couple of minutes instead of ten.
        """

        measurements: list[ContextMeasurement] = []
        for candidate in sorted(set(int(item) for item in candidates)):
            measurement = self.measure(spec, candidate, max_tokens=max_tokens)
            measurements.append(measurement)
            if not measurement.ok:
                break
        return measurements


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------

def reserved_vram_mib(host: HostInfo) -> int:
    """VRAM Jarvis refuses to consume, so the desktop stays usable.

    A flat allowance does not work across cards: 1 GiB is generous next to a
    48 GiB server card and far too thin next to an 8 GiB consumer one, where
    the compositor and a browser alone can want more than that.  Scaling with
    the card keeps the trade-off honest at both ends.
    """

    gpu = host.primary_gpu
    if not gpu or not gpu.total_mib:
        return 0
    return max(768, min(4096, int(gpu.total_mib * 0.20)))


@dataclass
class ResourcePolicy:
    """The settings Jarvis will actually run with, plus why.

    Persisted so a tuning run happens once rather than on every start, and so
    the report can quote real measured numbers.
    """

    #: Concurrent generations across the whole system.
    max_concurrent_generations: int = 1
    #: Tier -> chosen context window.
    context_windows: dict[str, int] = field(default_factory=dict)
    #: Tier -> keep_alive string.
    keep_alive: dict[str, str] = field(default_factory=dict)
    #: VRAM we refuse to consume, so the desktop keeps working.
    reserved_vram_mib: int = 1024
    host: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    tuned_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourcePolicy":
        return cls(
            max_concurrent_generations=int(data.get("max_concurrent_generations", 1)),
            context_windows={str(k): int(v) for k, v in (data.get("context_windows") or {}).items()},
            keep_alive={str(k): str(v) for k, v in (data.get("keep_alive") or {}).items()},
            reserved_vram_mib=int(data.get("reserved_vram_mib", 1024)),
            host=dict(data.get("host") or {}),
            measurements={str(k): list(v) for k, v in (data.get("measurements") or {}).items()},
            notes=[str(item) for item in data.get("notes") or []],
            tuned_at=str(data.get("tuned_at", "")),
        )

    def apply_to(self, catalog: ModelCatalog) -> ModelCatalog:
        """Push the tuned numbers into a catalog's specs."""

        from dataclasses import replace as _replace

        for tier_name, window in self.context_windows.items():
            try:
                tier = ModelTier(tier_name)
            except ValueError:
                continue
            spec = catalog.get(tier)
            catalog.set(
                tier,
                _replace(spec, context_window=int(window), keep_alive=self.keep_alive.get(tier_name, spec.keep_alive)),
            )
        return catalog

    @classmethod
    def default_for(cls, host: HostInfo) -> "ResourcePolicy":
        """A safe untuned policy, derived from VRAM alone.

        Used before a benchmark has ever run so the system is conservative by
        default rather than optimistic by default.
        """

        gpu = host.primary_gpu
        vram = gpu.total_mib if gpu else 0
        if vram >= 40000:
            fast, build = 32768, 32768
        elif vram >= 20000:
            fast, build = 16384, 16384
        elif vram >= 10000:
            fast, build = 8192, 12288
        elif vram >= 6000:
            fast, build = 8192, 8192
        else:
            fast, build = 4096, 4096

        return cls(
            max_concurrent_generations=1,
            context_windows={ModelTier.FAST_LOCAL.value: fast, ModelTier.BUILD_LOCAL.value: build},
            keep_alive={ModelTier.FAST_LOCAL.value: "10m", ModelTier.BUILD_LOCAL.value: "15m"},
            reserved_vram_mib=reserved_vram_mib(host),
            host=host.to_dict(),
            notes=["untuned default derived from VRAM size; run the tuner for measured values"],
            tuned_at="",
        )


class ResourceTuner:
    """Benchmarks the host and produces a :class:`ResourcePolicy`."""

    #: Candidates are powers of two spanning "definitely fits" to "probably
    #: does not", so the sweep brackets the real limit rather than confirming a
    #: guess.
    DEFAULT_CANDIDATES = (4096, 8192, 12288, 16384, 24576, 32768)

    #: A candidate must retain at least this fraction of the tier's best
    #: measured throughput.  0.85 tolerates ordinary run-to-run noise while
    #: still catching the sharp cliff that memory pressure produces.
    DEFAULT_THROUGHPUT_RETENTION = 0.85

    def __init__(
        self,
        catalog: ModelCatalog,
        *,
        benchmark: ContextBenchmark | None = None,
        host_probe: HostProbe | None = None,
        throughput_retention: float | None = None,
    ) -> None:
        self.catalog = catalog
        self.host_probe = host_probe or HostProbe()
        self.benchmark = benchmark or ContextBenchmark(host_probe=self.host_probe)
        self.throughput_retention = (
            self.DEFAULT_THROUGHPUT_RETENTION if throughput_retention is None else float(throughput_retention)
        )

    def tune(
        self,
        *,
        tiers: list[ModelTier] | None = None,
        candidates: list[int] | None = None,
        min_tokens_per_second: float = 3.0,
    ) -> ResourcePolicy:
        host = self.host_probe.detect()
        policy = ResourcePolicy.default_for(host)
        policy.host = host.to_dict()
        policy.notes = []
        policy.measurements = {}
        policy.tuned_at = datetime.now(timezone.utc).isoformat()

        targets = tiers or [ModelTier.FAST_LOCAL, ModelTier.BUILD_LOCAL]
        options = list(candidates or self.DEFAULT_CANDIDATES)

        for tier in targets:
            spec = self.catalog.get(tier)
            if not (spec.enabled and spec.configured):
                policy.notes.append(f"{tier.value}: skipped (not enabled/configured)")
                continue

            measurements = self.benchmark.sweep(spec, options)
            policy.measurements[tier.value] = [item.to_dict() for item in measurements]
            chosen, note = self._choose(tier, measurements, host, min_tokens_per_second)
            policy.context_windows[tier.value] = chosen
            policy.notes.append(note)

        policy.max_concurrent_generations = self._choose_concurrency(host)
        policy.reserved_vram_mib = reserved_vram_mib(host)
        return policy

    def _choose(
        self,
        tier: ModelTier,
        measurements: list[ContextMeasurement],
        host: HostInfo,
        min_tokens_per_second: float,
    ) -> tuple[int, str]:
        """Pick the largest context that is fast enough and leaves headroom.

        "Fast enough" is judged two ways, and a candidate must pass both:

        *Absolutely*, against ``min_tokens_per_second`` -- this catches a model
        that has spilled to CPU entirely.

        *Relatively*, against the best throughput measured for this same tier.
        This is the check that matters in practice.  On the GTX 1070,
        qwen3:4b sustains ~56 tok/s from 4k to 24k context and then drops to
        ~31 tok/s at 32k: still far above any absolute floor, but a 45% collapse
        that unmistakably signals memory pressure.  Taking that configuration to
        gain nominal context would trade the machine's responsiveness for a
        number nobody benefits from.
        """

        usable = [item for item in measurements if item.ok]
        if not usable:
            fallback = ResourcePolicy.default_for(host).context_windows.get(tier.value, 4096)
            reason = measurements[0].error if measurements else "no measurement"
            return fallback, f"{tier.value}: no context size completed ({reason}); falling back to {fallback}"

        gpu = host.primary_gpu
        reserve = reserved_vram_mib(host)
        peak = max(item.tokens_per_second for item in usable) or 0.0
        floor = max(min_tokens_per_second, peak * self.throughput_retention)

        acceptable = [
            item
            for item in usable
            if item.tokens_per_second >= floor
            and (not gpu or item.vram_free_mib == 0 or item.vram_free_mib >= reserve)
        ]
        if not acceptable:
            # Every size was either too slow or too tight; take the smallest
            # that at least completed rather than nothing.
            smallest = min(usable, key=lambda item: item.context_window)
            return (
                smallest.context_window,
                f"{tier.value}: no size held {floor:.1f} tok/s with {reserve} MiB headroom; "
                f"using smallest working size {smallest.context_window}",
            )

        best = max(acceptable, key=lambda item: item.context_window)
        rejected = [item for item in usable if item.context_window > best.context_window]
        note = (
            f"{tier.value}: chose {best.context_window} "
            f"({best.tokens_per_second} tok/s, {best.vram_free_mib} MiB VRAM free)"
        )
        if rejected:
            worst = rejected[0]
            reason = (
                f"{worst.tokens_per_second} tok/s < {floor:.1f} floor"
                if worst.tokens_per_second < floor
                else f"only {worst.vram_free_mib} MiB VRAM free < {reserve} reserve"
            )
            note += f"; rejected {worst.context_window} ({reason})"
        return best.context_window, note

    def _choose_concurrency(self, host: HostInfo) -> int:
        """One generation at a time unless there is clearly room for more.

        Two concurrent generations on a single 8 GB card means two resident
        models, which is exactly the situation that makes the desktop unusable.
        """

        gpu = host.primary_gpu
        if not gpu:
            return 1
        if len(host.gpus) > 1 or gpu.total_mib >= 24000:
            return 2
        return 1


class ResourcePolicyStore:
    """Persists the tuned policy so tuning happens once, not every start."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ResourcePolicy | None:
        if not self.path.exists():
            return None
        try:
            return ResourcePolicy.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def save(self, policy: ResourcePolicy) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(policy.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)
        return self.path

    def load_or_default(self, host: HostInfo | None = None) -> ResourcePolicy:
        existing = self.load()
        if existing is not None:
            return existing
        return ResourcePolicy.default_for(host or HostProbe().detect())
