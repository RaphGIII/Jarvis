"""Where the time actually goes, hop by hop.

``python -m jarvis.measure_pipeline`` -- runs the real components and reports
what each costs, so an optimisation can be aimed at a bottleneck rather than at
whatever felt slow.

Two honesty rules, both learned the hard way in this project.

*Measurement apparatus is not product behaviour.*  A previous harness reported
``wake word 28.17s`` for a hop that takes two tenths of a second, because it
timed a subprocess start in a separate venv along with the detection.  Anything
here that is setup says so and is excluded from the totals.

*Cold and warm are different facts.*  The first generation after a model load
costs tens of seconds and every later one costs a second.  Reporting one number
would misrepresent both, so both are reported.

The GPU holds one model at a time on this machine.  That makes model switching
a real, measurable cost rather than an implementation detail, and it is
measured here because two of the defects found this week were caused by
something touching the heavy model when it had no need to.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Something a user waits for, as opposed to something the harness had to do.
@dataclass
class Sample:
    name: str
    seconds: float
    detail: str = ""
    #: False for setup, warm-up and other apparatus.
    user_facing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop": self.name,
            "seconds": round(self.seconds, 3),
            "detail": self.detail[:200],
            "user_facing": self.user_facing,
        }


@dataclass
class PipelineMeasurement:
    samples: list[Sample] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, name: str, seconds: float, detail: str = "", *, user_facing: bool = True) -> None:
        self.samples.append(Sample(name, seconds, detail, user_facing))

    def time(self, name: str, work: Callable[[], Any], *, detail: str = "",
             user_facing: bool = True) -> Any:
        started = time.perf_counter()
        try:
            result = work()
        except Exception as exc:
            self.add(name, time.perf_counter() - started, f"FAILED: {type(exc).__name__}: {exc}",
                     user_facing=user_facing)
            return None
        elapsed = time.perf_counter() - started
        self.add(name, elapsed, detail() if callable(detail) else detail, user_facing=user_facing)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"samples": [s.to_dict() for s in self.samples], "notes": self.notes}

    def describe(self) -> str:
        width = max((len(s.name) for s in self.samples), default=12)
        lines = ["", "  ZEUS PIPELINE", ""]
        for sample in self.samples:
            mark = " " if sample.user_facing else "*"
            lines.append(f" {mark}{sample.name.ljust(width)}  {sample.seconds:8.3f}s   {sample.detail[:64]}")
        lines += ["", "  * = apparatus or setup, not something a user waits for"]
        if self.notes:
            lines += ["", "  notes:"] + [f"    - {note}" for note in self.notes]
        return "\n".join(lines)


def measure(*, heavy: bool = False) -> PipelineMeasurement:
    """Measure the pipeline.  ``heavy`` includes the BUILD_LOCAL costs."""

    from brain.tiers import ModelCatalog, ModelTier
    from core.kernel import JarvisKernel
    from runtime.environment import EnvironmentCache
    from service.core import JarvisCore
    from service.intent import classify
    from service.music import understand

    measurement = PipelineMeasurement()

    # -- things that cost nothing and are worth proving cost nothing ------
    started = time.perf_counter()
    for _ in range(2000):
        classify("Erklaer mir kurz was Rekursion ist.")
    measurement.add("classify (conversation)", (time.perf_counter() - started) / 2000,
                    "deterministic, per request")

    started = time.perf_counter()
    for _ in range(2000):
        classify("Erstelle die Datei x.txt mit dem Inhalt y")
    measurement.add("classify (action)", (time.perf_counter() - started) / 2000,
                    "deterministic, per request")

    started = time.perf_counter()
    for _ in range(2000):
        understand("Spiel Bohemian Rhapsody von Queen")
    measurement.add("parse (music)", (time.perf_counter() - started) / 2000,
                    "deterministic, no model call")

    # -- environment knowledge -------------------------------------------
    kernel = JarvisKernel()
    cache = EnvironmentCache(kernel.state_root / "environment.json")
    measurement.time("environment (probe)", lambda: cache.probe(),
                     detail="every probe, ignoring the cache", user_facing=False)
    measurement.time("environment (cached)", lambda: cache.facts(),
                     detail="fingerprint check, then the cache")

    # -- the conversation path -------------------------------------------
    core = JarvisCore(kernel=kernel)
    catalog = ModelCatalog()
    fast = catalog.get(ModelTier.FAST_LOCAL)

    provider = kernel.provider(ModelTier.FAST_LOCAL)
    measurement.time("FAST_LOCAL cold load", lambda: provider.generate("OK", max_tokens=4),
                     detail=f"{fast.model}", user_facing=False)

    def first_token() -> float:
        started_at = time.perf_counter()
        for _chunk in provider.generate_stream("Say the single word: ready", max_tokens=8):
            return time.perf_counter() - started_at
        return 0.0

    latency = measurement.time("FAST_LOCAL first token", lambda: first_token(), detail="warm")
    if isinstance(latency, float):
        measurement.samples[-1].seconds = latency

    measurement.time(
        "FAST_LOCAL full answer",
        lambda: "".join(provider.generate_stream(
            "Erklaer in zwei Saetzen, was Rekursion ist.", max_tokens=160)),
        detail="warm, ~2 sentences",
    )

    # -- the action path --------------------------------------------------
    from service.actions import ActionExecutor, ActionPlan

    executor = ActionExecutor(kernel)
    measurement.time(
        "tool execution (file.write)",
        lambda: executor.execute(
            ActionPlan("file.write", {"path": "_measure.txt", "content": "x"}), request="measure"),
        detail="write + independent readback + receipt",
    )

    # -- verification ------------------------------------------------------
    try:
        from tools import media_session

        measurement.time("verification (media session)", lambda: media_session.read(app="spotify"),
                         detail="one PowerShell + WinRT projection")
    except Exception:
        measurement.notes.append("media session unavailable; verification hop not measured")

    # -- the heavy tier, and what touching it costs ------------------------
    if heavy:
        build = catalog.get(ModelTier.BUILD_LOCAL)
        heavy_provider = kernel.provider(ModelTier.BUILD_LOCAL)
        measurement.time("BUILD_LOCAL cold load", lambda: heavy_provider.generate("OK", max_tokens=4),
                         detail=f"{build.model}", user_facing=False)
        measurement.time("BUILD_LOCAL generation",
                         lambda: heavy_provider.generate("Write a Python function that adds two numbers.",
                                                         max_tokens=120),
                         detail="warm")
        # The cost that matters: what the *next* conversational turn pays
        # because the heavy model is now resident.
        measurement.time("FAST_LOCAL after BUILD_LOCAL",
                         lambda: provider.generate("Hallo", max_tokens=8),
                         detail="the eviction tax on the next chat turn")
        measurement.notes.append(
            "One model fits in VRAM at a time, so the last hop is what any user request pays "
            "after anything touches the heavy tier."
        )

    return measurement


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jarvis.measure_pipeline")
    parser.add_argument("--heavy", action="store_true", help="also measure BUILD_LOCAL and eviction")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    measurement = measure(heavy=args.heavy)
    text = json.dumps(measurement.to_dict(), indent=2) if args.json else measurement.describe()
    print(text)
    if args.out:
        Path(args.out).write_text(json.dumps(measurement.to_dict(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
