from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class TimingBucket:
    seconds: float = 0.0
    count: int = 0

    def add(self, elapsed: float) -> None:
        self.seconds += float(elapsed)
        self.count += 1


@dataclass
class PerformanceProfiler:
    buckets: dict[str, TimingBucket] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_time(name, time.perf_counter() - started)

    def add_time(self, name: str, seconds: float) -> None:
        self.buckets.setdefault(name, TimingBucket()).add(seconds)

    def increment(self, name: str, amount: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + amount

    def summary(self) -> dict[str, object]:
        total = sum(bucket.seconds for bucket in self.buckets.values())
        timings = {
            name: {
                "seconds": bucket.seconds,
                "count": bucket.count,
                "percent": (bucket.seconds / total * 100.0) if total > 0 else 0.0,
                "mean_seconds": (bucket.seconds / bucket.count) if bucket.count else 0.0,
            }
            for name, bucket in sorted(self.buckets.items())
        }
        return {"total_profiled_seconds": total, "timings": timings, "counters": dict(sorted(self.counters.items()))}

    def format_summary(self) -> str:
        payload = self.summary()
        timings = payload["timings"]
        lines = ["PERFORMANCE:"]
        for name, values in timings.items():
            lines.append(f"{name}: {values['percent']:.1f}% | {values['seconds']:.3f}s | n={values['count']}")
        for name, value in payload["counters"].items():
            lines.append(f"{name}: {value:g}")
        return "\n".join(lines)
