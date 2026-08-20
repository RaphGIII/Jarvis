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
        total_step = self.buckets.get("total_step", TimingBucket()).seconds
        total_episode = self.buckets.get("total_episode", TimingBucket()).seconds
        wall_time = total_episode or total_step
        component_names = [
            "brain_candidate_generation",
            "semantic_observation_encoding",
            "semantic_action_encoding",
            "policy_q_world_scoring",
            "docker_execution",
            "optimizer_training_update",
        ]
        component_total = sum(self.buckets.get(name, TimingBucket()).seconds for name in component_names)
        denominator = wall_time or component_total
        timings = {
            name: {
                "seconds": bucket.seconds,
                "count": bucket.count,
                "percent": (bucket.seconds / denominator * 100.0) if denominator > 0 and name in component_names else 0.0,
                "mean_seconds": (bucket.seconds / bucket.count) if bucket.count else 0.0,
                "kind": "component" if name in component_names else "total",
            }
            for name, bucket in sorted(self.buckets.items())
        }
        other = max(0.0, denominator - component_total) if denominator > 0 else 0.0
        breakdown = {
            name: {
                "seconds": self.buckets.get(name, TimingBucket()).seconds,
                "percent": (self.buckets.get(name, TimingBucket()).seconds / denominator * 100.0) if denominator > 0 else 0.0,
            }
            for name in component_names
        }
        breakdown["other"] = {"seconds": other, "percent": (other / denominator * 100.0) if denominator > 0 else 0.0}
        return {
            "wall_time_seconds": wall_time,
            "total_step_seconds": total_step,
            "total_episode_seconds": total_episode,
            "breakdown": breakdown,
            "timings": timings,
            "counters": dict(sorted(self.counters.items())),
        }

    def format_summary(self) -> str:
        payload = self.summary()
        lines = [f"WALL TIME: {payload['wall_time_seconds']:.3f}s", "BREAKDOWN:"]
        for name, values in payload["breakdown"].items():
            lines.append(f"{name}: {values['seconds']:.3f}s | {values['percent']:.1f}%")
        lines.append("TOTALS:")
        for name in ["total_episode", "total_step"]:
            values = payload["timings"].get(name)
            if values:
                lines.append(f"{name}: {values['seconds']:.3f}s | n={values['count']}")
        for name, value in payload["counters"].items():
            lines.append(f"{name}: {value:g}")
        return "\n".join(lines)
