from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExponentialMovingAverage:
    alpha: float = 0.2
    value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return self.value
