"""What ZEUS has learned about which role is good enough for what.

Every completed request writes one privacy-safe observation: task class and
vector, the role and provider that served it, estimated and actual cost,
latency, and whether the *goal was verified* -- never the model's own opinion
of how it did, and never the prompt or the reasoning.

From those observations :class:`ReliabilityModel` maintains, per (role, task
class), a Beta posterior over the success probability:

    p_hat  = (alpha) / (alpha + beta)
    sigma  = sqrt(alpha * beta / ((alpha + beta)^2 (alpha + beta + 1)))
    q      = p_hat - risk_aversion * sigma

``q`` is the conservative lower estimate the router compares against the
task's required reliability.  The prior comes from configuration, so a fresh
installation has opinions -- the free model is probably fine for knowledge
questions, probably not for long-horizon composition -- and real evidence
moves them.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gateway.config import GatewayConfig
from gateway.task import TaskClass, TaskVector


@dataclass(frozen=True)
class Observation:
    task_class: str
    role: str
    provider: str
    model: str
    goal_verified: bool
    #: One of: "" (success), "task_failure", "provider_unavailable", "rate_limit",
    #: "quota_exhausted", "authentication_error", "model_unavailable",
    #: "budget_refused", "privacy_refused", "mode_refused".
    failure_class: str = ""
    estimated_eur: float = 0.0
    actual_eur: float = 0.0
    latency_seconds: float = 0.0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    task_vector: dict[str, float] = field(default_factory=dict)
    mode: str = ""
    #: The abstract reasoning effort the call used (FAST / NORMAL / DEEP / MAX), "" when none.
    thinking_level: str = ""
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def counts_for_reliability(self) -> bool:
        """Only task outcomes teach reliability.  An outage teaches nothing about the model."""

        return self.failure_class in {"", "task_failure"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_class": self.task_class, "role": self.role, "provider": self.provider, "model": self.model,
            "goal_verified": self.goal_verified, "failure_class": self.failure_class,
            "estimated_eur": self.estimated_eur, "actual_eur": self.actual_eur, "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens, "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens, "task_vector": dict(self.task_vector), "mode": self.mode,
            "thinking_level": self.thinking_level, "at": self.at,
        }


@dataclass(frozen=True)
class Reliability:
    role: str
    task_class: str
    alpha: float
    beta: float
    observations: int

    @property
    def p_hat(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def sigma(self) -> float:
        n = self.alpha + self.beta
        return math.sqrt(self.alpha * self.beta / (n * n * (n + 1.0)))

    def q(self, risk_aversion: float = 1.0) -> float:
        return max(0.0, self.p_hat - risk_aversion * self.sigma)

    def to_dict(self, risk_aversion: float = 1.0) -> dict[str, Any]:
        return {"role": self.role, "task_class": self.task_class, "p_hat": round(self.p_hat, 4),
                "sigma": round(self.sigma, 4), "q": round(self.q(risk_aversion), 4), "observations": self.observations,
                "alpha": round(self.alpha, 2), "beta": round(self.beta, 2)}


class PerformanceLedger:
    """Append-only observations, privacy-safe by construction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, observation: Observation) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(observation.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")

    def observations(self, *, limit: int | None = None) -> list[Observation]:
        if not self.path.is_file():
            return []
        rows: list[Observation] = []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if limit:
            lines = lines[-limit:]
        for line in lines:
            try:
                data = json.loads(line)
            except ValueError:
                continue
            try:
                rows.append(Observation(
                    task_class=str(data.get("task_class", "")), role=str(data.get("role", "")),
                    provider=str(data.get("provider", "")), model=str(data.get("model", "")),
                    goal_verified=bool(data.get("goal_verified", False)), failure_class=str(data.get("failure_class", "")),
                    estimated_eur=float(data.get("estimated_eur", 0.0)), actual_eur=float(data.get("actual_eur", 0.0)),
                    latency_seconds=float(data.get("latency_seconds", 0.0)), input_tokens=int(data.get("input_tokens", 0)),
                    cached_input_tokens=int(data.get("cached_input_tokens", 0)), output_tokens=int(data.get("output_tokens", 0)),
                    task_vector=dict(data.get("task_vector") or {}), mode=str(data.get("mode", "")), at=str(data.get("at", "")),
                ))
            except (TypeError, ValueError):
                continue
        return rows


class ReliabilityModel:
    """Beta posteriors per (role, task class), prior from config, evidence from the ledger."""

    #: Newest observations weigh more: each older observation decays by this factor
    #: per 50 observations of the same key, so a provider that improved is not
    #: held to last quarter's failures for ever.
    DECAY = 0.98

    def __init__(self, config: GatewayConfig, ledger: PerformanceLedger | None = None) -> None:
        self.config = config
        self.ledger = ledger
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str], tuple[float, float, int]] = {}
        self._load()

    def _prior(self, role: str, task_class: str) -> tuple[float, float]:
        binding = self.config.binding(role)
        if binding is not None:
            prior = binding.reliability_prior.get(task_class)
            if prior:
                return max(0.5, float(prior[0])), max(0.5, float(prior[1]))
            # A family-level default when the class is unknown to the prior.
            if binding.reliability_prior:
                alphas = [v[0] for v in binding.reliability_prior.values()]
                betas = [v[1] for v in binding.reliability_prior.values()]
                return max(0.5, sum(alphas) / len(alphas)), max(0.5, sum(betas) / len(betas))
        return 1.0, 1.0

    def _load(self) -> None:
        if self.ledger is None:
            return
        for observation in self.ledger.observations():
            self._absorb(observation)

    def _absorb(self, observation: Observation) -> None:
        if not observation.counts_for_reliability:
            return
        key = (observation.role, observation.task_class)
        alpha, beta, n = self._counts.get(key, (0.0, 0.0, 0))
        alpha *= self.DECAY
        beta *= self.DECAY
        if observation.goal_verified:
            alpha += 1.0
        else:
            beta += 1.0
        self._counts[key] = (alpha, beta, n + 1)

    def observe(self, observation: Observation) -> None:
        with self._lock:
            self._absorb(observation)
        if self.ledger is not None:
            self.ledger.record(observation)

    def reliability(self, role: str, task_class: TaskClass | str) -> Reliability:
        cls = task_class.value if isinstance(task_class, TaskClass) else str(task_class)
        prior_alpha, prior_beta = self._prior(role, cls)
        with self._lock:
            alpha, beta, n = self._counts.get((role, cls), (0.0, 0.0, 0))
        return Reliability(role=role, task_class=cls, alpha=prior_alpha + alpha, beta=prior_beta + beta, observations=n)

    def q(self, role: str, task: TaskVector) -> float:
        return self.reliability(role, task.task_class).q(self.config.risk_aversion)

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for role in self.config.roles:
            for cls in TaskClass:
                rel = self.reliability(role, cls)
                rows.append(rel.to_dict(self.config.risk_aversion))
        return rows
