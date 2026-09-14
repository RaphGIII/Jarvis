"""The cost governor: reserve before you spend, and never past the cap.

Every paid request follows the same protocol:

    estimate -> reserve(estimate * safety_factor) -> call -> settle(actual)

:meth:`BudgetGovernor.reserve` refuses when the reservation would push any
hard cap -- month, day, task, provider, role family -- past its limit.  A
refused reservation is the end of the request; nothing downstream may proceed
without a :class:`Reservation` in hand.  Settlement replaces the reservation
with the provider-reported actual and releases the difference.

The ledger is an append-only JSONL file so the month's spend survives
restarts and can be audited line by line.  Open reservations count as spent
until settled or released; a process that dies mid-call leaves its
reservation counted, which errs on the side of not overspending.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gateway.config import BudgetConfig
from gateway.modes import RoleFamily


class BudgetRefused(RuntimeError):
    """A reservation would exceed a hard cap.  Nothing was spent."""

    def __init__(self, cap: str, requested: float, available: float, detail: str = "") -> None:
        self.cap = cap
        self.requested = requested
        self.available = available
        message = f"budget refused by {cap}: requested EUR {requested:.4f}, available EUR {max(0.0, available):.4f}"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


@dataclass
class Reservation:
    reservation_id: str
    role: str
    provider: str
    model: str
    task_id: str
    estimated_eur: float
    reserved_eur: float
    at: str
    mode: str = ""
    settled: bool = False
    actual_eur: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reservation_id": self.reservation_id, "role": self.role, "provider": self.provider, "model": self.model,
            "task_id": self.task_id, "estimated_eur": self.estimated_eur, "reserved_eur": self.reserved_eur,
            "at": self.at, "mode": self.mode, "settled": self.settled, "actual_eur": self.actual_eur,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _month_key(when: datetime) -> str:
    return when.strftime("%Y-%m")


def _day_key(when: datetime) -> str:
    return when.strftime("%Y-%m-%d")


@dataclass
class SpendSummary:
    month: float
    day: float
    reasoning_month: float
    engineering_month: float
    per_provider_month: dict[str, float]
    open_reservations: int
    month_key: str
    day_key: str
    config: BudgetConfig
    entries_month: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": "EUR",
            "month": round(self.month, 4), "day": round(self.day, 4),
            "reasoning_month": round(self.reasoning_month, 4), "engineering_month": round(self.engineering_month, 4),
            "per_provider_month": {k: round(v, 4) for k, v in self.per_provider_month.items()},
            "open_reservations": self.open_reservations,
            "month_key": self.month_key, "day_key": self.day_key,
            "monthly_hard_cap": self.config.monthly_hard_cap, "daily_hard_cap": self.config.daily_hard_cap,
            "per_task_hard_cap": self.config.per_task_hard_cap,
            "reasoning_hard_cap": self.config.reasoning_hard_cap, "engineering_hard_cap": self.config.engineering_hard_cap,
            "remaining_month": round(max(0.0, self.config.monthly_hard_cap - self.month), 4),
            "remaining_day": round(max(0.0, self.config.daily_hard_cap - self.day), 4),
            "entries_month": self.entries_month,
        }


class BudgetGovernor:
    """Hard caps over an append-only spend ledger."""

    def __init__(self, ledger_path: str | Path, config: BudgetConfig | None = None) -> None:
        self.path = Path(ledger_path)
        self.config = config or BudgetConfig()
        self._lock = threading.RLock()
        #: reservation_id -> Reservation, for those not yet settled or released.
        self._open: dict[str, Reservation] = {}
        #: Cached settled amounts: (month_key, day_key, role, provider, task_id, eur)
        self._settled: list[tuple[str, str, str, str, str, float]] = []
        self._load()

    # -- persistence -----------------------------------------------------------

    def _load(self) -> None:
        self._open.clear()
        self._settled.clear()
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            kind = row.get("kind")
            if kind == "reserve":
                self._open[row["reservation_id"]] = Reservation(
                    reservation_id=row["reservation_id"], role=row.get("role", ""), provider=row.get("provider", ""),
                    model=row.get("model", ""), task_id=row.get("task_id", ""), estimated_eur=float(row.get("estimated_eur", 0.0)),
                    reserved_eur=float(row.get("reserved_eur", 0.0)), at=row.get("at", ""), mode=row.get("mode", ""),
                )
            elif kind in {"settle", "release"}:
                reservation = self._open.pop(row.get("reservation_id", ""), None)
                if kind == "settle":
                    when = row.get("at", "")
                    self._settled.append((when[:7], when[:10], row.get("role", getattr(reservation, "role", "")),
                                          row.get("provider", getattr(reservation, "provider", "")),
                                          row.get("task_id", getattr(reservation, "task_id", "")),
                                          float(row.get("actual_eur", 0.0))))

    def _append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    # -- accounting -------------------------------------------------------------

    def summary(self, *, when: datetime | None = None) -> SpendSummary:
        when = when or _now()
        month, day = _month_key(when), _day_key(when)
        with self._lock:
            total_month = total_day = reasoning = engineering = 0.0
            per_provider: dict[str, float] = {}
            entries = 0
            for m, d, role, provider, _task, eur in self._settled:
                if m != month:
                    continue
                entries += 1
                total_month += eur
                per_provider[provider] = per_provider.get(provider, 0.0) + eur
                if d == day:
                    total_day += eur
                family = role.split(".", 1)[0]
                if family == RoleFamily.REASONING.value:
                    reasoning += eur
                elif family == RoleFamily.ENGINEER.value:
                    engineering += eur
            for reservation in self._open.values():
                if reservation.at[:7] != month:
                    continue
                total_month += reservation.reserved_eur
                per_provider[reservation.provider] = per_provider.get(reservation.provider, 0.0) + reservation.reserved_eur
                if reservation.at[:10] == day:
                    total_day += reservation.reserved_eur
                family = reservation.role.split(".", 1)[0]
                if family == RoleFamily.REASONING.value:
                    reasoning += reservation.reserved_eur
                elif family == RoleFamily.ENGINEER.value:
                    engineering += reservation.reserved_eur
            return SpendSummary(month=total_month, day=total_day, reasoning_month=reasoning, engineering_month=engineering,
                                per_provider_month=per_provider, open_reservations=len(self._open), month_key=month,
                                day_key=day, config=self.config, entries_month=entries)

    def task_spend(self, task_id: str) -> float:
        with self._lock:
            settled = sum(eur for *_rest, task, eur in self._settled if task == task_id)
            open_ = sum(r.reserved_eur for r in self._open.values() if r.task_id == task_id)
            return settled + open_

    # -- the protocol ------------------------------------------------------------

    def can_reserve(self, *, role: str, provider: str, estimated_eur: float, task_id: str = "",
                    task_cap_eur: float | None = None, provider_cap_eur: float | None = None) -> tuple[bool, str, float]:
        """Would :meth:`reserve` succeed?  Returns (ok, cap name, reserved amount)."""

        reserved = round(max(0.0, float(estimated_eur)) * self.config.safety_factor, 6)
        if reserved <= 0.0:
            return True, "", 0.0
        summary = self.summary()
        checks: list[tuple[str, float, float]] = [
            ("monthly_hard_cap", self.config.monthly_hard_cap, summary.month),
            ("daily_hard_cap", self.config.daily_hard_cap, summary.day),
        ]
        cap_task = self.config.per_task_hard_cap if task_cap_eur is None else min(self.config.per_task_hard_cap, task_cap_eur)
        checks.append(("per_task_hard_cap", cap_task, self.task_spend(task_id) if task_id else 0.0))
        family = role.split(".", 1)[0]
        if family == RoleFamily.REASONING.value:
            checks.append(("reasoning_hard_cap", self.config.reasoning_hard_cap, summary.reasoning_month))
        elif family == RoleFamily.ENGINEER.value:
            checks.append(("engineering_hard_cap", self.config.engineering_hard_cap, summary.engineering_month))
        provider_cap = self.config.provider_hard_caps.get(provider)
        if provider_cap_eur is not None:
            provider_cap = provider_cap_eur if provider_cap is None else min(provider_cap, provider_cap_eur)
        if provider_cap is not None:
            checks.append((f"provider_hard_cap:{provider}", provider_cap, summary.per_provider_month.get(provider, 0.0)))
        for name, cap, spent in checks:
            if spent + reserved > cap + 1e-9:
                return False, name, reserved
        return True, "", reserved

    def reserve(self, *, role: str, provider: str, model: str, estimated_eur: float, task_id: str = "",
                mode: str = "", task_cap_eur: float | None = None, provider_cap_eur: float | None = None) -> Reservation:
        with self._lock:
            ok, cap, reserved = self.can_reserve(role=role, provider=provider, estimated_eur=estimated_eur, task_id=task_id,
                                                 task_cap_eur=task_cap_eur, provider_cap_eur=provider_cap_eur)
            if not ok:
                summary = self.summary()
                available = {
                    "monthly_hard_cap": self.config.monthly_hard_cap - summary.month,
                    "daily_hard_cap": self.config.daily_hard_cap - summary.day,
                    "reasoning_hard_cap": self.config.reasoning_hard_cap - summary.reasoning_month,
                    "engineering_hard_cap": self.config.engineering_hard_cap - summary.engineering_month,
                }.get(cap)
                if available is None and cap == "per_task_hard_cap":
                    limit = self.config.per_task_hard_cap if task_cap_eur is None else min(self.config.per_task_hard_cap, task_cap_eur)
                    available = limit - self.task_spend(task_id)
                if available is None:
                    available = 0.0
                raise BudgetRefused(cap, reserved, available, detail=f"role={role} provider={provider}")
            reservation = Reservation(
                reservation_id=uuid.uuid4().hex[:12], role=role, provider=provider, model=model, task_id=task_id,
                estimated_eur=round(float(estimated_eur), 6), reserved_eur=reserved, at=_now().isoformat(), mode=mode,
            )
            if reserved > 0.0:
                self._open[reservation.reservation_id] = reservation
                self._append({"kind": "reserve", **reservation.to_dict()})
            return reservation

    def settle(self, reservation: Reservation, actual_eur: float, *, usage: dict[str, Any] | None = None,
               native: float | None = None, currency: str = "EUR", rate_source: str = "") -> None:
        with self._lock:
            reservation.settled = True
            reservation.actual_eur = round(max(0.0, float(actual_eur)), 6)
            if reservation.reserved_eur <= 0.0 and reservation.actual_eur <= 0.0:
                return
            self._open.pop(reservation.reservation_id, None)
            when = _now().isoformat()
            self._settled.append((when[:7], when[:10], reservation.role, reservation.provider, reservation.task_id,
                                  reservation.actual_eur))
            self._append({"kind": "settle", "reservation_id": reservation.reservation_id, "role": reservation.role,
                          "provider": reservation.provider, "model": reservation.model, "task_id": reservation.task_id,
                          "estimated_eur": reservation.estimated_eur, "reserved_eur": reservation.reserved_eur,
                          "actual_eur": reservation.actual_eur, "usage": dict(usage or {}), "at": when,
                          "actual_native": round(float(native), 6) if native is not None else reservation.actual_eur,
                          "currency": currency, "rate_source": rate_source})

    def release(self, reservation: Reservation, reason: str = "") -> None:
        """The call never happened (or failed before billing): give the money back."""

        with self._lock:
            if self._open.pop(reservation.reservation_id, None) is None:
                return
            self._append({"kind": "release", "reservation_id": reservation.reservation_id, "reason": reason[:200],
                          "at": _now().isoformat()})

    def history(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows
