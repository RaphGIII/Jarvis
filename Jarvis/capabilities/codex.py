from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class CodexAvailabilityState(str, Enum):
    READY = "READY"
    BUSY = "BUSY"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    OFFLINE = "OFFLINE"
    NOT_INSTALLED = "NOT_INSTALLED"
    ERROR = "ERROR"


@dataclass
class CodexAvailability:
    state: CodexAvailabilityState
    detail: str = ""
    checked_at: float = 0.0
    evidence: str = ""

    @property
    def ready(self) -> bool:
        return self.state is CodexAvailabilityState.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "detail": self.detail,
            "checked_at": self.checked_at,
            "evidence": self.evidence,
            "ready": self.ready,
        }


class StaticCodexAvailability:
    """Injectable availability for tests and controlled offline runs."""

    def __init__(self, state: CodexAvailabilityState | str, detail: str = "") -> None:
        raw = state.value if isinstance(state, CodexAvailabilityState) else str(state)
        self.state = CodexAvailabilityState(raw)
        self.detail = detail
        self.calls = 0

    def status(self, *, refresh: bool = False) -> CodexAvailability:
        self.calls += 1
        return CodexAvailability(self.state, self.detail, checked_at=time.time(), evidence="static")

    def invalidate(self, reason: str = "") -> None:
        self.detail = reason or self.detail


class CodexAvailabilityCache:
    """Fast cached status for the official ChatGPT-authenticated Codex client."""

    def __init__(
        self,
        *,
        gateway: Any = None,
        probe: Callable[[], CodexAvailability] | None = None,
        ttl_seconds: float = 90.0,
        executable: str | None = None,
    ) -> None:
        self.gateway = gateway
        self.probe = probe
        self.ttl_seconds = max(1.0, ttl_seconds)
        self.executable = executable
        self._cached: CodexAvailability | None = None

    def status(self, *, refresh: bool = False) -> CodexAvailability:
        if not refresh and self._cached is not None and time.time() - self._cached.checked_at <= self.ttl_seconds:
            return self._cached
        try:
            current = self.probe() if self.probe is not None else self._probe()
        except Exception as exc:  # noqa: BLE001 - availability is state, not a crash
            current = CodexAvailability(CodexAvailabilityState.ERROR, f"{type(exc).__name__}: {exc}")
        current.checked_at = current.checked_at or time.time()
        self._cached = current
        return current

    def invalidate(self, reason: str = "") -> None:
        if self._cached is not None and reason:
            self._cached.detail = reason
        self._cached = None

    def _probe(self) -> CodexAvailability:
        if self.gateway is not None:
            if hasattr(self.gateway, "status"):
                return _from_gateway_status(self.gateway.status())
            if hasattr(self.gateway, "availability"):
                return _from_provider_availability(self.gateway.availability())
            return CodexAvailability(CodexAvailabilityState.ERROR, "gateway has no status or availability method", evidence="expert gateway")
        exe = self.executable if self.executable is not None else shutil.which("codex")
        if not exe:
            return CodexAvailability(
                CodexAvailabilityState.NOT_INSTALLED,
                "codex CLI is not installed; install and sign in with the ChatGPT subscription client",
                evidence="PATH lookup",
            )
        try:
            completed = subprocess.run(
                [exe, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            return CodexAvailability(CodexAvailabilityState.ERROR, str(exc), evidence="codex --version")
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:300]
            lowered = detail.lower()
            if "auth" in lowered or "sign in" in lowered or "login" in lowered:
                return CodexAvailability(CodexAvailabilityState.AUTH_REQUIRED, detail, evidence="codex --version")
            return CodexAvailability(CodexAvailabilityState.ERROR, detail, evidence="codex --version")
        return CodexAvailability(CodexAvailabilityState.READY, (completed.stdout or "").strip()[:120], evidence="codex --version")


def _from_gateway_status(status: dict[str, Any]) -> CodexAvailability:
    state = str(status.get("state") or "").upper()
    if status.get("expert_available") or state in {"AVAILABLE", "READY"}:
        return CodexAvailability(CodexAvailabilityState.READY, "Codex expert gateway has an available subscription provider", evidence="expert gateway")
    if status.get("quota_exhausted") or state in {"QUOTA_EXHAUSTED", "RATE_LIMITED"}:
        return CodexAvailability(CodexAvailabilityState.QUOTA_EXHAUSTED, "subscription quota exhausted or rate limited", evidence="expert gateway")
    if state in {"BUSY"}:
        return CodexAvailability(CodexAvailabilityState.BUSY, "Codex is busy", evidence="expert gateway")
    if state in {"NOT_AUTHENTICATED", "AUTH_REQUIRED"}:
        return CodexAvailability(CodexAvailabilityState.AUTH_REQUIRED, "Codex needs ChatGPT authentication", evidence="expert gateway")
    if state in {"NOT_INSTALLED", "NOT_CONFIGURED"}:
        return CodexAvailability(CodexAvailabilityState.NOT_INSTALLED, "Codex is not installed or configured", evidence="expert gateway")
    if state in {"OFFLINE"}:
        return CodexAvailability(CodexAvailabilityState.OFFLINE, "Codex is offline", evidence="expert gateway")
    # The provider's own sentence, not the whole status object. Serialising the
    # dict put a JSON blob -- policy flags, timestamps and all -- into the
    # answer the owner reads: measured live, "Nicht gelernt: Codex is ERROR:
    # {"checked_at": 1788845136.37, "expert_available": false, "policy": {...".
    # The gateway already asked each provider why, and that answer is a
    # sentence.
    reasons = [
        str(row.get("detail") or "").strip()
        for row in (status.get("providers") or [])
        if isinstance(row, dict) and row.get("permitted") and not row.get("available")
    ]
    detail = "; ".join(reason for reason in reasons if reason)[:300]
    if not detail:
        detail = json.dumps(status, sort_keys=True, default=str)[:300] if status else "no gateway status"
    return CodexAvailability(CodexAvailabilityState.ERROR, detail, evidence="expert gateway")


def _from_provider_availability(status: Any) -> CodexAvailability:
    available = bool(getattr(status, "available", False))
    detail = str(getattr(status, "reason", "") or "")
    version = str(getattr(status, "version", "") or "")
    text = " ".join(part for part in (detail, version) if part).strip()
    lowered = text.lower()
    if available:
        return CodexAvailability(CodexAvailabilityState.READY, text or "Codex provider available", evidence="provider availability")
    if any(marker in lowered for marker in ("quota", "rate limit", "usage limit", "insufficient_quota")):
        return CodexAvailability(CodexAvailabilityState.QUOTA_EXHAUSTED, text, evidence="provider availability")
    if any(marker in lowered for marker in ("auth", "sign in", "login", "not authenticated")):
        return CodexAvailability(CodexAvailabilityState.AUTH_REQUIRED, text, evidence="provider availability")
    if "not installed" in lowered or "not configured" in lowered:
        return CodexAvailability(CodexAvailabilityState.NOT_INSTALLED, text, evidence="provider availability")
    if "offline" in lowered or "network" in lowered:
        return CodexAvailability(CodexAvailabilityState.OFFLINE, text, evidence="provider availability")
    return CodexAvailability(CodexAvailabilityState.ERROR, text or "Codex provider unavailable", evidence="provider availability")


@dataclass
class CapabilityRequest:
    goal: str
    context: dict[str, Any] = field(default_factory=dict)
    requested_at: float = field(default_factory=time.time)
    priority: str = "normal"
    request_id: str = field(default_factory=lambda: "capreq_" + uuid.uuid4().hex[:12])
    reason: str = ""
    status: str = "queued"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CapabilityRequestQueue:
    """Append-only queue for capability learning when Codex is unavailable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        goal: str,
        *,
        context: dict[str, Any] | None = None,
        priority: str = "normal",
        reason: str = "",
    ) -> CapabilityRequest:
        request = CapabilityRequest(goal=goal, context=dict(context or {}), priority=priority, reason=reason)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request.to_dict(), sort_keys=True) + "\n")
        return request

    def list(self, *, status: str = "queued") -> list[CapabilityRequest]:
        """The current state of every request, newest write per id wins.

        The file is append-only, so a request that was later served appears
        twice. Reading it as a log rather than as a table is what makes
        "still waiting" and "already handled" distinguishable -- without it a
        queue only ever grows and the Capability Center shows work that was
        finished hours ago as outstanding.
        """

        latest: dict[str, CapabilityRequest] = {}
        order: list[str] = []
        for data in self._rows():
            request = CapabilityRequest(
                goal=str(data.get("goal", "")),
                context=dict(data.get("context") or {}),
                requested_at=float(data.get("requested_at", 0.0) or 0.0),
                priority=str(data.get("priority", "normal")),
                request_id=str(data.get("request_id", "")),
                reason=str(data.get("reason", "")),
                status=str(data.get("status", "queued")),
            )
            if request.request_id not in latest:
                order.append(request.request_id)
            elif not request.goal:
                request.goal = latest[request.request_id].goal
                request.context = request.context or latest[request.request_id].context
            latest[request.request_id] = request
        return [latest[key] for key in order if not status or latest[key].status == status]

    def resolve(self, request_id: str, *, detail: str = "", status: str = "served") -> CapabilityRequest | None:
        """Mark one queued request as handled, without rewriting history."""

        current = {request.request_id: request for request in self.list(status="")}
        request = current.get(str(request_id))
        if request is None or request.status != "queued":
            return request
        request.status = status
        request.reason = detail or request.reason
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request.to_dict(), sort_keys=True) + "\n")
        return request

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if isinstance(data, dict):
                rows.append(data)
        return rows
