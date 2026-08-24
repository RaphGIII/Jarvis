"""Identity, pairing and presence for the things that talk to Jarvis Core.

Until now there was one shared token. That is correct for a browser on
loopback and wrong the moment a second machine is involved, for a reason worth
stating: a single token cannot be revoked. Losing a phone means changing the
secret every other client already uses.

So each device gets its own credential, and three properties follow from that
which the shared token could not provide:

*Revocation is per device.* Unpairing the box in the kitchen does not disturb
the browser or the phone.

*The audit trail is real.* "Audio arrived" becomes "audio arrived from the
kitchen box at 14:02", which is the difference between a log and evidence.

*Enrolment is a decision.* A device cannot pair itself. It asks, the request
appears with a short code, and a human approves it -- because anything that can
reach the port would otherwise be able to join, and on a home network that is
every appliance in the house.

Presence is tracked separately from pairing for the same reason liveness is
tracked separately from progress elsewhere in this system: a device that is
enrolled and unplugged is a different state from one that was never enrolled,
and the UI needs to say which.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: How long a pairing request stays open. Short: a code that lives for hours is
#: a code an attacker has hours to use.
PAIRING_TTL_SECONDS = 300.0

#: A device unheard from for this long is absent rather than merely quiet.
PRESENCE_TIMEOUT_SECONDS = 90.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Device:
    """One paired client."""

    id: str
    name: str
    #: "browser" | "listener" | "box" | "phone" | "tv" | anything a client calls itself.
    kind: str = "generic"
    #: What this device can do, so the core does not send a display command to
    #: a thing with no screen.
    capabilities: list[str] = field(default_factory=list)
    paired_at: str = field(default_factory=_now)
    last_seen: str = ""
    revoked: bool = False
    #: Never serialised to a client. Compared with compare_digest.
    token: str = ""

    def to_dict(self, *, include_token: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_token:
            data.pop("token", None)
        data["present"] = self.present
        return data

    @property
    def present(self) -> bool:
        if not self.last_seen or self.revoked:
            return False
        try:
            seen = datetime.fromisoformat(self.last_seen)
        except ValueError:
            return False
        age = (datetime.now(timezone.utc) - seen).total_seconds()
        return age <= PRESENCE_TIMEOUT_SECONDS


@dataclass
class PairingRequest:
    """A device asking to join, waiting for a human."""

    code: str
    name: str
    kind: str
    capabilities: list[str] = field(default_factory=list)
    requested_at: float = field(default_factory=time.monotonic)
    #: Set once approved; this is what the device collects.
    device_id: str = ""
    token: str = ""
    denied: bool = False

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.requested_at) > PAIRING_TTL_SECONDS

    @property
    def approved(self) -> bool:
        return bool(self.token) and not self.denied

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "kind": self.kind,
            "capabilities": self.capabilities,
            "approved": self.approved,
            "denied": self.denied,
            "expired": self.expired,
        }


class DeviceGateway:
    """The registry, the pairing flow, and per-device presence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._devices: dict[str, Device] = {}
        self._pending: dict[str, PairingRequest] = {}
        #: Display/action commands waiting for each device to collect them.
        self._outbox: dict[str, list[dict[str, Any]]] = {}
        self._load()

    # -- persistence -----------------------------------------------------

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for item in data.get("devices", []) if isinstance(data, dict) else []:
            try:
                device = Device(**item)
            except TypeError:
                continue
            self._devices[device.id] = device

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"devices": [asdict(device) for device in self._devices.values()]}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    # -- pairing ---------------------------------------------------------

    def request_pairing(self, name: str, *, kind: str = "generic", capabilities: list[str] | None = None) -> PairingRequest:
        """A device asks to join.  Returns the code a human must approve."""

        with self._lock:
            self._sweep()
            # Six digits, not a UUID: it has to be readable off one screen and
            # typed into another, and it only has to survive five minutes.
            code = f"{secrets.randbelow(1_000_000):06d}"
            request = PairingRequest(
                code=code,
                name=(name or "unnamed device").strip()[:60],
                kind=(kind or "generic").strip()[:30],
                capabilities=list(capabilities or []),
            )
            self._pending[code] = request
            return request

    def pending(self) -> list[PairingRequest]:
        with self._lock:
            self._sweep()
            return [item for item in self._pending.values() if not item.approved and not item.denied]

    def approve(self, code: str) -> Device | None:
        """A human accepts a pairing request; the device gets its own token."""

        with self._lock:
            self._sweep()
            request = self._pending.get(code)
            if request is None or request.expired or request.denied:
                return None
            device = Device(
                id=f"dev_{secrets.token_hex(6)}",
                name=request.name,
                kind=request.kind,
                capabilities=list(request.capabilities),
                token=secrets.token_urlsafe(32),
            )
            self._devices[device.id] = device
            request.device_id = device.id
            request.token = device.token
            self.save()
            return device

    def deny(self, code: str) -> bool:
        with self._lock:
            request = self._pending.get(code)
            if request is None:
                return False
            request.denied = True
            return True

    def collect(self, code: str) -> dict[str, Any]:
        """The device polls for the outcome of its own request."""

        with self._lock:
            self._sweep()
            request = self._pending.get(code)
            if request is None:
                return {"status": "unknown", "detail": "no such pairing request"}
            if request.denied:
                return {"status": "denied"}
            if request.expired:
                return {"status": "expired"}
            if not request.approved:
                return {"status": "pending"}
            # Handed over exactly once: a token that can be collected twice is
            # a token that can be collected by someone else.
            self._pending.pop(code, None)
            return {"status": "paired", "device_id": request.device_id, "token": request.token}

    def _sweep(self) -> None:
        for code in [key for key, item in self._pending.items() if item.expired]:
            self._pending.pop(code, None)

    # -- authentication --------------------------------------------------

    def authenticate(self, device_id: str, token: str) -> Device | None:
        """Identify a device from its credential, or None."""

        with self._lock:
            device = self._devices.get(str(device_id))
        if device is None or device.revoked or not device.token:
            return None
        if not secrets.compare_digest(device.token, str(token or "")):
            return None
        return device

    def touch(self, device_id: str) -> Device | None:
        """Record that a device is present.  Called on every heartbeat."""

        with self._lock:
            device = self._devices.get(str(device_id))
            if device is None:
                return None
            device.last_seen = _now()
            return device

    def revoke(self, device_id: str) -> bool:
        with self._lock:
            device = self._devices.get(str(device_id))
            if device is None:
                return False
            device.revoked = True
            device.token = ""
            self._outbox.pop(device_id, None)
            self.save()
            return True

    def forget(self, device_id: str) -> bool:
        with self._lock:
            existed = self._devices.pop(str(device_id), None) is not None
            self._outbox.pop(device_id, None)
            if existed:
                self.save()
            return existed

    # -- listing ---------------------------------------------------------

    def devices(self, *, include_revoked: bool = False) -> list[Device]:
        with self._lock:
            return [
                device for device in self._devices.values()
                if include_revoked or not device.revoked
            ]

    def get(self, device_id: str) -> Device | None:
        with self._lock:
            return self._devices.get(str(device_id))

    def status(self) -> dict[str, Any]:
        return {
            "devices": [device.to_dict() for device in self.devices()],
            "pending": [item.to_dict() for item in self.pending()],
            "present": sum(1 for device in self.devices() if device.present),
        }

    # -- commands to a device --------------------------------------------

    def send(self, device_id: str, command: str, payload: dict[str, Any] | None = None) -> bool:
        """Queue a command for a device to collect on its next poll.

        Queued rather than pushed because a device is not always connected, and
        a display command that vanishes when the TV is asleep is a display
        command that never worked.
        """

        with self._lock:
            device = self._devices.get(str(device_id))
            if device is None or device.revoked:
                return False
            outbox = self._outbox.setdefault(device.id, [])
            outbox.append({"command": command, "payload": payload or {}, "at": _now()})
            # Bounded: a device that never collects must not grow the core's
            # memory without limit.
            del outbox[:-50]
            return True

    def broadcast(self, command: str, payload: dict[str, Any] | None = None, *, capability: str = "") -> int:
        """Send to every present device, optionally only those that can act on it."""

        sent = 0
        for device in self.devices():
            if capability and capability not in device.capabilities:
                continue
            if self.send(device.id, command, payload):
                sent += 1
        return sent

    def drain(self, device_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return self._outbox.pop(str(device_id), [])
