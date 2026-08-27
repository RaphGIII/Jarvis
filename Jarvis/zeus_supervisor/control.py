"""How the application talks to the supervisor: files, not imports.

The core cannot call into the supervisor -- the supervisor is a different
process and, when frozen, a different program.  What it can do is leave a
small JSON file in a directory both of them know, then exit with an agreed
code.  The supervisor reads the file after the child exits, so a request
survives the process that made it.

Requests:

``restart``
    Start again from the current tree.  Carries the reason and, for a
    promotion, the revision that is expected to come up and the promotion id
    so the deployment receipt can name it.  A restart that follows a
    promotion is verified against the known-good pointer: healthy means the
    new revision becomes known-good; unhealthy means rollback.

``shutdown``
    Stay down.  The supervisor stops everything and exits.

The application also reads the supervisor's *status* file to show whether it
is supervised, what the known-good revision is and what happened at the last
restart -- read-only from its side.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ControlRequest:
    action: str  # restart | shutdown | relaunch
    reason: str = ""
    expected_revision: str = ""
    promotion_id: str = ""
    requested_by: str = ""
    at: str = field(default_factory=_now)
    #: ``relaunch`` only: the promoted executable and the release to restore.
    exe: str = ""
    previous: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ControlChannel:
    """One pending request at a time, in ``<state_dir>/control/request.json``."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.directory = self.state_dir / "control"
        self.request_path = self.directory / "request.json"
        self.status_path = self.state_dir / "status.json"

    # -- application side ---------------------------------------------

    def request(self, action: str, **fields: Any) -> ControlRequest:
        req = ControlRequest(action=action, **fields)
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.request_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(req.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.request_path)
        return req

    def read_status(self) -> dict[str, Any]:
        if not self.status_path.is_file():
            return {}
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # -- supervisor side -----------------------------------------------

    def take(self) -> ControlRequest | None:
        """Read and remove the pending request, if any."""

        if not self.request_path.is_file():
            return None
        try:
            data = json.loads(self.request_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        try:
            self.request_path.unlink()
        except OSError:
            pass
        if not isinstance(data, dict) or not data.get("action"):
            return None
        return ControlRequest(
            action=str(data.get("action")),
            reason=str(data.get("reason", "")),
            expected_revision=str(data.get("expected_revision", "")),
            promotion_id=str(data.get("promotion_id", "")),
            requested_by=str(data.get("requested_by", "")),
            at=str(data.get("at", "")),
            exe=str(data.get("exe", "")),
            previous=str(data.get("previous", "")),
        )

    def write_status(self, status: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(status)
        payload["updated_at"] = _now()
        payload["supervisor_pid"] = os.getpid()
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.status_path)


def supervised() -> bool:
    """Whether this process was started by the supervisor."""

    return os.environ.get("ZEUS_SUPERVISED", "") == "1"


def control_dir_from_env() -> Path | None:
    raw = os.environ.get("ZEUS_SUPERVISOR_DIR", "").strip()
    return Path(raw) if raw else None
