"""The execution context: which device ZEUS is acting through, and what it has.

Not a smart home.  The one abstraction the future needs -- "play it here",
"show it on the TV", "turn off the light here" -- is a record of where a
request is being handled: a device id and type, a room, the inputs and
outputs present, and the capabilities that make sense there.  The desktop
this process runs on is the first device; paired devices (``/api/device/*``)
are the others.  The composer reads ``available`` to refuse a step that needs
a speaker on a device that has none.
"""

from __future__ import annotations

import json
import os
import platform
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DeviceContext:
    device_id: str
    device_type: str  # desktop | tv | tablet | phone | node
    name: str = ""
    room: str = ""
    inputs: list[str] = field(default_factory=list)     # keyboard, mouse, touch, microphone, remote
    outputs: list[str] = field(default_factory=list)    # screen, speaker
    screen: dict[str, Any] = field(default_factory=dict)
    speaker: bool = False
    microphone: bool = False
    capabilities: list[str] = field(default_factory=list)
    online: bool = True
    latency_ms: float = 0.0

    @property
    def available(self) -> list[str]:
        out = []
        if "screen" in self.outputs:
            out.append("screen")
        if self.speaker:
            out.append("speaker")
        if self.microphone:
            out.append("microphone")
        out.append("network")
        return out

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["available"] = self.available
        return data


def _config_path(core: Any) -> Path:
    return Path(core.kernel.state_root) / "devices" / "this_device.json"


def current_context(core: Any) -> DeviceContext:
    """This machine, from what the runtime can see, overridable by a small config."""

    hostname = socket.gethostname()
    override: dict[str, Any] = {}
    try:
        path = _config_path(core)
        if path.is_file():
            override = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        override = {}
    stages = {}
    try:
        stages = core.lifecycle.stages or {}
    except Exception:  # noqa: BLE001
        pass
    desktop = getattr(core.lifecycle, "desktop", None)
    has_window = bool(desktop is not None and desktop.status().get("available"))
    voice_ok = bool(stages.get("voice", {}).get("ok"))
    recogniser_ok = bool(stages.get("recogniser", {}).get("ok"))
    caps: list[str] = []
    try:
        caps = [str(m.capability_id) for m in core.capabilities.registry.all()]
    except Exception:  # noqa: BLE001
        pass
    ctx = DeviceContext(
        device_id=str(override.get("device_id") or f"desktop:{hostname.lower()}"),
        device_type=str(override.get("device_type") or "desktop"),
        name=str(override.get("name") or hostname),
        room=str(override.get("room") or ""),
        inputs=list(override.get("inputs") or (["keyboard", "mouse"] + (["microphone"] if recogniser_ok else []))),
        outputs=list(override.get("outputs") or ((["screen"] if has_window or os.environ.get("SESSIONNAME") else []) + (["speaker"] if voice_ok else []))),
        screen={"platform": platform.system(), "window": has_window},
        speaker=bool(override.get("speaker", voice_ok)),
        microphone=bool(override.get("microphone", recogniser_ok)),
        capabilities=caps,
    )
    return ctx


def set_context(core: Any, **fields: Any) -> DeviceContext:
    """Owner-provided facts about this device (its room, its name), kept on disk."""

    path = _config_path(core)
    path.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, Any] = {}
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        current = {}
    current.update({k: v for k, v in fields.items() if k in DeviceContext.__dataclass_fields__})
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current_context(core)
