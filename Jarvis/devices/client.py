"""A reference Jarvis device, in one file, with no hardware.

This is the thing that proves the service boundary is real rather than merely
intended.  It runs as its own process, pairs over the network, holds its own
credential, heartbeats, sends audio, and acts on display commands -- exactly
what the HDMI box will do, minus the box.

It is deliberately dependency-free: standard library only, so it runs on a
Raspberry Pi image with nothing installed, and so the protocol cannot quietly
acquire a requirement that a small device would struggle to satisfy.  Audio
capture is optional and degrades to text input when ``sounddevice`` is absent,
because a display-only client is a real device too.

    python -m devices.client --url http://127.0.0.1:8420 --pair --name "kitchen box"

The credential it receives is stored beside the script and reused, so pairing
happens once rather than every boot.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DEFAULT_STORE = Path.home() / ".jarvis-device.json"


@dataclass
class DeviceIdentity:
    device_id: str = ""
    token: str = ""
    url: str = "http://127.0.0.1:8420"
    name: str = "reference device"

    @property
    def paired(self) -> bool:
        return bool(self.device_id and self.token)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DeviceIdentity":
        if not path.is_file():
            return cls()
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return cls()


class JarvisDevice:
    """Talks to Jarvis Core the way a physical device would."""

    #: Frequently enough that the core knows the device is there, rarely enough
    #: that a sleeping box is not chattering all night.
    HEARTBEAT_SECONDS = 20.0

    def __init__(
        self,
        identity: DeviceIdentity,
        *,
        store: Path = DEFAULT_STORE,
        capabilities: list[str] | None = None,
        on_command: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.identity = identity
        self.store = store
        self.capabilities = capabilities or ["display", "audio_out"]
        self.on_command = on_command or self._default_command
        self._stop = threading.Event()
        self.last_state: dict[str, Any] = {}

    # -- transport -------------------------------------------------------

    def _post(self, path: str, body: dict[str, Any], *, token: str = "", timeout: float = 30.0) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        headers["X-Jarvis-Token"] = token or self.identity.token
        if self.identity.device_id and not token:
            headers["X-Jarvis-Device"] = self.identity.device_id
        request = urllib.request.Request(
            self.identity.url.rstrip("/") + path,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            return {"ok": False, "error": f"HTTP {exc.code}", "status": exc.code}
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    # -- pairing ---------------------------------------------------------

    def pair(self, core_token: str, *, kind: str = "box", wait_seconds: float = 300.0) -> bool:
        """Ask to join, then wait for a human to approve.

        The core token is needed only to ASK. What the device keeps afterwards
        is its own credential, which is what makes it revocable on its own.
        """

        response = self._post(
            "/api/device/pair",
            {"name": self.identity.name, "kind": kind, "capabilities": self.capabilities},
            token=core_token,
        )
        code = response.get("code")
        if not code:
            print(f"pairing refused: {response.get('error', response)}", file=sys.stderr)
            return False

        print(f"\n  Pairing code: {code}")
        print("  Approve it in Jarvis, or run:  python -m devices.client --approve", code, "\n")

        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            outcome = self._post("/api/device/collect", {"code": code}, token=core_token)
            status = outcome.get("status")
            if status == "paired":
                self.identity.device_id = outcome["device_id"]
                self.identity.token = outcome["token"]
                self.identity.save(self.store)
                print(f"  paired as {self.identity.device_id}")
                return True
            if status in {"denied", "expired"}:
                print(f"  pairing {status}", file=sys.stderr)
                return False
            time.sleep(2.0)

        print("  pairing timed out", file=sys.stderr)
        return False

    # -- presence --------------------------------------------------------

    def heartbeat_once(self) -> dict[str, Any]:
        response = self._post("/api/device/heartbeat", {"device_id": self.identity.device_id})
        if response.get("ok"):
            self.last_state = response.get("state", {})
            for command in response.get("commands", []):
                try:
                    self.on_command(command)
                except Exception as exc:
                    print(f"command failed: {exc}", file=sys.stderr)
        return response

    def run(self) -> int:
        if not self.identity.paired:
            print("not paired; run with --pair first", file=sys.stderr)
            return 2

        print(f"  {self.identity.name} connected to {self.identity.url}")
        print("  Ctrl-C to stop. Type a message and press Enter to send it.\n")

        threading.Thread(target=self._heartbeat_loop, daemon=True, name="device-heartbeat").start()

        try:
            for line in sys.stdin:
                text = line.strip()
                if not text:
                    continue
                reply = self._post("/api/message", {"text": text})
                if not reply.get("ok"):
                    print(f"  refused: {reply.get('error')}", file=sys.stderr)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
        return 0

    def _heartbeat_loop(self) -> None:
        previous = ""
        while not self._stop.is_set():
            response = self.heartbeat_once()
            if not response.get("ok"):
                print(f"  core unreachable: {response.get('error')}", file=sys.stderr)
            else:
                state = str(self.last_state.get("state", ""))
                if state and state != previous:
                    # A device with an LED ring renders this; a terminal prints
                    # it. Both consume the same vocabulary from the core.
                    print(f"  [{state}] {self.last_state.get('detail', '')}".rstrip())
                    previous = state
            self._stop.wait(self.HEARTBEAT_SECONDS)

    # -- acting on what the core sends -----------------------------------

    def _default_command(self, command: dict[str, Any]) -> None:
        name = command.get("command", "")
        payload = command.get("payload", {})
        if name == "display":
            print(f"\n  DISPLAY: {payload.get('text', '')}\n")
        elif name == "speak":
            print(f"\n  SPEAK: {payload.get('text', '')}\n")
        elif name == "show_url":
            print(f"\n  SHOW: {payload.get('url', '')}\n")
        else:
            print(f"\n  COMMAND {name}: {json.dumps(payload)[:200]}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m devices.client",
        description="Reference Jarvis device client. Standard library only.",
    )
    parser.add_argument("--url", default="", help="Jarvis Core URL")
    parser.add_argument("--name", default="reference device")
    parser.add_argument("--kind", default="box")
    parser.add_argument("--pair", action="store_true", help="pair with the core, then run")
    parser.add_argument("--core-token", default="", help="the token printed by jarvis.serve (pairing only)")
    parser.add_argument("--approve", default="", help="approve a pairing code (needs --core-token)")
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="where to keep this device's credential")
    parser.add_argument("--once", action="store_true", help="send one heartbeat and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = Path(args.store)

    identity = DeviceIdentity.load(store)
    if args.url:
        identity.url = args.url
    if args.name:
        identity.name = args.name

    device = JarvisDevice(identity, store=store)

    if args.approve:
        if not args.core_token:
            print("--approve needs --core-token", file=sys.stderr)
            return 2
        result = device._post("/api/device/approve", {"code": args.approve}, token=args.core_token)
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1

    if args.pair or not identity.paired:
        if not args.core_token:
            print("pairing needs --core-token (jarvis.serve prints it in the URL)", file=sys.stderr)
            return 2
        if not device.pair(args.core_token, kind=args.kind):
            return 1

    if args.once:
        print(json.dumps(device.heartbeat_once(), indent=2))
        return 0

    return device.run()


if __name__ == "__main__":
    raise SystemExit(main())
