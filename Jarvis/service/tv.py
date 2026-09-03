"""LG webOS TV control: real local protocol, no Bluetooth fiction.

Discovery is SSDP (the webOS second-screen service answers M-SEARCH on the
LAN); pairing is SSAP over websocket — the TV shows its own confirmation
prompt and hands back a client-key, which is stored under the owner state
directory and reused.  Power-on works via Wake-on-LAN when the TV's MAC was
captured at pairing time and the TV has network standby enabled — it is
reported as "attempted", never claimed as success without the TV answering.

"Zeig dich auf dem Fernseher" opens the ZEUS web UI (?tv=1 kiosk mode, which
already exists) in the TV's browser over the LAN — a remote Presence
display, not brittle screen mirroring.
"""

from __future__ import annotations

import json
import re
import socket
import ssl
import struct
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

SSDP_ADDR = ("239.255.255.250", 1900)
SSDP_ST = "urn:lge-com:service:webos-second-screen:1"
PAIR_TIMEOUT = 60.0

#: The standard SSAP registration manifest (public second-screen permissions).
_MANIFEST = {
    "manifestVersion": 1,
    "permissions": [
        "LAUNCH", "LAUNCH_WEBAPP", "APP_TO_APP", "CONTROL_AUDIO",
        "CONTROL_POWER", "READ_INSTALLED_APPS", "CONTROL_DISPLAY",
        "CONTROL_INPUT_JOYSTICK", "CONTROL_INPUT_MEDIA_PLAYBACK",
        "CONTROL_INPUT_TV", "READ_INPUT_DEVICE_LIST", "READ_TV_CHANNEL_LIST",
        "WRITE_NOTIFICATION_TOAST", "CONTROL_INPUT_TEXT",
        "CONTROL_MOUSE_AND_KEYBOARD", "READ_CURRENT_CHANNEL",
        "READ_RUNNING_APPS",
    ],
}


def discover(timeout: float = 4.0) -> list[dict[str, Any]]:
    """webOS TVs on this LAN, via SSDP M-SEARCH.  Empty list = none answered."""

    message = "\r\n".join([
        "M-SEARCH * HTTP/1.1", f"HOST: {SSDP_ADDR[0]}:{SSDP_ADDR[1]}",
        'MAN: "ssdp:discover"', "MX: 3", f"ST: {SSDP_ST}", "", ""]).encode()
    found: dict[str, dict[str, Any]] = {}
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.sendto(message, SSDP_ADDR)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                break
            headers = {k.lower(): v for k, v in
                       (line.split(":", 1) for line in data.decode("utf-8", "replace").splitlines() if ":" in line)}
            ip = addr[0]
            entry = {"ip": ip, "location": headers.get("location", "").strip(),
                     "server": headers.get("server", "").strip(), "name": ""}
            found[ip] = entry
        sock.close()
    except OSError as exc:
        return [{"error": f"SSDP fehlgeschlagen: {exc}"}]
    for entry in found.values():
        if entry["location"]:
            try:
                with urllib.request.urlopen(entry["location"], timeout=3) as response:
                    xml = response.read(20000).decode("utf-8", "replace")
                m = re.search(r"<friendlyName>(.*?)</friendlyName>", xml)
                if m:
                    entry["name"] = m.group(1)
            except OSError:
                pass
    return list(found.values())


class TVService:
    """One paired LG TV; the client-key persists under the owner state dir."""

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)
        self._lock = threading.Lock()

    # -- pairing state ---------------------------------------------------

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def status(self) -> dict[str, Any]:
        data = self._load()
        return {"ok": True, "paired": bool(data.get("client_key")), "ip": data.get("ip", ""),
                "name": data.get("name", ""), "mac": data.get("mac", "")}

    # -- SSAP ------------------------------------------------------------

    def _connect(self, ip: str):
        import websocket

        last_error = None
        for url, ssl_opts in ((f"wss://{ip}:3001", {"cert_reqs": ssl.CERT_NONE, "check_hostname": False}),
                              (f"ws://{ip}:3000", None)):
            try:
                return websocket.create_connection(url, timeout=8, sslopt=ssl_opts)
            except Exception as exc:  # noqa: BLE001 - try the older port next
                last_error = exc
        raise ConnectionError(f"TV {ip} nicht erreichbar: {last_error}")

    def _register(self, ws, client_key: str = "", *, wait: float = 8.0) -> dict[str, Any]:
        payload: dict[str, Any] = {"forcePairing": False, "pairingType": "PROMPT", "manifest": _MANIFEST}
        if client_key:
            payload["client-key"] = client_key
        ws.send(json.dumps({"type": "register", "id": "reg0", "payload": payload}))
        deadline = time.time() + wait
        while time.time() < deadline:
            message = json.loads(ws.recv())
            if message.get("type") == "registered":
                return {"ok": True, "client_key": message.get("payload", {}).get("client-key", client_key)}
            if message.get("type") == "error":
                return {"ok": False, "error": str(message.get("error", "registration failed"))}
            # "response" with pairingType PROMPT: the TV is showing the dialog
        return {"ok": False, "error": "keine Antwort auf die Registrierung"}

    def pair(self, ip: str, *, name: str = "") -> dict[str, Any]:
        """Start pairing: the TV shows its confirmation; we wait for the owner's OK."""

        with self._lock:
            try:
                ws = self._connect(ip)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
            try:
                result = self._register(ws, wait=PAIR_TIMEOUT)
                if not result.get("ok"):
                    return result
                mac = self._arp_mac(ip)
                self._save({"ip": ip, "name": name, "client_key": result["client_key"],
                            "mac": mac, "paired_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
                return {"ok": True, "ip": ip, "name": name, "mac": mac}
            finally:
                try:
                    ws.close()
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _arp_mac(ip: str) -> str:
        import subprocess

        try:
            out = subprocess.run(["arp", "-a", ip], capture_output=True, timeout=6,
                                 creationflags=getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0)).stdout
            m = re.search(rb"([0-9a-fA-F]{2}[-:]){5}[0-9a-fA-F]{2}", out or b"")
            return m.group(0).decode().replace("-", ":").lower() if m else ""
        except (OSError, Exception):  # noqa: BLE001
            return ""

    def request(self, uri: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = self._load()
        if not data.get("client_key"):
            return {"ok": False, "error": "kein TV gekoppelt — erst unter Owner → Geräte koppeln"}
        with self._lock:
            try:
                ws = self._connect(data["ip"])
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc), "unreachable": True}
            try:
                reg = self._register(ws, data["client_key"])
                if not reg.get("ok"):
                    return reg
                ws.send(json.dumps({"type": "request", "id": "req1", "uri": uri, "payload": payload or {}}))
                deadline = time.time() + 8
                while time.time() < deadline:
                    message = json.loads(ws.recv())
                    if message.get("id") == "req1":
                        ok = message.get("type") == "response" and message.get("payload", {}).get("returnValue", True)
                        return {"ok": bool(ok), "payload": message.get("payload", {})}
                return {"ok": False, "error": "keine Antwort vom TV"}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
            finally:
                try:
                    ws.close()
                except Exception:  # noqa: BLE001
                    pass

    # -- commands --------------------------------------------------------

    def power_off(self) -> dict[str, Any]:
        return self.request("ssap://system/turnOff")

    def power_on(self) -> dict[str, Any]:
        """Wake-on-LAN: attempted, and honestly reported as an attempt."""

        data = self._load()
        mac = str(data.get("mac", ""))
        if not re.fullmatch(r"([0-9a-f]{2}:){5}[0-9a-f]{2}", mac):
            return {"ok": False, "error": "keine MAC-Adresse bekannt — der TV muss einmal im laufenden Zustand gekoppelt sein"}
        packet = b"\xff" * 6 + bytes.fromhex(mac.replace(":", "")) * 16
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(packet, ("255.255.255.255", 9))
            sock.close()
        except OSError as exc:
            return {"ok": False, "error": f"WoL fehlgeschlagen: {exc}"}
        return {"ok": True, "attempted": True,
                "note": "Magic Packet gesendet — ob der TV angeht, hängt von dessen Netzwerk-Standby-Einstellung ab"}

    def volume_step(self, up: bool) -> dict[str, Any]:
        return self.request("ssap://audio/volumeUp" if up else "ssap://audio/volumeDown")

    def volume(self, level: int) -> dict[str, Any]:
        return self.request("ssap://audio/setVolume", {"volume": max(0, min(100, int(level)))})

    def mute(self, muted: bool) -> dict[str, Any]:
        return self.request("ssap://audio/setMute", {"mute": bool(muted)})

    def launch_app(self, app_id: str) -> dict[str, Any]:
        return self.request("ssap://system.launcher/launch", {"id": app_id})

    def open_url(self, url: str) -> dict[str, Any]:
        return self.request("ssap://system.launcher/open", {"target": url})

    def toast(self, message: str) -> dict[str, Any]:
        return self.request("ssap://system.notifications/createToast", {"message": str(message)[:120]})

    def show_zeus(self, zeus_url: str) -> dict[str, Any]:
        """The LAN remote display: the ZEUS ?tv=1 kiosk page in the TV browser."""

        url = zeus_url + ("&" if "?" in zeus_url else "?") + "tv=1"
        result = self.open_url(url)
        result["url"] = url
        return result
