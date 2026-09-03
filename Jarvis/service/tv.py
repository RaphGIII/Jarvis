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


#: SSAP WebSocket ports a webOS TV listens on when network control is enabled.
SSAP_PORTS = (3000, 3001)
#: LG Electronics OUI prefixes (lowercased, no separators) — a secondary hint.
LG_OUI = frozenset({
    "00e091", "001e75", "001c62", "0019e3", "a816b2", "b81db4", "cc2d8c",
    "dc0b34", "e8f2e2", "c4366c", "202267", "38d40b", "700514", "b4e62a",
    "a06faa", "d013fd", "6cdd6c", "48594e", "8c3c4a", "c49a02", "a009ed",
    "34fcb9", "e85b5b", "58a2b5", "10683f", "2c598a", "88c9d0", "f80cf3",
})


def _local_ipv4() -> tuple[str, str]:
    """(the PC's LAN IP, its /24 prefix like '192.168.0.')."""

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip, ip.rsplit(".", 1)[0] + "."
    except OSError:
        return "", ""


def _tcp_open(ip: str, port: int, timeout: float = 0.6) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((ip, port)) == 0
    except OSError:
        return False
    finally:
        s.close()


def _arp_table() -> dict[str, str]:
    import subprocess

    try:
        raw = subprocess.run(["arp", "-a"], capture_output=True, timeout=8,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, Exception):  # noqa: BLE001
        return {}
    text = (raw or b"").decode("utf-8", "replace")
    out: dict[str, str] = {}
    for m in re.finditer(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F]{2}[-:]){5}[0-9a-fA-F]{2}", text):
        line = m.group(0)
        ip = line.split()[0]
        mac = re.search(r"([0-9a-fA-F]{2}[-:]){5}[0-9a-fA-F]{2}", line).group(0).replace("-", ":").lower()
        out[ip] = mac
    return out


def _ssdp(timeout: float = 3.0) -> dict[str, dict[str, str]]:
    """M-SEARCH bound to the LAN interface; often blocked by the host firewall."""

    self_ip, _ = _local_ipv4()
    found: dict[str, dict[str, str]] = {}
    for st in (SSDP_ST, "urn:dial-multiscreen-org:service:dial:1", "ssdp:all"):
        message = ("M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\n"
                   'MAN: "ssdp:discover"\r\nMX: 2\r\nST: ' + st + "\r\n\r\n").encode()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if self_ip:
                try:
                    sock.bind((self_ip, 0))
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(self_ip))
                except OSError:
                    pass
            sock.settimeout(timeout)
            sock.sendto(message, SSDP_ADDR)
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    data, addr = sock.recvfrom(4096)
                except socket.timeout:
                    break
                headers = {k.lower(): v.strip() for k, v in
                           (line.split(":", 1) for line in data.decode("utf-8", "replace").splitlines() if ":" in line)}
                found.setdefault(addr[0], {}).update({"server": headers.get("server", ""), "location": headers.get("location", "")})
            sock.close()
        except OSError:
            continue
    return found


def diagnostics(timeout: float = 4.0) -> dict[str, Any]:
    """A full, honest discovery report (§ show why each candidate was accepted/rejected)."""

    import concurrent.futures as cf

    self_ip, prefix = _local_ipv4()
    report: dict[str, Any] = {"self_ip": self_ip, "subnet": (prefix + "0/24") if prefix else "?",
                              "methods": [], "candidates": [], "rejected": []}
    if not prefix:
        report["error"] = "keine LAN-IPv4-Adresse gefunden"
        return report

    # method 1: SSDP
    ssdp = _ssdp(timeout=3.0)
    report["methods"].append({"ssdp": {"replies": len(ssdp), "note": "vom Host-Firewall oft blockiert" if not ssdp else ""}})

    # method 2: subnet SSAP-port sweep (the reliable one)
    hosts = [prefix + str(i) for i in range(1, 255) if prefix + str(i) != self_ip]
    ssap_open: dict[str, list[int]] = {}
    with cf.ThreadPoolExecutor(max_workers=80) as ex:
        futs = {ex.submit(_tcp_open, ip, p): (ip, p) for ip in hosts for p in SSAP_PORTS}
        for fut in cf.as_completed(futs):
            ip, p = futs[fut]
            if fut.result():
                ssap_open.setdefault(ip, []).append(p)
    report["methods"].append({"ssap_port_sweep": {"hosts_scanned": len(hosts), "ssap_hosts": len(ssap_open)}})

    # method 3: ARP table + OUI (populate it first with a light ping sweep)
    import subprocess

    with cf.ThreadPoolExecutor(max_workers=80) as ex:
        list(ex.map(lambda ip: subprocess.run(["ping", "-n", "1", "-w", "400", ip], stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)), hosts))
    arp = _arp_table()
    report["methods"].append({"arp": {"neighbours": len([ip for ip in arp if ip.startswith(prefix)])}})

    # build candidates: SSAP port open (strong) or LG OUI (secondary)
    seen = set()
    for ip in sorted(set(list(ssap_open) + list(ssdp) + [i for i in arp if i.startswith(prefix)]),
                     key=lambda x: int(x.split(".")[-1])):
        mac = arp.get(ip, "")
        oui = mac.replace(":", "")[:6]
        ssap = ssap_open.get(ip, [])
        server = ssdp.get(ip, {}).get("server", "")
        is_lg_mac = oui in LG_OUI
        webos = bool(ssap) or "webos" in server.lower() or is_lg_mac
        entry = {"ip": ip, "mac": mac, "ssap_ports": ssap, "ssdp_server": server,
                 "lg_oui": is_lg_mac, "webos_likely": webos}
        if webos:
            report["candidates"].append(entry)
        else:
            report["rejected"].append({**entry, "reason": "kein SSAP-Port, keine LG-Kennung"})
        seen.add(ip)
    report["found"] = len(report["candidates"])
    return report


def discover(timeout: float = 4.0) -> list[dict[str, Any]]:
    """webOS TV candidates on this LAN (SSAP port + LG OUI + SSDP).  [] = none."""

    diag = diagnostics(timeout=timeout)
    out = []
    for c in diag.get("candidates", []):
        out.append({"ip": c["ip"], "mac": c.get("mac", ""), "name": c.get("ssdp_server", "") or "webOS-Gerät",
                    "ssap": bool(c.get("ssap_ports")), "location": ""})
    return out


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
