"""What is true about this machine, determined once and remembered.

Every capability acquisition so far has begun by rediscovering the same
things: which Python is running, whether PowerShell exists, what Ollama has
pulled, whether Spotify is installed, which third-party packages can be
imported.  Four tool calls per INVESTIGATE phase, once per attempt, six
attempts for one capability -- all of it re-establishing facts that had not
changed since the last time anybody asked.

Worse than the cost is the failure mode.  Four of the six previous music
attempts failed on facts about the environment nobody had told the model:
packages that were not installed, ZEUS tools that are not importable from
generated source.  A capability brief that is *wrong* about the machine is
worse than one that says nothing, and a brief written by hand goes stale the
moment anything is installed.

So the facts are probed deterministically -- ``shutil.which``, an HTTP call to
Ollama, a directory listing, an import attempt -- never guessed by a model, and
cached with a fingerprint of the things that would make them wrong.  When the
fingerprint changes the cache is rebuilt; when it does not, an acquisition
starts already knowing where it is.

Every fact records *how* it was determined, so a brief can say "PowerShell is
at C:\\... (found with shutil.which)" rather than asserting it.  A fact whose
provenance is unrecorded is indistinguishable from a fact somebody made up,
which is the distinction this whole system is built around.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: How stale a cached picture may be before it is rebuilt even though the
#: fingerprint still matches.  A backstop, not the primary mechanism: the
#: fingerprint catches real change, and this catches change the fingerprint
#: does not think to look at.
TTL_SECONDS = 6 * 60 * 60

#: Executables worth knowing about, checked with ``shutil.which``.
EXECUTABLES = (
    "python", "py", "pip", "powershell", "pwsh", "git", "node", "npm",
    "ffmpeg", "curl", "tar", "where", "nvidia-smi", "ollama", "code", "rg",
)

#: Third-party packages a generated capability might reach for.  Checked by
#: import machinery rather than by reading a requirements file, because what
#: matters is whether ``import x`` works in the interpreter that will run it.
PACKAGES = (
    "numpy", "requests", "chess", "cv2", "PIL", "yaml", "bs4", "lxml",
    "pandas", "scipy", "matplotlib", "pygame", "pydub", "mutagen",
    "psutil", "pyautogui", "comtypes", "pycaw", "win32api", "torch",
    "winrt", "winsdk",
)

#: Applications worth knowing are present, and the URI scheme each registers.
#: Kept as data rather than as branches so adding one is a line, not a code path.
APPLICATIONS = (
    ("spotify", "Spotify.exe", "spotify"),
    ("vlc", "vlc.exe", "vlc"),
    ("chrome", "chrome.exe", ""),
    ("firefox", "firefox.exe", ""),
)


@dataclass(frozen=True)
class Fact:
    """One thing that is true, and how that was established."""

    key: str
    value: Any
    probe: str
    at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": self.value, "probe": self.probe, "at": self.at}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fact":
        return cls(key=str(data.get("key", "")), value=data.get("value"),
                   probe=str(data.get("probe", "")), at=str(data.get("at", "")))

    def describe(self) -> str:
        if isinstance(self.value, bool):
            body = "yes" if self.value else "no"
        elif isinstance(self.value, (list, tuple)):
            body = ", ".join(str(item) for item in self.value) or "(none)"
        else:
            body = str(self.value)
        return f"{self.key}: {body}"


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Probes.  Each returns a value and says how it got it.
# --------------------------------------------------------------------------

def probe_platform() -> list[Fact]:
    return [
        Fact("os", f"{platform.system()} {platform.release()}", "platform.system/release", _now()),
        Fact("machine", platform.machine(), "platform.machine", _now()),
        Fact("python.executable", sys.executable, "sys.executable", _now()),
        Fact("python.version", platform.python_version(), "platform.python_version", _now()),
    ]


def probe_executables() -> list[Fact]:
    found = {name: shutil.which(name) for name in EXECUTABLES}
    return [
        Fact("executables.available", sorted(k for k, v in found.items() if v), "shutil.which", _now()),
        Fact("executables.paths", {k: v for k, v in found.items() if v}, "shutil.which", _now()),
    ]


def probe_packages() -> list[Fact]:
    import importlib.util

    importable = []
    for name in PACKAGES:
        try:
            if importlib.util.find_spec(name) is not None:
                importable.append(name)
        except (ImportError, ValueError, ModuleNotFoundError):
            continue
    missing = [name for name in PACKAGES if name not in importable]
    return [
        Fact("packages.importable", sorted(importable), "importlib.util.find_spec", _now()),
        Fact("packages.absent", sorted(missing), "importlib.util.find_spec", _now()),
    ]


def probe_ollama() -> list[Fact]:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = sorted(str(item.get("name", "")) for item in payload.get("models", []))
        return [
            Fact("ollama.reachable", True, "GET /api/tags", _now()),
            Fact("ollama.models", models, "GET /api/tags", _now()),
        ]
    except Exception as exc:
        return [Fact("ollama.reachable", False, f"GET /api/tags failed: {type(exc).__name__}", _now())]


def probe_gpu() -> list[Fact]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return [Fact("gpu", "none detected", "nvidia-smi not on PATH", _now())]
    try:
        completed = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [Fact("gpu", f"unreadable: {type(exc).__name__}", "nvidia-smi", _now())]
    line = (completed.stdout or "").strip().splitlines()
    return [Fact("gpu", line[0].strip() if line else "unknown", "nvidia-smi --query-gpu", _now())]


def probe_applications() -> list[Fact]:
    facts: list[Fact] = []
    roots = [
        Path.home() / "AppData/Local/Microsoft/WindowsApps",
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
        Path.home() / "AppData/Roaming",
    ]
    for name, executable, scheme in APPLICATIONS:
        location = ""
        for root in roots:
            candidate = root / executable
            if candidate.is_file():
                location = str(candidate)
                break
            found = shutil.which(executable)
            if found:
                location = found
                break
        facts.append(Fact(f"app.{name}", location or "not found", "filesystem + shutil.which", _now()))
        if scheme and location:
            facts.append(
                Fact(f"uri_scheme.{scheme}", _scheme_registered(scheme),
                     "HKEY_CLASSES_ROOT lookup", _now())
            )
    return facts


def _scheme_registered(scheme: str) -> bool:
    """Whether a URI scheme has a handler, read from the registry."""

    try:
        import winreg
    except ImportError:
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, scheme) as key:
            value, _kind = winreg.QueryValueEx(key, "URL Protocol")
            return True
    except OSError:
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, scheme):
                return True
        except OSError:
            return False


def probe_repository() -> list[Fact]:
    root = Path(__file__).resolve().parent.parent
    packages = sorted(
        item.name for item in root.iterdir()
        if item.is_dir() and (item / "__init__.py").is_file() and not item.name.startswith(".")
    )
    return [
        Fact("repo.root", str(root), "__file__", _now()),
        Fact("repo.packages", packages, "directories containing __init__.py", _now()),
    ]


#: Every probe, by name.  Named so a caller can refresh one without the rest,
#: and so a slow or failing probe is identifiable rather than anonymous.
PROBES: dict[str, Callable[[], list[Fact]]] = {
    "platform": probe_platform,
    "executables": probe_executables,
    "packages": probe_packages,
    "ollama": probe_ollama,
    "gpu": probe_gpu,
    "applications": probe_applications,
    "repository": probe_repository,
}

#: Facts whose change means everything else may be wrong.  Cheap to compute,
#: which is the point: the fingerprint has to be affordable on every read or it
#: will not be checked on every read.
FINGERPRINT_KEYS = (
    "os", "python.version", "python.executable",
    "executables.available", "packages.importable", "ollama.models",
)


class EnvironmentCache:
    """A durable, fingerprinted picture of the machine."""

    def __init__(self, path: str | Path, *, ttl_seconds: float = TTL_SECONDS) -> None:
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._facts: dict[str, Fact] = {}
        self._probed_at = 0.0
        self._fingerprint = ""
        self.probe_seconds = 0.0

    # -- probing ---------------------------------------------------------

    def probe(self, only: tuple[str, ...] = ()) -> dict[str, Fact]:
        """Run the probes and replace the cache."""

        started = time.perf_counter()
        facts: dict[str, Fact] = {}
        for name, probe in PROBES.items():
            if only and name not in only:
                continue
            try:
                for fact in probe():
                    facts[fact.key] = fact
            except Exception as exc:
                # A probe that fails is a fact about the machine too, and
                # certainly better recorded than allowed to abort the rest.
                facts[f"probe.{name}.failed"] = Fact(
                    f"probe.{name}.failed", f"{type(exc).__name__}: {exc}", "probe raised", _now()
                )
        with self._lock:
            self._facts = facts
            self._probed_at = time.time()
            self._fingerprint = _fingerprint_of(facts)
            self.probe_seconds = time.perf_counter() - started
        self.save()
        return dict(facts)

    def facts(self, *, refresh: bool = False) -> dict[str, Fact]:
        """The current picture, re-probing only when it might be wrong."""

        if refresh:
            return self.probe()
        if not self._facts:
            self.load()
        if not self._facts:
            return self.probe()
        if time.time() - self._probed_at > self.ttl_seconds:
            return self.probe()
        # Cheap probes only: if the things most likely to have changed still
        # hash the same, the expensive ones are trusted.
        current = _fingerprint_of({
            fact.key: fact
            for name in ("platform", "executables", "packages", "ollama")
            for fact in PROBES[name]()
        })
        if current != self._fingerprint:
            return self.probe()
        return dict(self._facts)

    # -- persistence -----------------------------------------------------

    def save(self) -> Path:
        with self._lock:
            payload = {
                "probed_at": self._probed_at,
                "fingerprint": self._fingerprint,
                "facts": [fact.to_dict() for fact in self._facts.values()],
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tmp.replace(self.path)
        return self.path

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        with self._lock:
            self._facts = {
                fact.key: fact
                for fact in (Fact.from_dict(item) for item in payload.get("facts", []))
            }
            self._probed_at = float(payload.get("probed_at") or 0.0)
            self._fingerprint = str(payload.get("fingerprint") or "")

    # -- reporting -------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        fact = self.facts().get(key)
        return default if fact is None else fact.value

    def briefing(self) -> list[str]:
        """The facts as sentences, each carrying its provenance.

        Written for a capability brief.  A model reading "PowerShell is at
        C:\\... (shutil.which)" can tell the difference between something that
        was checked and something that was assumed; a bare assertion cannot.
        """

        facts = self.facts()
        lines: list[str] = []

        def add(key: str, template: str) -> None:
            fact = facts.get(key)
            if fact is not None:
                lines.append(template.format(value=_render(fact.value)) + f"  [{fact.probe}]")

        add("os", "This machine runs {value}.")
        add("gpu", "GPU: {value}.")
        add("python.executable", "The interpreter that will run generated code is {value}.")
        add("packages.importable", "Third-party packages that CAN be imported: {value}.")
        add("packages.absent", "Packages that CANNOT be imported and must not be used: {value}.")
        add("executables.available", "Executables on PATH: {value}.")
        add("ollama.models", "Local models available: {value}.")

        for key, fact in sorted(facts.items()):
            if key.startswith("app.") and fact.value != "not found":
                lines.append(f"{key[4:]} is installed at {fact.value}.  [{fact.probe}]")
            if key.startswith("uri_scheme.") and fact.value:
                lines.append(
                    f"The '{key[11:]}:' URI scheme has a registered handler, so handing such a "
                    f"URI to the shell opens it in that application.  [{fact.probe}]"
                )
        return lines

    def to_dict(self) -> dict[str, Any]:
        facts = self.facts()
        return {
            "fingerprint": self._fingerprint,
            "probed_at": self._probed_at,
            "age_seconds": round(time.time() - self._probed_at, 1) if self._probed_at else None,
            "probe_seconds": round(self.probe_seconds, 3),
            "facts": {key: fact.to_dict() for key, fact in facts.items()},
        }


def _render(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "(none)"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in sorted(value.items()))
    return str(value)


def _fingerprint_of(facts: dict[str, Fact]) -> str:
    material = json.dumps(
        {key: facts[key].value for key in FINGERPRINT_KEYS if key in facts},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    """``python -m runtime.environment [--refresh] [--json]``."""

    import argparse

    parser = argparse.ArgumentParser(prog="python -m runtime.environment")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--path", default="")
    args = parser.parse_args(argv)

    default = Path(__file__).resolve().parent.parent / "data" / "jarvis" / "environment.json"
    cache = EnvironmentCache(args.path or default)
    started = time.perf_counter()
    cache.facts(refresh=args.refresh)
    elapsed = time.perf_counter() - started

    if args.json:
        print(json.dumps(cache.to_dict(), indent=2, default=str))
    else:
        for line in cache.briefing():
            print(f"  {line}")
        print(f"\n  ({'probed' if args.refresh else 'read'} in {elapsed:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
