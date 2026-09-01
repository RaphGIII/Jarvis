"""What has to be true before ZEUS is worth starting.

Each check is a fact with a remedy, not a boolean.  "Ollama is not running" and
"Ollama is running but cannot generate on this GPU" both stop the boot, and
they need different sentences on the screen -- the second one is the failure
that used to present as an eye stuck on THINKING for ever.

Order matters: a check that would be meaningless after an earlier failure is
skipped rather than reported as a second failure.

The only check that costs anything is the last one, a real generation on the
conversation model.  It is the only honest test of "can this machine answer",
and it doubles as the warm-up the core would have paid anyway.
"""

from __future__ import annotations

import json
import os
import shutil
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import SupervisorConfig


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    remedy: str = ""
    #: A failed optional check degrades rather than blocks.
    required: bool = True
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)
    revision: str = ""
    dirty: bool = False
    ollama_version: str = ""
    ollama_started_by_supervisor: bool = False
    ollama_pid: int = 0
    #: RUNNING | STARTING | UNAVAILABLE | FAILED | MISSING, with the reason.
    ollama_state: str = ""
    ollama_reason: str = ""

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.required)

    @property
    def blocker(self) -> Check | None:
        for check in self.checks:
            if check.required and not check.ok:
                return check
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "revision": self.revision,
            "dirty": self.dirty,
            "ollama_version": self.ollama_version,
            "ollama_state": self.ollama_state,
            "ollama_reason": self.ollama_reason,
            "checks": [c.to_dict() for c in self.checks],
        }


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 5.0) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.strip().lstrip("v").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def git(repository: Path, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return subprocess.run(
        ["git", *args], cwd=str(repository), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, env=env,
        creationflags=_no_window(),
    )


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


#: Checks whose answer cannot change while their inputs do not: the Python
#: interpreter, the Ollama binary, the speech venv.  Cached under a
#: fingerprint of those inputs, so a boot does not spawn a Python and read
#: a version string to learn what it learned yesterday.
STABLE_CHECKS = ("python", "ollama.binary", "speech")


class PreflightCache:
    """Stable check results, keyed by a fingerprint of what they depend on."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def fingerprint(self, config: SupervisorConfig) -> str:
        parts = []
        for candidate in (config.python, str(config.speech_python or ""), str(config.repository / "config" / "supervisor.json")):
            try:
                stat = Path(candidate).stat()
                parts.append(f"{candidate}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                parts.append(f"{candidate}:absent")
        import shutil as _shutil

        ollama = _shutil.which("ollama") or ""
        try:
            stat = Path(ollama).stat() if ollama else None
            parts.append(f"{ollama}:{stat.st_mtime_ns if stat else 'absent'}")
        except OSError:
            parts.append(f"{ollama}:unreadable")
        return "|".join(parts)

    def load(self, fingerprint: str) -> dict[str, Check]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if data.get("fingerprint") != fingerprint:
            return {}
        out = {}
        for row in data.get("checks", []):
            if row.get("name") in STABLE_CHECKS and row.get("ok"):
                out[row["name"]] = Check(row["name"], True, str(row.get("detail", "")), "", required=bool(row.get("required", True)))
        return out

    def save(self, fingerprint: str, checks: list[Check]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"fingerprint": fingerprint, "saved_at": time.time(),
                       "checks": [c.to_dict() for c in checks if c.name in STABLE_CHECKS and c.ok]}
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass


def find_ollama_exe(explicit: str = "") -> str:
    """The ``ollama`` binary: an explicit path, PATH, then the usual install dirs.  "" when absent."""

    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("ollama")
    if found:
        return found
    for candidate in (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
        Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama"),
    ):
        if candidate.is_file():
            return str(candidate)
    return ""


class Preflight:
    def __init__(self, config: SupervisorConfig, *, log=print, cache: PreflightCache | None = None, ollama: Any = None) -> None:
        self.config = config
        self.log = log
        self.cache = cache
        self._ollama = ollama
        #: Whether this run may start services (Ollama).  The supervisor's boot
        #: does; a diagnostics read must never spawn anything.
        self.start_services = True

    @property
    def ollama(self) -> Any:
        """The Ollama lifecycle object, shared with the supervisor when it passes one in."""

        if self._ollama is None:
            from .ollama import OllamaService

            self._ollama = OllamaService(
                url=self.config.ollama_url, exe_finder=lambda: find_ollama_exe(self.config.ollama_exe),
                models_dir=self.config.ollama_models_dir, start_timeout=self.config.ollama_start_timeout,
                spawn_cooldown=self.config.ollama_spawn_cooldown, max_spawns=self.config.ollama_max_spawns, log=self.log,
            )
        return self._ollama

    def run(self, *, generation: bool = True, start_services: bool = True) -> PreflightReport:
        """The checks, in dependency order.

        ``generation`` (a real answer out of FAST_LOCAL) is the expensive one,
        and it is exactly what the core proves again when it reports READY;
        at boot the supervisor therefore skips it and lets the core's own
        generation be the evidence, which is what puts the window on screen
        30 seconds earlier.  ``zeus check`` still runs it.

        ``start_services=False`` observes only: Ollama is reported as it is
        and never started (diagnostics, a status read).

        A check that raises is a failed check with the exception as its
        detail -- never an exception out of the boot.  The live product hung
        on exactly that: a cached binary check left a field unset, the server
        check raised, and the frozen crash handler waited in a message box.
        """

        self.start_services = start_services
        report = PreflightReport()
        steps = [
            self._check_python,
            self._check_repository,
            self._check_ollama_binary,
            self._check_ollama_server,
            self._check_ollama_version,
            self._check_models,
            self._check_speech,
        ]
        if generation:
            steps.append(self._check_generation)
        cached: dict[str, Check] = {}
        fingerprint = ""
        if self.cache is not None:
            fingerprint = self.cache.fingerprint(self.config)
            cached = self.cache.load(fingerprint)
        for step in steps:
            started = time.monotonic()
            name = step.__name__.removeprefix("_check_").replace("_", ".", 1) if step.__name__ != "_check_ollama_binary" else "ollama.binary"
            check = cached.get(name)
            if check is not None:
                detail = check.detail
                while detail.endswith(" (cached)"):
                    detail = detail[: -len(" (cached)")]
                check = Check(check.name, True, detail + " (cached)", "", required=check.required)
            else:
                try:
                    check = step(report)
                except Exception as exc:  # noqa: BLE001 - a broken check is a diagnosis, not a crash
                    check = Check(name, False, f"check failed: {type(exc).__name__}: {exc}",
                                  "This is a defect in the supervisor's preflight; the log has the details")
            check.seconds = round(time.monotonic() - started, 2)
            report.checks.append(check)
            self.log(f"  [{'ok' if check.ok else ('..' if not check.required else 'FAIL')}] {check.name}: {check.detail}")
            if check.required and not check.ok:
                break
        if self.cache is not None and report.ok:
            self.cache.save(fingerprint, report.checks)
        return report

    # -- checks --------------------------------------------------------

    def _check_python(self, report: PreflightReport) -> Check:
        python = self.config.python
        try:
            completed = subprocess.run(
                [python, "-c", "import sys; print(sys.version.split()[0])"],
                capture_output=True, text=True, timeout=30, creationflags=_no_window(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Check("python", False, f"{python}: {exc}", "Install Python 3.11+ and record its path in install.json")
        version = completed.stdout.strip()
        if completed.returncode != 0 or _version_tuple(version) < (3, 11):
            return Check("python", False, f"{python} reports {version or completed.stderr.strip()}",
                         "ZEUS needs Python 3.11 or newer")
        return Check("python", True, f"{version} at {python}")

    def _check_repository(self, report: PreflightReport) -> Check:
        repo = self.config.repository
        if not (repo / "jarvis" / "serve.py").is_file():
            return Check("repository", False, f"{repo} does not contain jarvis/serve.py",
                         "Point ZEUS_REPO or install.json at the Jarvis directory")
        head = git(repo, "rev-parse", "HEAD")
        if head.returncode != 0:
            return Check("repository", False, f"not a git repository: {head.stderr.strip()[:200]}",
                         "Known-good rollback needs git; restore the repository from a clone")
        report.revision = head.stdout.strip()
        status = git(repo, "status", "--porcelain")
        report.dirty = bool(status.stdout.strip())
        return Check("repository", True, f"{report.revision[:12]}{' (uncommitted changes)' if report.dirty else ''} at {repo}")

    def _check_ollama_binary(self, report: PreflightReport) -> Check:
        # The path is resolved again by whoever needs it (the service's
        # exe_finder), never stored as a side effect of this check: a cached
        # check must leave the later steps exactly as capable as a fresh one.
        found = find_ollama_exe(self.config.ollama_exe)
        if not found:
            return Check("ollama.binary", False, "ollama is not installed",
                         "Install Ollama from https://ollama.com/download and restart ZEUS")
        return Check("ollama.binary", True, found)

    def _server_up(self) -> str:
        return str(self.ollama.version() or "")

    #: What the owner is told for each Ollama state, when it is not RUNNING.
    OLLAMA_REMEDIES = {
        "MISSING": "Install Ollama from https://ollama.com/download and start ZEUS again",
        "STARTING": "Ollama is still starting; ZEUS keeps checking and continues by itself when it answers",
        "FAILED": "Run `ollama serve` in a terminal and read the error it prints; ZEUS retries by itself when it answers",
        "UNAVAILABLE": "Start Ollama (the tray app or `ollama serve`); ZEUS continues by itself when it answers",
    }

    def _check_ollama_server(self, report: PreflightReport) -> Check:
        """RUNNING is the only pass.  Starting is bounded; nothing here raises or blocks past the timeout."""

        service = self.ollama
        status = service.ensure() if self.start_services else service.status()
        report.ollama_state = status.state.value
        report.ollama_reason = status.reason
        if status.ok:
            report.ollama_version = status.version
            report.ollama_started_by_supervisor = bool(status.started_by_supervisor)
            report.ollama_pid = int(status.pid or 0)
            how = (f"started by ZEUS (pid {status.pid}), models from {self.config.ollama_models_dir or 'its default store'}"
                   if status.started_by_supervisor else "already running")
            return Check("ollama.server", True, f"RUNNING: version {status.version}, {how}")
        return Check("ollama.server", False, f"{status.state.value}: {status.reason}",
                     self.OLLAMA_REMEDIES.get(status.state.value, "Start Ollama and start ZEUS again"))

    def _check_ollama_version(self, report: PreflightReport) -> Check:
        version = report.ollama_version
        if version in self.config.ollama_incompatible_versions:
            return Check("ollama.version", False, f"{version} is on the incompatible list for this GPU",
                         "Install a version known to generate on this card; do not upgrade blindly")
        minimum = self.config.ollama_min_version
        if minimum and _version_tuple(version) < _version_tuple(minimum):
            return Check("ollama.version", False, f"{version} is older than the required {minimum}",
                         f"Install Ollama {minimum} or newer")
        return Check("ollama.version", True, version)

    def _check_models(self, report: PreflightReport) -> Check:
        try:
            tags = _http_json(self.config.ollama_url + "/api/tags", timeout=10)
        except Exception as exc:
            return Check("models", False, f"could not list models: {exc}", "Check that Ollama answers on " + self.config.ollama_url)
        available = {str(m.get("name", "")) for m in tags.get("models", [])}
        available |= {name.removesuffix(":latest") for name in available}
        missing = {tier: model for tier, model in self.config.models.items() if model not in available}
        if not missing:
            return Check("models", True, ", ".join(f"{t}={m}" for t, m in self.config.models.items()))
        remedy = "; ".join(f"ollama pull {m}" for m in missing.values())
        if self.config.ollama_models_dir:
            remedy = (f"the running Ollama serves a store without {', '.join(missing.values())}; "
                      f"a complete store is at {self.config.ollama_models_dir} -- stop Ollama and let ZEUS start it, "
                      f"or set OLLAMA_MODELS to that directory; otherwise {remedy}")
        # The conversation model is required; the coder only degrades
        # self-development, which the health report names separately.
        required = "FAST_LOCAL" in missing
        return Check("models", not required, f"missing: {missing}", remedy, required=True if required else False)

    def _check_speech(self, report: PreflightReport) -> Check:
        python = self.config.speech_python
        if python is None:
            return Check("speech", False, "no .venv-speech; voice is off", "See JARVIS_USER_GUIDE.md, 'Voice setup'", required=False)
        return Check("speech", True, str(python), required=False)

    def _check_generation(self, report: PreflightReport) -> Check:
        model = self.config.models["FAST_LOCAL"]
        started = time.monotonic()
        try:
            data = _http_json(
                self.config.ollama_url + "/api/generate",
                {"model": model, "prompt": "Reply with the single word: OK", "stream": False,
                 "options": {"num_predict": 8, "temperature": 0.0}, "keep_alive": "10m"},
                timeout=self.config.generation_timeout,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            return Check("generation", False, f"{model}: HTTP {exc.code}: {body}",
                         "Ollama accepted the model but cannot run it -- often a GPU/driver/version mismatch. "
                         f"Check `ollama serve` output; version {report.ollama_version}")
        except Exception as exc:
            return Check("generation", False, f"{model}: {type(exc).__name__}: {exc}",
                         f"No answer within {self.config.generation_timeout:.0f}s: Ollama {report.ollama_version} "
                         "may be incompatible with this GPU. Check `ollama ps` and `nvidia-smi`.")
        text = str(data.get("response", "")).strip()
        elapsed = time.monotonic() - started
        if not text:
            return Check("generation", False, f"{model} returned an empty answer in {elapsed:.1f}s",
                         "The model loaded but produced nothing; check Ollama's log for CUDA errors")
        return Check("generation", True, f"{model} answered {text[:20]!r} in {elapsed:.1f}s")
