"""Where everything is: the repository, the interpreter, Ollama, the models.

The supervisor may run three ways -- ``python -m zeus_supervisor`` from the
repository, a frozen ``ZEUS.exe`` sitting inside the repository, or a frozen
``ZEUS.exe`` installed somewhere else with an ``install.json`` beside it -- and
all three must agree about where ZEUS lives.  Discovery is explicit and
recorded, so a wrong answer is visible in the diagnostics rather than a silent
launch of the wrong tree.

Nothing here imports the application.  The model names are read the same way
:mod:`brain.tiers` reads them (defaults, ``config/models.json``, environment)
but without importing it, because the supervisor must be able to diagnose a
repository whose imports are broken.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Mirrors ``brain.tiers.default_catalog`` for the two tiers the boot needs.
#: Kept in sync by ``tests/test_supervisor.py``.
DEFAULT_MODELS = {
    "FAST_LOCAL": "qwen3:4b-instruct",
    "BUILD_LOCAL": "qwen2.5-coder:7b-instruct-q4_K_M",
}
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_PORT = 8420

#: Directories that may hold an Ollama model store on this machine.  The user
#: environment can point ``OLLAMA_MODELS`` at a store that lacks the coder,
#: while a complete store sits next to it; the supervisor prefers whichever
#: store actually holds every required model.
OLLAMA_STORE_CANDIDATES = (
    os.environ.get("OLLAMA_MODELS", ""),
    r"D:\JarvisLocal\ollama_models",
    r"D:\OllamaModels",
    str(Path.home() / ".ollama" / "models"),
)


def _is_repo(path: Path) -> bool:
    return (path / "jarvis" / "serve.py").is_file() and (path / "service" / "core.py").is_file()


def find_repository(start: Path | None = None) -> Path | None:
    """The ``Jarvis`` package directory, found from an explicit hint, the
    install record, the frozen executable's location, or this file."""

    hint = os.environ.get("ZEUS_REPO", "").strip()
    if hint and _is_repo(Path(hint)):
        return Path(hint).resolve()

    origins: list[Path] = []
    if start is not None:
        origins.append(Path(start))
    if getattr(sys, "frozen", False):
        origins.append(Path(sys.executable).resolve().parent)
        record = Path(sys.executable).resolve().parent / "install.json"
        if record.is_file():
            try:
                data = json.loads(record.read_text(encoding="utf-8"))
                candidate = Path(str(data.get("repository", "")))
                if _is_repo(candidate):
                    return candidate.resolve()
            except (OSError, ValueError):
                pass
    origins.append(Path(__file__).resolve().parent)
    origins.append(Path.cwd())

    for origin in origins:
        for candidate in (origin, *origin.parents):
            if _is_repo(candidate):
                return candidate.resolve()
            if _is_repo(candidate / "Jarvis"):
                return (candidate / "Jarvis").resolve()
    return None


def find_python(repository: Path) -> str:
    """The interpreter that runs ZEUS.

    Recorded at build time in ``install.json`` when frozen; otherwise the
    running interpreter, which is the right answer for ``python -m
    zeus_supervisor``.  Falls back to whatever ``python`` is on PATH.
    """

    if getattr(sys, "frozen", False):
        record = Path(sys.executable).resolve().parent / "install.json"
        if record.is_file():
            try:
                data = json.loads(record.read_text(encoding="utf-8"))
                candidate = str(data.get("python", ""))
                if candidate and Path(candidate).is_file():
                    return candidate
            except (OSError, ValueError):
                pass
        for name in ("python3.14", "python", "py"):
            found = shutil.which(name)
            if found:
                return found
        return "python"
    return sys.executable


@dataclass
class SupervisorConfig:
    repository: Path
    python: str
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_models_dir: str = ""
    #: An explicit ``ollama`` binary (else PATH and the usual install dirs).
    ollama_exe: str = ""
    #: Seconds a freshly started ``ollama serve`` gets to answer /api/version.
    #: Measured on this machine: the Vulkan/CUDA device scan delays the bind
    #: by 5-20 s on a cold start.
    ollama_start_timeout: float = 45.0
    #: Storm guard for restarts of a dying Ollama: seconds between spawns and
    #: how many spawns one failure window may hold before it is reported FAILED.
    ollama_spawn_cooldown: float = 30.0
    ollama_max_spawns: int = 3
    #: How often the running supervisor looks whether Ollama is still there.
    ollama_watch_interval: float = 15.0
    #: How often a held supervisor re-runs its preflight to recover by itself.
    hold_retry_interval: float = 30.0
    #: Ollama versions known not to generate on this GPU.  Empty means none
    #: are known; the check is a real generation either way.
    ollama_incompatible_versions: tuple[str, ...] = ()
    ollama_min_version: str = ""
    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    #: Seconds the core gets to report READY after launch.  Cold start on this
    #: machine loads a 4B model (~30 s) and whisper (~50 s); 300 s is generous.
    ready_timeout: float = 300.0
    #: Seconds a real preflight generation may take, including model load.
    generation_timeout: float = 180.0
    #: Whether boot runs its own generation before launching the core.  Off:
    #: the core's READY is the generation that counts, and the window is on
    #: screen while the model loads.
    preflight_generation: bool = False
    #: Seconds to wait for a graceful stop before terminating.
    stop_timeout: float = 25.0
    #: Boot-loop guard: this many failed starts inside the window means hold.
    max_failures: int = 3
    failure_window: float = 600.0
    #: Whether to start the hands-free listener from the speech venv.
    voice: bool = True
    open_browser: bool = True

    @property
    def state_dir(self) -> Path:
        return self.repository / "data" / "jarvis" / "supervisor"

    @property
    def speech_python(self) -> Path | None:
        for candidate in (
            self.repository / ".venv-speech" / "Scripts" / "python.exe",
            self.repository / ".venv-speech" / "bin" / "python",
        ):
            if candidate.is_file():
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": str(self.repository),
            "python": self.python,
            "port": self.port,
            "ollama_url": self.ollama_url,
            "ollama_models_dir": self.ollama_models_dir,
            "ollama_exe": self.ollama_exe,
            "ollama_start_timeout": self.ollama_start_timeout,
            "models": dict(self.models),
            "voice": self.voice,
            "speech_python": str(self.speech_python or ""),
        }

    @classmethod
    def load(cls, repository: Path | None = None, **overrides: Any) -> "SupervisorConfig":
        repo = repository or find_repository()
        if repo is None:
            raise FileNotFoundError(
                "could not find the ZEUS repository (a directory containing jarvis/serve.py); "
                "set ZEUS_REPO or run from inside it"
            )
        config = cls(repository=repo, python=find_python(repo))
        config.models = _resolve_models(repo)

        file = repo / "config" / "supervisor.json"
        if file.is_file():
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if isinstance(data, dict):
                for key in ("port", "host", "ollama_url", "ollama_models_dir", "ollama_exe", "ready_timeout",
                            "generation_timeout", "stop_timeout", "max_failures", "failure_window",
                            "voice", "open_browser", "ollama_min_version", "preflight_generation",
                            "ollama_start_timeout", "ollama_spawn_cooldown", "ollama_max_spawns",
                            "ollama_watch_interval", "hold_retry_interval"):
                    if key in data:
                        setattr(config, key, type(getattr(config, key))(data[key]))
                if isinstance(data.get("ollama_incompatible_versions"), list):
                    config.ollama_incompatible_versions = tuple(str(v) for v in data["ollama_incompatible_versions"])
        # Environment overrides exist so the failure paths can be exercised
        # against the real executable without touching the owner's config.
        if os.environ.get("ZEUS_OLLAMA_URL", "").strip():
            config.ollama_url = os.environ["ZEUS_OLLAMA_URL"].strip()
        if os.environ.get("ZEUS_OLLAMA_EXE", "").strip():
            config.ollama_exe = os.environ["ZEUS_OLLAMA_EXE"].strip()
        if os.environ.get("ZEUS_OLLAMA_START_TIMEOUT", "").strip():
            try:
                config.ollama_start_timeout = float(os.environ["ZEUS_OLLAMA_START_TIMEOUT"])
            except ValueError:
                pass
        for key, value in overrides.items():
            if value is not None:
                setattr(config, key, value)
        if not config.ollama_models_dir:
            config.ollama_models_dir = discover_ollama_store(config.models.values())
        return config


def _resolve_models(repository: Path) -> dict[str, str]:
    """Defaults, then ``config/models.json``, then ``JARVIS_<TIER>_MODEL``."""

    models = dict(DEFAULT_MODELS)
    file = repository / "config" / "models.json"
    if file.is_file():
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if isinstance(data, dict):
            for tier in models:
                entry = data.get(tier) or data.get(tier.lower()) or {}
                if isinstance(entry, dict) and entry.get("model"):
                    models[tier] = str(entry["model"])
    for tier in models:
        override = os.environ.get(f"JARVIS_{tier}_MODEL", "").strip()
        if override:
            models[tier] = override
    return models


def _store_has(store: Path, model: str) -> bool:
    name, _, tag = model.partition(":")
    manifest = store / "manifests" / "registry.ollama.ai" / "library" / name / (tag or "latest")
    return manifest.is_file()


def discover_ollama_store(models: Any) -> str:
    """The model directory that holds every required model, if one does.

    Returns "" when none does, in which case Ollama's own default applies and
    the preflight will report exactly which model is missing from it.
    """

    required = [m for m in models if m]
    for candidate in OLLAMA_STORE_CANDIDATES:
        if not candidate:
            continue
        store = Path(candidate)
        if store.is_dir() and all(_store_has(store, m) for m in required):
            return str(store)
    return ""
