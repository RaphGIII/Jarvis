"""SystemContext: the bounded facts ZEUS knows about its own installation.

"dein Repo", "deine Dateien", "dein Datenordner", "deine Modelle" must resolve
to REAL paths, deterministically, never guessed by FAST_LOCAL.  Every path is
derived from where this package actually lives plus the known state root, so it
is correct on any machine without hardcoding.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _package_root() -> Path:
    # service/system_context.py -> Jarvis/ (the package root)
    return Path(__file__).resolve().parent.parent


def _repo_root() -> Path:
    pkg = _package_root()
    # the git repository is the parent of the Jarvis package when it is named
    # "Jarvis" under a "repo" dir; otherwise the package root itself
    parent = pkg.parent
    if (parent / ".git").is_dir() or parent.name.lower() in {"repo", "jarvis_recovery"}:
        return parent
    if (pkg / ".git").is_dir():
        return pkg
    return parent


class SystemContext:
    """Deterministic self-knowledge; every value is a real path or a real fact."""

    def __init__(self, state_root: str | Path | None = None) -> None:
        self.package_root = _package_root()
        self.repo_root = _repo_root()
        self.state_root = Path(state_root) if state_root else (self.package_root / "data" / "jarvis")

    def facts(self) -> dict[str, str]:
        return {
            "repository_root": str(self.repo_root),
            "package_root": str(self.package_root),
            "data_root": str(self.state_root),
            "generated_media_root": os.environ.get("ZEUS_IMAGE_DIR", r"D:\ZEUS_Wissen\Bilder"),
            "model_root": os.environ.get("OLLAMA_MODELS", r"D:\JarvisLocal\ollama_models"),
            "library_root": r"D:\ZEUS_Wissen",
            "release_dir": str(self.repo_root / "dist" / "ZEUS"),
        }

    #: phrases that name a ZEUS-owned location, mapped to a facts() key
    _SELF_REF = (
        (re.compile(r"\b(dein(?:e[nmrs]?)?|unser(?:e[nmrs]?)?|your|our)\s+"
                    r"(repo(?:sitory)?|quell\s*code|quellcode|source\s*code|projektordner|projekt-ordner)\b", re.I), "repository_root"),
        (re.compile(r"\b(dein(?:e[nmrs]?)?|your)\s+(daten(?:ordner)?|data(?:\s*folder)?|zustands?ordner)\b", re.I), "data_root"),
        (re.compile(r"\b(dein(?:e[nmrs]?)?|your)\s+(modelle?|models?|modellordner)\b", re.I), "model_root"),
        (re.compile(r"\b(dein(?:e[nmrs]?)?|your)\s+(bilder|bilderordner|generierten?\s+bilder|generated\s+images?)\b", re.I), "generated_media_root"),
        (re.compile(r"\b(dein(?:e[nmrs]?)?|your)\s+(bibliothek|wissen|library)\b", re.I), "library_root"),
        (re.compile(r"\b(dein\s+eigenes?\s+repo|dein\s+repo|own\s+repo(?:sitory)?)\b", re.I), "repository_root"),
    )

    def resolve_self_reference(self, text: str) -> tuple[str, str]:
        """(facts-key, path) for a "dein X" phrase, or ("", "")."""

        for pattern, key in self._SELF_REF:
            if pattern.search(text or ""):
                return key, self.facts().get(key, "")
        return "", ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": True, **self.facts()}
