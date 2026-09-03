"""Owner creation defaults: where generated things go and what they are named.

One small JSON store under the owner state directory.  Every generator asks
here UNLESS the owner's request names an explicit path — the spoken word
always outranks the default.  Templates use {date} {time} {slug} {original}
{name} placeholders; unknown placeholders survive verbatim rather than
crashing a generation over a typo.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

KEYS = {
    "image_dir": r"D:\ZEUS_Wissen\Bilder",
    "image_name": "{date}_{time}_{slug}",
    "pdf_summary_dir": r"D:\ZEUS_Wissen\Zusammenfassungen",
    "pdf_summary_name": "{original}_summary",
    "knowledge_dir": r"D:\ZEUS_Wissen",
    "export_dir": r"D:\ZEUS_Wissen\Exporte",
    "screenshot_dir": r"D:\ZEUS_Wissen\Screenshots",
    "project_root": "",
}


class CreationDefaults:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._data is None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._data = {k: str(v) for k, v in (raw or {}).items() if k in KEYS}
            except (OSError, ValueError):
                self._data = {}
        return self._data

    def all(self) -> dict[str, str]:
        with self._lock:
            data = self._load()
            return {key: data.get(key, fallback) for key, fallback in KEYS.items()}

    def get(self, key: str) -> str:
        if key not in KEYS:
            raise KeyError(key)
        with self._lock:
            return self._load().get(key, KEYS[key])

    def set(self, key: str, value: str) -> dict[str, str]:
        if key not in KEYS:
            raise KeyError(key)
        with self._lock:
            data = self._load()
            value = str(value or "").strip()
            if value:
                data[key] = value
            else:
                data.pop(key, None)  # empty = back to the built-in default
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.all()

    @staticmethod
    def render(template: str, **values: str) -> str:
        out = str(template or "")
        defaults = {"date": time.strftime("%Y%m%d"), "time": time.strftime("%H%M%S")}
        for key, value in {**defaults, **values}.items():
            out = out.replace("{" + key + "}", str(value))
        return out
