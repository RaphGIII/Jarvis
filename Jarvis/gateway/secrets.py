"""Provider credentials, stored encrypted and never repeated.

The owner types an API key once into the settings UI.  From then on it lives
in one file, encrypted with the Windows Data Protection API for the current
user account (``CryptProtectData``), so neither another account on the machine
nor a copied file yields the key.  Off Windows the store falls back to a file
with owner-only permissions -- honest about being weaker, but never plain text
in a place anything else reads.

Three rules the rest of the gateway relies on:

* :meth:`CredentialStore.get` is the only way to read a secret, and only the
  transport layer calls it.  Nothing puts a key into a prompt, a log line, an
  event or an engineer's context.
* :func:`redact` scrubs every stored secret out of arbitrary text before that
  text is logged or shown.  Provider error bodies echo request URLs, and
  Gemini puts the key in the URL.
* Presence is reportable, value is not: :meth:`CredentialStore.status` says which
  slots are filled and the last four characters, nothing more.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

#: The credential slots the settings UI offers.  Keys are provider names as
#: used in ``config/providers.json``; the value is the human label.
SECRET_SLOTS: dict[str, str] = {
    "gemini": "Google / Gemini API key",
    "openai": "OpenAI API key",
    "anthropic": "Anthropic API key",
}

_ENTROPY = b"ZEUS.gateway.secrets.v1"


def _dpapi_available() -> bool:
    return sys.platform == "win32"


def _protect(plain: bytes) -> bytes:  # pragma: no cover - exercised on Windows only
    import ctypes
    import ctypes.wintypes as wt

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    blob_in = DATA_BLOB(len(plain), ctypes.cast(ctypes.create_string_buffer(plain, len(plain)), ctypes.POINTER(ctypes.c_char)))
    entropy = DATA_BLOB(len(_ENTROPY), ctypes.cast(ctypes.create_string_buffer(_ENTROPY, len(_ENTROPY)), ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    if not crypt32.CryptProtectData(ctypes.byref(blob_in), None, ctypes.byref(entropy), None, None, 0, ctypes.byref(blob_out)):
        raise OSError("CryptProtectData failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def _unprotect(cipher: bytes) -> bytes:  # pragma: no cover - exercised on Windows only
    import ctypes
    import ctypes.wintypes as wt

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    blob_in = DATA_BLOB(len(cipher), ctypes.cast(ctypes.create_string_buffer(cipher, len(cipher)), ctypes.POINTER(ctypes.c_char)))
    entropy = DATA_BLOB(len(_ENTROPY), ctypes.cast(ctypes.create_string_buffer(_ENTROPY, len(_ENTROPY)), ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    if not crypt32.CryptUnprotectData(ctypes.byref(blob_in), None, ctypes.byref(entropy), None, None, 0, ctypes.byref(blob_out)):
        raise OSError("CryptUnprotectData failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


class CredentialStore:
    """Encrypted-at-rest credential slots.

    The on-disk format is one JSON document ``{"scheme": ..., "slots": {name:
    base64}}``.  ``scheme`` is ``dpapi`` on Windows and ``plain-0600`` elsewhere;
    a file written under one scheme is refused under the other rather than
    decoded wrongly.
    """

    def __init__(self, path: str | Path, *, use_dpapi: bool | None = None) -> None:
        self.path = Path(path)
        self._use_dpapi = _dpapi_available() if use_dpapi is None else use_dpapi
        self._lock = threading.RLock()
        self._cache: dict[str, str] | None = None

    # -- persistence -------------------------------------------------------

    @property
    def scheme(self) -> str:
        return "dpapi" if self._use_dpapi else "plain-0600"

    def _encode(self, value: str) -> str:
        raw = value.encode("utf-8")
        payload = _protect(raw) if self._use_dpapi else raw
        return base64.b64encode(payload).decode("ascii")

    def _decode(self, value: str) -> str:
        payload = base64.b64decode(value.encode("ascii"))
        raw = _unprotect(payload) if self._use_dpapi else payload
        return raw.decode("utf-8")

    def _load(self) -> dict[str, str]:
        with self._lock:
            if self._cache is not None:
                return self._cache
            slots: dict[str, str] = {}
            if self.path.is_file():
                try:
                    document = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    document = {}
                if isinstance(document, dict) and document.get("scheme") == self.scheme:
                    for name, encoded in (document.get("slots") or {}).items():
                        try:
                            slots[str(name)] = self._decode(str(encoded))
                        except (OSError, ValueError):
                            # A slot that no longer decrypts (different user
                            # account, damaged file) is treated as empty; the
                            # owner re-enters it.  Never as an error that
                            # blocks startup.
                            continue
            self._cache = slots
            return slots

    def _save(self, slots: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {"scheme": self.scheme, "slots": {name: self._encode(value) for name, value in slots.items()}}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass
        tmp.replace(self.path)
        self._cache = dict(slots)

    # -- the API -------------------------------------------------------------

    def set(self, name: str, value: str) -> None:
        name = str(name).strip()
        if name not in SECRET_SLOTS:
            raise KeyError(f"unknown secret slot: {name!r}")
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("an empty value is not a credential; use clear() to remove one")
        with self._lock:
            slots = dict(self._load())
            slots[name] = cleaned
            self._save(slots)

    def clear(self, name: str) -> bool:
        with self._lock:
            slots = dict(self._load())
            existed = slots.pop(str(name), None) is not None
            self._save(slots)
            return existed

    def get(self, name: str) -> str:
        """The secret itself.  Only the transport layer has business calling this."""

        return self._load().get(str(name), "")

    def has(self, name: str) -> bool:
        return bool(self.get(name))

    def status(self) -> dict[str, dict[str, Any]]:
        """What the settings UI may know: filled or not, and a four-character hint."""

        slots = self._load()
        out: dict[str, dict[str, Any]] = {}
        for name, label in SECRET_SLOTS.items():
            value = slots.get(name, "")
            out[name] = {
                "label": label,
                "configured": bool(value),
                "hint": ("…" + value[-4:]) if len(value) >= 8 else ("configured" if value else ""),
                "scheme": self.scheme,
            }
        return out

    def values(self) -> list[str]:
        """Every stored secret, for :func:`redact`.  Not for anything else."""

        return [value for value in self._load().values() if value]


_KEY_SHAPES = [
    # Provider key shapes, so a key that arrived via the environment rather
    # than the store is still scrubbed from anything we print.
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|x-goog-api-key|authorization)([=: ]+)(bearer )?([A-Za-z0-9_\-\.]{12,})"),
]


def redact(text: str, secrets: list[str] | CredentialStore | None = None) -> str:
    """Scrub stored secrets and anything key-shaped out of ``text``."""

    out = str(text)
    values = secrets.values() if isinstance(secrets, CredentialStore) else list(secrets or [])
    for value in sorted(values, key=len, reverse=True):
        if value:
            out = out.replace(value, "[secret]")
    for pattern in _KEY_SHAPES[:3]:
        out = pattern.sub("[secret]", out)
    out = _KEY_SHAPES[3].sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3) or ''}[secret]", out)
    return out
