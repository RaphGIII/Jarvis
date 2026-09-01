"""The Owner Security Gate: a manually typed password in front of the changes that matter.

Design constraints, all absolute:

* the password is verified **here**, in deterministic code — no model
  (FAST_LOCAL, BUILD_LOCAL, an expert) is ever shown it, asked about it, or
  able to claim it was verified;
* nothing plaintext is ever stored: only a verifier derived with **scrypt**
  (n=2^15, r=8, p=1 — a modern memory-hard KDF from the standard library)
  under a unique random salt; on Windows the verifier blob is additionally
  wrapped with DPAPI (``CryptProtectData``) so it is bound to this user
  account;
* a successful check issues a **short-lived, scoped** authorization token —
  "PERSONALITY_EDIT for 5 minutes", never "everything";
* everything locks on restart; the token store is memory only;
* the password never appears in logs, Activity, Knowledge or prompts — the
  API layer receives it, passes it in, and this module returns only booleans
  and scope tokens.

Levels (enforced by the callers through :meth:`required_level` /
:meth:`authorize`):

    0  SAFE            conversation, reads, media transport, opening folders
    1  NORMAL_WRITE    ordinary project/note creation, reversible edits
    2  IMPORTANT       protected personality, security policy, SelfDev
                       promotion, installs, broad/mass file changes,
                       permanent deletion, startup integration
    3  CRITICAL        disabling security, credential export, major data
                       deletion, changing this authentication itself
                       (always password + explicit confirmation)

Recovery: there are no security questions.  If the password is lost, the
owner — with local machine access — deletes ``auth.json`` from the state
directory and sets a new password; local filesystem access is the actual
trust anchor on a single-user machine, and pretending otherwise would only
add a weaker backdoor.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: scrypt work factors: ~32 MiB, tens of milliseconds on this machine.
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 15, 8, 1
KEY_LEN = 32

#: Operation scopes a token can carry.
SCOPES = ("PERSONALITY_EDIT", "SECURITY_CONFIG", "SELFDEV_PROMOTE", "INSTALL", "FILESYSTEM_DESTRUCTIVE",
          "PROJECT_DELETE", "CREDENTIALS", "SYSTEM_INTEGRATION", "AUTH_ADMIN")

#: Which level each scope sits at (3 = also needs explicit confirmation).
SCOPE_LEVELS = {
    "PERSONALITY_EDIT": 2, "SECURITY_CONFIG": 2, "SELFDEV_PROMOTE": 2, "INSTALL": 2,
    "FILESYSTEM_DESTRUCTIVE": 2, "PROJECT_DELETE": 2, "SYSTEM_INTEGRATION": 2,
    "CREDENTIALS": 3, "AUTH_ADMIN": 3,
}

DEFAULT_TOKEN_SECONDS = 300.0


def _dpapi_protect(blob: bytes) -> bytes | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        import ctypes.wintypes as wt

        class BLOB(ctypes.Structure):
            _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

        crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
        inp = BLOB(len(blob), ctypes.cast(ctypes.create_string_buffer(blob, len(blob)), ctypes.POINTER(ctypes.c_char)))
        out = BLOB()
        if not crypt32.CryptProtectData(ctypes.byref(inp), "ZEUS owner auth", None, None, None, 0, ctypes.byref(out)):
            return None
        try:
            return ctypes.string_at(out.pbData, out.cbData)
        finally:
            kernel32.LocalFree(out.pbData)
    except Exception:  # noqa: BLE001 - DPAPI is an extra layer, never a blocker
        return None


def _dpapi_unprotect(blob: bytes) -> bytes | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        import ctypes.wintypes as wt

        class BLOB(ctypes.Structure):
            _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

        crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
        inp = BLOB(len(blob), ctypes.cast(ctypes.create_string_buffer(blob, len(blob)), ctypes.POINTER(ctypes.c_char)))
        out = BLOB()
        if not crypt32.CryptUnprotectData(ctypes.byref(inp), None, None, None, None, 0, ctypes.byref(out)):
            return None
        try:
            return ctypes.string_at(out.pbData, out.cbData)
        finally:
            kernel32.LocalFree(out.pbData)
    except Exception:  # noqa: BLE001
        return None


@dataclass
class AuthToken:
    scope: str
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    issued_at: float = field(default_factory=time.monotonic)
    seconds: float = DEFAULT_TOKEN_SECONDS

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.issued_at >= self.seconds if self.seconds <= 0 else time.monotonic() - self.issued_at > self.seconds


class SecurityGate:
    """Password verifier + scoped, short-lived authorizations.  Memory-only tokens."""

    def __init__(self, path: str | Path, *, token_seconds: float = DEFAULT_TOKEN_SECONDS) -> None:
        self.path = Path(path)
        self.token_seconds = float(token_seconds)
        self._lock = threading.Lock()
        self._tokens: dict[str, AuthToken] = {}
        self._failures: list[float] = []

    # -- storage ---------------------------------------------------------

    def _read(self) -> dict[str, Any] | None:
        try:
            raw = self.path.read_bytes()
        except OSError:
            return None
        # DPAPI-wrapped (binary) or plain JSON (older / non-Windows)
        if raw[:1] != b"{":
            un = _dpapi_unprotect(raw)
            if un is None:
                return None
            raw = un
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def _write(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(record).encode("utf-8")
        protected = _dpapi_protect(raw)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(protected if protected is not None else raw)
        os.replace(tmp, self.path)

    # -- password lifecycle ---------------------------------------------

    @property
    def configured(self) -> bool:
        return self._read() is not None

    @staticmethod
    def _derive(password: str, salt: bytes) -> bytes:
        return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, maxmem=64 * 1024 * 1024, dklen=KEY_LEN)

    def setup(self, password: str, *, current: str = "") -> dict[str, Any]:
        """Set or change the password.  Changing requires the current one (AUTH_ADMIN territory)."""

        if not password or len(password) < 8:
            return {"ok": False, "error": "the password needs at least 8 characters"}
        if self.configured and not self.verify(current):
            return {"ok": False, "error": "the current password is wrong"}
        salt = secrets.token_bytes(16)
        record = {"kdf": "scrypt", "n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P,
                  "salt": base64.b64encode(salt).decode(), "verifier": base64.b64encode(self._derive(password, salt)).decode(),
                  "set_at": time.time(), "version": 1}
        self._write(record)
        with self._lock:
            self._tokens.clear()
        return {"ok": True, "configured": True}

    def verify(self, password: str) -> bool:
        """Constant-time check with a small failure backoff.  Never logs anything."""

        record = self._read()
        if record is None or not password:
            return False
        with self._lock:
            now = time.monotonic()
            self._failures = [t for t in self._failures if now - t < 60.0]
            if len(self._failures) >= 5:
                return False  # a minute of quiet after five wrong tries
        try:
            salt = base64.b64decode(record["salt"])
            wanted = base64.b64decode(record["verifier"])
            got = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(record.get("n", SCRYPT_N)), r=int(record.get("r", SCRYPT_R)),
                                 p=int(record.get("p", SCRYPT_P)), maxmem=64 * 1024 * 1024, dklen=len(wanted))
        except (KeyError, ValueError, TypeError):
            return False
        ok = hmac.compare_digest(wanted, got)
        if not ok:
            with self._lock:
                self._failures.append(time.monotonic())
        return ok

    # -- scoped authorization -------------------------------------------

    def unlock(self, password: str, scope: str, *, seconds: float | None = None) -> dict[str, Any]:
        """Password -> one scoped, short-lived token.  The only mint."""

        if scope not in SCOPES:
            return {"ok": False, "error": f"unknown scope {scope!r}"}
        if not self.configured:
            return {"ok": False, "error": "no owner password is set yet", "needs_setup": True}
        if not self.verify(password):
            return {"ok": False, "error": "wrong password"}
        token = AuthToken(scope=scope, seconds=float(self.token_seconds if seconds is None else seconds))
        with self._lock:
            self._tokens[token.token] = token
        return {"ok": True, "scope": scope, "authorization": token.token, "expires_in": token.seconds,
                "level": SCOPE_LEVELS.get(scope, 2)}

    def authorized(self, authorization: str, scope: str) -> bool:
        """Whether this token currently authorizes this exact scope.

        This is the only function executors may trust.  A model saying
        "password verified" is a string; this is a lookup in memory that only
        :meth:`unlock` can populate.
        """

        with self._lock:
            token = self._tokens.get(str(authorization or ""))
            if token is None or token.scope != scope:
                return False
            if token.expired:
                del self._tokens[token.token]
                return False
            return True

    def consume(self, authorization: str, scope: str) -> bool:
        """authorized() and burn the token (single-use for level-3 operations)."""

        with self._lock:
            token = self._tokens.get(str(authorization or ""))
            if token is None or token.scope != scope or token.expired:
                return False
            del self._tokens[token.token]
            return True

    def lock(self, scope: str = "") -> int:
        """Drop live tokens (all, or one scope).  Restart does this implicitly."""

        with self._lock:
            if not scope:
                n = len(self._tokens)
                self._tokens.clear()
                return n
            doomed = [k for k, t in self._tokens.items() if t.scope == scope]
            for k in doomed:
                del self._tokens[k]
            return len(doomed)

    def status(self) -> dict[str, Any]:
        with self._lock:
            live = [{"scope": t.scope, "expires_in": round(max(0.0, t.seconds - (time.monotonic() - t.issued_at)), 1)}
                    for t in self._tokens.values() if not t.expired]
        return {"configured": self.configured, "locked": not live, "sessions": live,
                "scopes": list(SCOPES), "levels": dict(SCOPE_LEVELS), "kdf": "scrypt",
                "storage": "dpapi+file" if sys.platform == "win32" else "file",
                "path": str(self.path)}
