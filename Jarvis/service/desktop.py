"""The desktop window as something the core owns, shows, hides and finds again.

Before this module the window was a Chromium ``--app`` process the core fired
off at startup and forgot: closing it with X left the core running with no way
back, a restart opened a second one, and a second ``ZEUS.exe`` did nothing
visible.  The engine is unchanged -- a Chromium app window is still the
smallest robust shell on a machine where adding packages is not on the table
-- but the window is now a *managed* thing:

* it is found by what it is (a visible top-level Chromium window whose title
  is exactly the product name -- a browser tab of the same page carries the
  browser's name in its title, an app window does not), never by a remembered
  pid, because Chromium hands a second launch to the running browser process
  and the pid the core recorded exits at once;
* ``show()`` focuses the window that exists or launches one, and measures how
  long until it is visible; ``hide()`` hides it without ending the process, so
  the next ``show()`` is instant; X (the owner closing it) ends the engine
  process and the core keeps running -- that is the "Iron Man" case, and the
  next ``show()`` relaunches in about a second;
* a restart of the core reuses the window that survived it instead of adding
  another (the page reconnects its event stream on its own);
* a second ``ZEUS.exe`` reaches this through ``/api/window/show`` or, while
  the core is still booting, through a beacon file the window watcher polls;
* the window carries its own Windows identity: an AppUserModelID of its own
  (its own taskbar group, not the browser's) and ZEUS's icon, set on the
  window handle after it appears.

Everything Win32 is ``ctypes`` against user32/shell32/ole32 and degrades to
"not available" elsewhere; nothing here imports the service.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jarvis.window import DEFAULT_SIZE, default_profile_dir, find_engine, window_command

APP_USER_MODEL_ID = "ZEUS.Desktop"
BEACON_NAME = "window-show"


# --------------------------------------------------------------------------
# Win32, guarded
# --------------------------------------------------------------------------

def _win32() -> Any:
    if sys.platform != "win32":
        return None
    import ctypes
    import ctypes.wintypes as wt

    user32 = ctypes.windll.user32
    user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM), wt.LPARAM]
    user32.GetWindowTextLengthW.argtypes = [wt.HWND]
    user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
    user32.IsWindowVisible.argtypes = [wt.HWND]
    user32.IsWindow.argtypes = [wt.HWND]
    user32.IsIconic.argtypes = [wt.HWND]
    user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wt.HWND]
    user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    user32.SendMessageW.argtypes = [wt.HWND, ctypes.c_uint, wt.WPARAM, wt.LPARAM]
    user32.LoadImageW.argtypes = [wt.HINSTANCE, wt.LPCWSTR, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    user32.LoadImageW.restype = wt.HANDLE
    user32.GetWindow.argtypes = [wt.HWND, ctypes.c_uint]
    user32.GetWindow.restype = wt.HWND
    user32.GetWindowRect.argtypes = [wt.HWND, ctypes.c_void_p]
    return ctypes, wt, user32


@dataclass(frozen=True)
class FoundWindow:
    hwnd: int
    pid: int
    title: str
    visible: bool
    minimized: bool


def find_windows(title: str, *, class_prefix: str = "Chrome_WidgetWin", include_hidden: bool = True) -> list[FoundWindow]:
    """Top-level Chromium windows titled exactly ``title``."""

    w = _win32()
    if w is None:
        return []
    ctypes, wt, user32 = w
    found: list[FoundWindow] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def visit(hwnd, _lparam):
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if buffer.value != title:
            return True
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls, 64)
        if not cls.value.startswith(class_prefix):
            return True
        visible = bool(user32.IsWindowVisible(hwnd))
        if not visible and not include_hidden:
            return True
        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        found.append(FoundWindow(int(hwnd), int(pid.value), buffer.value, visible, bool(user32.IsIconic(hwnd))))
        return True

    user32.EnumWindows(visit, 0)
    return found


def find_windows_of(pids: set[int], *, class_prefix: str = "Chrome_WidgetWin", min_width: int = 400) -> list[FoundWindow]:
    """Top-level main windows of the given processes, whatever their title says.

    A window mid-navigation has no title for a moment; a window that belongs
    to *our* profile's engine process is ours regardless.  Popups and tool
    windows are excluded by size and by having an owner.
    """

    w = _win32()
    if w is None or not pids:
        return []
    ctypes, wt, user32 = w
    found: list[FoundWindow] = []

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def visit(hwnd, _lparam):
        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if int(pid.value) not in pids:
            return True
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls, 64)
        if not cls.value.startswith(class_prefix):
            return True
        if user32.GetWindow(hwnd, 4):  # GW_OWNER: owned windows are popups
            return True
        rect = RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        if rect.right - rect.left < min_width:
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        found.append(FoundWindow(int(hwnd), int(pid.value), buffer.value, bool(user32.IsWindowVisible(hwnd)),
                                 bool(user32.IsIconic(hwnd))))
        return True

    user32.EnumWindows(visit, 0)
    return found


def focus(hwnd: int) -> bool:
    w = _win32()
    if w is None:
        return False
    ctypes, wt, user32 = w
    if not user32.IsWindow(hwnd):
        return False
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    else:
        user32.ShowWindow(hwnd, 5)  # SW_SHOW
    user32.SetForegroundWindow(hwnd)
    return True


def style_frameless(hwnd: int) -> bool:
    """Native borderless + maximized (to the work area), WITHOUT browser fullscreen.

    Removes WS_CAPTION | WS_THICKFRAME so there is no Windows title bar and no
    resize frame, then maximizes.  Because the window keeps WS_OVERLAPPED (not
    WS_POPUP) it maximizes to the *work area* -- the taskbar stays -- and Edge
    never enters its Fullscreen mode, so the "Vollbildmodus beenden" toast that
    --start-fullscreen produced is gone.  The page draws its own top bar.
    """

    if sys.platform != "win32" or not hwnd:
        return False
    w = _win32()
    if w is None:
        return False
    ctypes, wt, user32 = w
    if not user32.IsWindow(hwnd):
        return False
    GWL_STYLE = -16
    WS_CAPTION = 0x00C00000
    WS_THICKFRAME = 0x00040000
    SWP_FRAMECHANGED, SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER = 0x0020, 0x0002, 0x0001, 0x0004
    try:
        user32.GetWindowLongW.argtypes = [wt.HWND, ctypes.c_int]
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.argtypes = [wt.HWND, ctypes.c_int, ctypes.c_long]
        user32.SetWindowLongW.restype = ctypes.c_long
        user32.SetWindowPos.argtypes = [wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        style = user32.GetWindowLongW(hwnd, GWL_STYLE)
        user32.SetWindowLongW(hwnd, GWL_STYLE, style & ~(WS_CAPTION | WS_THICKFRAME))
        user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER)
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE first
        user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE (to work area, since not WS_POPUP)
        return True
    except Exception:  # noqa: BLE001 - styling is cosmetic; never break the launch
        return False


def minimize_window(hwnd: int) -> bool:
    """Minimize to the taskbar -- works on fullscreen windows too."""
    if sys.platform != "win32" or not hwnd:
        return False
    user32 = _user32()
    user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
    return True


def hide_window(hwnd: int) -> bool:
    w = _win32()
    if w is None:
        return False
    _ctypes, _wt, user32 = w
    if not user32.IsWindow(hwnd):
        return False
    user32.ShowWindow(hwnd, 0)  # SW_HIDE
    return True


def apply_identity(hwnd: int, *, app_id: str = APP_USER_MODEL_ID, icon: Path | None = None) -> dict[str, Any]:
    """Give the window its own taskbar identity and icon.

    ``SHGetPropertyStoreForWindow`` + ``PKEY_AppUserModel_ID`` is what Windows
    groups taskbar buttons by; without it the window is one of the browser's.
    The icon is sent with ``WM_SETICON`` for both sizes.
    """

    report: dict[str, Any] = {"app_id": False, "icon": False}
    w = _win32()
    if w is None:
        return report
    ctypes, wt, user32 = w
    try:
        import ctypes.wintypes as _wt
        from ctypes import POINTER, Structure, byref, c_ulong, c_ushort, c_ubyte, c_void_p, c_wchar_p

        class GUID(Structure):
            _fields_ = [("Data1", c_ulong), ("Data2", c_ushort), ("Data3", c_ushort), ("Data4", c_ubyte * 8)]

        class PROPERTYKEY(Structure):
            _fields_ = [("fmtid", GUID), ("pid", c_ulong)]

        class PROPVARIANT(Structure):
            _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort), ("r3", c_ushort), ("pwszVal", c_wchar_p), ("pad", c_void_p)]

        # PKEY_AppUserModel_ID = {9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}, 5
        fmtid = GUID(0x9F4C2855, 0x9F79, 0x4B39, (c_ubyte * 8)(0xA8, 0xD0, 0xE1, 0xD4, 0x2D, 0xE1, 0xD5, 0xF3))
        key = PROPERTYKEY(fmtid, 5)
        # IID_IPropertyStore = {886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}
        iid = GUID(0x886D8EEB, 0x8CF2, 0x4446, (c_ubyte * 8)(0x8D, 0x02, 0xCD, 0xBA, 0x1D, 0xBD, 0xCF, 0x99))
        store = c_void_p()
        shell32 = ctypes.windll.shell32
        hr = shell32.SHGetPropertyStoreForWindow(wt.HWND(hwnd), byref(iid), byref(store))
        if hr == 0 and store.value:
            # IPropertyStore vtable: QueryInterface, AddRef, Release, GetCount, GetAt, GetValue, SetValue, Commit
            vtable = ctypes.cast(ctypes.cast(store, POINTER(c_void_p))[0], POINTER(c_void_p))
            set_value = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, POINTER(PROPERTYKEY), POINTER(PROPVARIANT))(vtable[6])
            commit = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p)(vtable[7])
            release = ctypes.WINFUNCTYPE(c_ulong, c_void_p)(vtable[2])
            value = PROPVARIANT()
            value.vt = 31  # VT_LPWSTR
            value.pwszVal = app_id
            ok = set_value(store, byref(key), byref(value)) == 0 and commit(store) == 0
            release(store)
            report["app_id"] = bool(ok)
    except Exception as exc:  # noqa: BLE001 - identity is cosmetic; never fail the window for it
        report["app_id_error"] = str(exc)[:200]
    if icon is not None and Path(icon).is_file():
        try:
            for size, which in ((16, 0), (32, 1)):
                handle = user32.LoadImageW(None, str(icon), 1, size, size, 0x0010)  # IMAGE_ICON, LR_LOADFROMFILE
                if handle:
                    user32.SendMessageW(hwnd, 0x0080, which, handle)  # WM_SETICON
                    report["icon"] = True
        except Exception as exc:  # noqa: BLE001
            report["icon_error"] = str(exc)[:200]
    return report


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

@dataclass
class DesktopWindow:
    url: str
    title: str
    state_root: Path
    icon: Path | None = None
    size: tuple[int, int] = DEFAULT_SIZE
    emit: Callable[[str, dict[str, Any]], None] | None = None
    profile_dir: Path | None = None
    hwnd: int = 0
    engine: str = ""
    last_shown_at: float = 0.0
    last_show_seconds: float = 0.0
    identity: dict[str, Any] = field(default_factory=dict)
    _watcher: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.state_root = Path(self.state_root)
        self.profile_dir = Path(self.profile_dir) if self.profile_dir else default_profile_dir()
        self.engine = find_engine()

    # -- files -----------------------------------------------------------

    @property
    def session_path(self) -> Path:
        return self.state_root / "window" / "session.json"

    @property
    def beacon_path(self) -> Path:
        return self.state_root / "supervisor" / "control" / BEACON_NAME

    @staticmethod
    def beacon_for(state_root: Path) -> Path:
        return Path(state_root) / "supervisor" / "control" / BEACON_NAME

    def _write_session(self, **extra: Any) -> None:
        try:
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"url": self.url, "title": self.title, "hwnd": self.hwnd, "engine": self.engine,
                       "profile_dir": str(self.profile_dir), "core_pid": os.getpid(), "at": time.time(), **extra}
            tmp = self.session_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.session_path)
        except OSError:
            pass

    # -- discovery -------------------------------------------------------

    def profile_pids(self) -> set[int]:
        """Engine processes started with our profile directory."""

        try:
            from service.processes import list_processes

            marker = str(self.profile_dir).lower()
            return {p.pid for p in list_processes([marker]) if marker in p.command.lower()
                    and p.name.lower() in {"msedge.exe", "chrome.exe", "brave.exe", "vivaldi.exe", "chromium.exe"}}
        except Exception:  # noqa: BLE001
            return set()

    def find(self, *, include_hidden: bool = True) -> FoundWindow | None:
        windows = find_windows(self.title, include_hidden=include_hidden)
        if not windows:
            # No window carries the title right now -- it may be navigating
            # (the status page handing over to the interface).  A *visible*
            # main window of our own profile's engine process is ours
            # regardless; the engine's hidden helper windows are not.
            windows = [w for w in find_windows_of(self.profile_pids()) if w.visible]
        if not windows:
            return None
        # Prefer the one we already know, then a visible one.
        for w in windows:
            if w.hwnd == self.hwnd:
                return w
        windows.sort(key=lambda w: (not w.visible, w.hwnd))
        return windows[0]

    def status(self) -> dict[str, Any]:
        found = self.find()
        extra = find_windows(self.title)
        return {
            "available": bool(self.engine) and sys.platform == "win32",
            "engine": Path(self.engine).name if self.engine else "",
            "exists": found is not None,
            "visible": bool(found and found.visible and not found.minimized),
            "minimized": bool(found and found.minimized),
            "hwnd": found.hwnd if found else 0,
            "pid": found.pid if found else 0,
            "windows": len(extra),
            "last_show_seconds": self.last_show_seconds,
            "identity": self.identity,
            "app_id": APP_USER_MODEL_ID,
        }

    # -- actions ---------------------------------------------------------

    def show(self, *, reason: str = "") -> dict[str, Any]:
        """Focus the window that exists, or open one.  Reports the time to visible."""

        with self._lock:
            started = time.perf_counter()
            found = self.find()
            action = ""
            if found is not None:
                self.hwnd = found.hwnd
                focus(found.hwnd)
                action = "focused" if found.visible and not found.minimized else "restored"
                if not self.identity:
                    # A window the supervisor opened at launch, or one that
                    # survived a restart: give it its identity now.
                    self.identity = apply_identity(found.hwnd, icon=self.icon)
            else:
                launched = self._launch()
                if not launched.get("ok"):
                    return {"ok": False, "action": "launch_failed", "detail": launched.get("detail", ""), "reason": reason}
                action = "launched"
                found = self._wait_for_window(timeout=8.0)
                if found is not None:
                    self.hwnd = found.hwnd
                    focus(found.hwnd)
                    self.identity = apply_identity(found.hwnd, icon=self.icon)
            # native borderless + maximized, unless the owner chose an explicit
            # immersive/windowed mode.  Idempotent, so focusing an already-styled
            # window costs nothing.
            mode = os.getenv("ZEUS_WINDOW_MODE", "").strip().lower() or "borderless"
            if self.hwnd and mode in {"borderless", "maximized"}:
                style_frameless(self.hwnd)
            self.last_show_seconds = round(time.perf_counter() - started, 3)
            self.last_shown_at = time.time()
            self._ensure_single(keep=self.hwnd)
            self._write_session(action=action, reason=reason, seconds=self.last_show_seconds)
            result = {"ok": found is not None, "action": action, "seconds": self.last_show_seconds,
                      "hwnd": self.hwnd, "reason": reason, "identity": self.identity}
            if self.emit is not None:
                self.emit("tool", {"summary": f"window {action} in {self.last_show_seconds}s" + (f" ({reason})" if reason else ""),
                                   "source": "desktop", "window": result})
            return result

    def minimize(self, *, reason: str = "") -> dict[str, Any]:
        """Minimize to the taskbar; the eye keeps running, voice stays armed."""

        with self._lock:
            found = self.find()
            if found is None:
                return {"ok": True, "action": "absent", "reason": reason}
            ok = minimize_window(found.hwnd)
            self.hwnd = found.hwnd
            return {"ok": ok, "action": "minimized", "hwnd": found.hwnd, "reason": reason}

    def hide(self, *, reason: str = "") -> dict[str, Any]:
        """Hide without ending the engine; the next show is instant."""

        with self._lock:
            found = self.find()
            if found is None:
                return {"ok": True, "action": "absent", "reason": reason}
            ok = hide_window(found.hwnd)
            self.hwnd = found.hwnd
            self._write_session(action="hidden", reason=reason)
            if self.emit is not None:
                self.emit("tool", {"summary": "window hidden" + (f" ({reason})" if reason else ""), "source": "desktop"})
            return {"ok": ok, "action": "hidden", "hwnd": found.hwnd, "reason": reason}

    def close(self, *, reason: str = "") -> dict[str, Any]:
        """End the window process(es): the full-quit case."""

        with self._lock:
            from service.processes import kill

            closed = []
            for w in find_windows(self.title):
                if kill(w.pid, tree=True):
                    closed.append(w.pid)
            self.hwnd = 0
            self._write_session(action="closed", reason=reason)
            return {"ok": True, "action": "closed", "pids": closed, "reason": reason}

    # -- internals -------------------------------------------------------

    def _launch(self) -> dict[str, Any]:
        if not self.engine:
            return {"ok": False, "detail": "no Chromium engine (Edge, Chrome, Brave, Vivaldi) was found on this machine"}
        try:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            command = window_command(self.engine, self.url, profile_dir=self.profile_dir, size=self.size)
            subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             close_fds=True)
            return {"ok": True}
        except (OSError, ValueError) as exc:
            return {"ok": False, "detail": f"{Path(self.engine).name} would not start: {exc}"}

    def _wait_for_window(self, timeout: float) -> FoundWindow | None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            found = self.find(include_hidden=False)
            if found is not None:
                return found
            time.sleep(0.05)
        return None

    def _ensure_single(self, *, keep: int) -> list[int]:
        """One window.  Duplicates (a restart's leftovers) are closed."""

        from service.processes import kill

        removed = []
        seen: dict[int, FoundWindow] = {w.hwnd: w for w in find_windows(self.title)}
        for w in find_windows_of(self.profile_pids()):
            # Only visible main windows: the engine's hidden helper windows
            # belong to the same process, and closing one closes the engine.
            if w.visible:
                seen.setdefault(w.hwnd, w)
        for w in seen.values():
            if w.hwnd == keep or not w.visible:
                continue
            w2 = _win32()
            if w2 is not None:
                _ctypes, _wt, user32 = w2
                # Same process as the kept window: closing the process would
                # close both; send WM_CLOSE to the extra window instead.
                keep_pid = next((x.pid for x in find_windows(self.title) if x.hwnd == keep), 0)
                if w.pid == keep_pid:
                    user32.SendMessageW(w.hwnd, 0x0010, 0, 0)  # WM_CLOSE
                    removed.append(w.hwnd)
                    continue
            if kill(w.pid, tree=True):
                removed.append(w.hwnd)
        return removed

    # -- the beacon watcher ---------------------------------------------

    def start_watcher(self, interval: float = 0.25) -> None:
        """Poll the beacon a second ``ZEUS.exe`` leaves while the API is not yet up."""

        if self._watcher is not None:
            return

        def run() -> None:
            while not self._stop.wait(interval):
                try:
                    if self.beacon_path.is_file():
                        self.beacon_path.unlink(missing_ok=True)
                        self.show(reason="beacon")
                except Exception:  # noqa: BLE001 - the watcher must survive anything
                    continue

        self._watcher = threading.Thread(target=run, daemon=True, name="desktop-window-watcher")
        self._watcher.start()

    def stop_watcher(self) -> None:
        self._stop.set()
