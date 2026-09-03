"""The interface in its own desktop window, not in the owner's browser.

Starting ZEUS used to hand the interface to whatever browser the machine
happens to prefer: a new tab among the owner's other tabs, sharing that
profile's session, extensions and history, and disappearing the moment the tab
was closed by accident.  The owner asked for an application instead -- one
window, its own taskbar button, no address bar, nothing of the browser around
it.

There is no dependency budget for that: the brief forbids adding packages, and
the two stdlib options are a Tk window that cannot render HTML and
``webbrowser``, which is precisely the thing being replaced.  What every
Windows machine *does* already have is a Chromium engine (Edge ships with the
OS), and Chromium's ``--app=`` mode is exactly an application window -- no
tabs, no toolbar, its own entry in the taskbar, the page ``<title>`` as the
window title.  Pointed at ``--user-data-dir`` of our own it is also a separate
profile and a separate process from the owner's browsing, so closing ZEUS never
closes their tabs and vice versa.

The fallback matters as much as the window.  A machine with no Chromium engine
must still get an interface, so this degrades to ``webbrowser`` rather than to
nothing: the worst outcome here is the old behaviour, never a blank screen.

One window per start of the core, deliberately not deduplicated.  A restart --
a promotion, or a crash the supervisor recovers from -- leaves the previous
window open and adds another, because every way of recognising the old one
(a recorded pid, a lock file) can be wrong in the direction that matters: it
would decide a window is already there when it is not, and the owner would be
left staring at nothing.  An extra window is a nuisance; no window is a
failure.

Nothing in this module imports the service, so it can be exercised -- and its
command line inspected -- without starting ZEUS.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Chromium builds keep their per-profile state in a directory of their own.
#: Ours lives beside the rest of the state so that a backup of ``data/jarvis``
#: carries the window's zoom level and geometry with it.
PROFILE_DIRNAME = "window"

#: Default geometry.  Large enough for the eye, the log and a side panel
#: without being a maximised window the owner has to shrink every time.
DEFAULT_SIZE = (1280, 860)

#: Chromium engines on Windows, relative to a program-files or local-app-data
#: root.  Edge first: it is present on every supported Windows installation,
#: so it is the one answer that is almost always right.
_WINDOWS_ENGINES = (
    r"Microsoft\Edge\Application\msedge.exe",
    r"Google\Chrome\Application\chrome.exe",
    r"Chromium\Application\chrome.exe",
    r"BraveSoftware\Brave-Browser\Application\brave.exe",
    r"Vivaldi\Application\vivaldi.exe",
)

#: Names to try on PATH when none of the usual install roots has one.
_WINDOWS_ON_PATH = ("msedge", "chrome", "chromium", "brave", "vivaldi")

_MACOS_ENGINES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
)

_POSIX_ENGINES = (
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "microsoft-edge", "microsoft-edge-stable", "brave-browser",
)


def default_profile_dir() -> Path:
    """Where the window keeps its profile.  Honours ``JARVIS_STATE_ROOT``."""

    root = os.getenv("JARVIS_STATE_ROOT", "").strip()
    base = Path(root) if root else Path(__file__).resolve().parent.parent / "data" / "jarvis"
    return base / PROFILE_DIRNAME


def _windows_roots() -> list[str]:
    return [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA", ""),
    ]


def find_engine() -> str:
    """The Chromium executable to host the window, or ``""`` if there is none.

    ``ZEUS_WINDOW_BROWSER`` overrides the search.  An override that does not
    resolve returns nothing rather than quietly falling back to a different
    browser: being pointed at the wrong engine is worth noticing.
    """

    override = os.getenv("ZEUS_WINDOW_BROWSER", "").strip().strip('"')
    if override:
        if Path(override).is_file():
            return override
        return shutil.which(override) or ""

    if sys.platform == "win32":
        for relative in _WINDOWS_ENGINES:
            for root in _windows_roots():
                if root and (Path(root) / relative).is_file():
                    return str(Path(root) / relative)
        for name in _WINDOWS_ON_PATH:
            found = shutil.which(name)
            if found:
                return found
        return ""

    if sys.platform == "darwin":
        for path in _MACOS_ENGINES:
            if Path(path).is_file():
                return path

    for name in _POSIX_ENGINES:
        found = shutil.which(name)
        if found:
            return found
    return ""


def window_command(
    engine: str,
    url: str,
    *,
    profile_dir: str | Path,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> list[str]:
    """The command line that turns a Chromium engine into an application window.

    ``--app`` is what removes the browser: no tabs, no address bar, its own
    taskbar button.  ``--user-data-dir`` is what keeps it out of the owner's
    browsing: a separate profile, so their session, extensions and open tabs
    are untouched and closing one never closes the other.
    """

    width, height = size
    command = [
        str(engine),
        f"--app={url}",
        f"--user-data-dir={Path(profile_dir)}",
        f"--window-size={int(width)},{int(height)}",
        # A first-run wizard or a "make me your default browser" bar in front
        # of the interface would defeat the point of an application window.
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-mode",
        # ZEUS speaks without being clicked first.  Chromium's autoplay policy
        # blocks `audio.play()` until a user gesture *on the page*, so a fresh
        # window sat silent until the owner happened to click the input --
        # which read as "voice only works after clicking Talk to Zeus".  This
        # is ZEUS's own shell window, not the owner's browser; its answers are
        # the point of the page.
        "--autoplay-policy=no-user-gesture-required",
        # No "Diese Seite übersetzen?" bar over the interface: the page is the
        # product, not a foreign website.  Scoped to this profile only.
        "--disable-features=Translate,TranslateUI",
        # An unfocused or occluded ZEUS still renders its state live: the eye,
        # the event stream and the voice-state animation must not be throttled
        # just because the owner is looking at another window.
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
    ]
    # ZEUS is an operating environment, not a browser page.  The default shell
    # is a NATIVE borderless MAXIMIZED window: no Windows title bar, no visible
    # frame -- but NOT the browser Fullscreen API.  --start-fullscreen made
    # Edge show its "Vollbildmodus beenden" toast on every launch; the frame is
    # instead removed by Win32 after the window appears (service.desktop
    # _style_frameless).  The window is launched maximized so the borderless
    # restyle has a full-size window to strip.  ZEUS_WINDOW_MODE overrides per
    # device: fullscreen (old kiosk-style immersive), kiosk, windowed.
    mode = os.getenv("ZEUS_WINDOW_MODE", "").strip().lower() or "borderless"
    if mode == "kiosk":
        command.append("--kiosk")
    elif mode == "fullscreen":
        command.append("--start-fullscreen")
    else:  # borderless (default) and maximized both launch maximized
        command.append("--start-maximized")
    return command


@dataclass
class WindowLaunch:
    """What actually happened, so the caller can say it out loud."""

    ok: bool = False
    #: ``window`` (its own desktop window), ``browser`` (the fallback) or
    #: ``none`` (nothing was opened).
    mode: str = "none"
    engine: str = ""
    pid: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "mode": self.mode, "engine": self.engine,
            "pid": self.pid, "detail": self.detail,
        }

    def describe(self) -> str:
        if self.mode == "window":
            return f"in its own window ({Path(self.engine).name})"
        if self.mode == "browser":
            return f"in the default browser -- {self.detail}"
        return self.detail or "not opened"


def open_window(
    url: str,
    *,
    profile_dir: str | Path | None = None,
    size: tuple[int, int] = DEFAULT_SIZE,
    fallback: bool = True,
) -> WindowLaunch:
    """Show ``url`` in a desktop window, falling back to the browser.

    Never raises: a failure to open a window must not take the service down
    with it, because the service is perfectly usable from another device once
    it is running.
    """

    engine = find_engine()
    if engine:
        directory = Path(profile_dir) if profile_dir is not None else default_profile_dir()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            command = window_command(engine, url, profile_dir=directory, size=size)
            process = subprocess.Popen(  # noqa: S603 - a located executable, fixed arguments
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            return WindowLaunch(True, "window", engine, process.pid, "application window")
        except (OSError, ValueError) as exc:
            detail = f"{Path(engine).name} would not start: {exc}"
    else:
        detail = "no Chromium engine (Edge, Chrome, Brave, Vivaldi) was found on this machine"

    if not fallback:
        return WindowLaunch(False, "none", engine, 0, detail)
    try:
        opened = bool(webbrowser.open(url))
    except Exception as exc:  # noqa: BLE001 - any browser failure is just "no UI here"
        return WindowLaunch(False, "none", engine, 0, f"{detail}; the browser also failed: {exc}")
    if not opened:
        return WindowLaunch(False, "none", engine, 0, f"{detail}; no browser could be opened either")
    return WindowLaunch(True, "browser", engine, 0, detail)


def main(argv: list[str] | None = None) -> int:
    """``python -m jarvis.window <url>`` -- open a window at a URL.

    Useful on its own: it is how a second window onto a running ZEUS is
    opened, and how the engine search is checked on a new machine.
    """

    parser = argparse.ArgumentParser(
        prog="python -m jarvis.window",
        description="Open a URL in its own desktop window instead of a browser tab.",
    )
    parser.add_argument("url", nargs="?", default="", help="the URL to show")
    parser.add_argument("--profile-dir", default="", help="override the window's profile directory")
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    parser.add_argument("--no-fallback", action="store_true", help="fail rather than use the browser")
    parser.add_argument("--which", action="store_true", help="print the engine that would be used and stop")
    args = parser.parse_args(argv)

    if args.which:
        engine = find_engine()
        print(engine or "no Chromium engine found")
        return 0 if engine else 1

    if not args.url:
        parser.error("a URL is required")

    launch = open_window(
        args.url,
        profile_dir=args.profile_dir or None,
        size=(args.width, args.height),
        fallback=not args.no_fallback,
    )
    print(launch.describe())
    return 0 if launch.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
