"""Tools that reach outside the workspace: the machine itself.

Everything in :mod:`tools.builtin` is bounded by a project directory, which is
what makes it safe to hand to a small model.  These are not.  Launching an
application, controlling playback, writing the clipboard and taking screenshots
all affect the user's actual desktop, so the design here is about limiting blast
radius rather than capability:

*Risk is declared honestly.*  Opening a URL is MODERATE, not LOW, because it can
reach the network and start a program.  Reading the clipboard is HIGH: it
routinely contains passwords the user copied a moment ago, and a tool that
quietly forwards it into a model's context is an exfiltration path however
benign the intent.

*Reads before writes.*  ``list_windows`` and ``running_processes`` exist so a
capability can find out what is installed and choose, instead of guessing a
program name and failing.  That is what makes "play some music" answerable on a
machine whose software Jarvis has never seen.

*Every side effect supports a dry run.*  The same contract capabilities use: a
tool that would launch something reports what it *would* launch when asked, so
acquisition can verify the decision without twenty windows opening.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

from tools.registry import RiskLevel, ToolContext, ToolSpec

#: Media files Jarvis will treat as playable when searching the music folders.
AUDIO_SUFFIXES = frozenset({".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".wma"})

#: Extensions that must never be launched by name from a model's suggestion.
#: Not a security boundary on its own -- the risk level and the permission tier
#: are -- but it removes the most obvious way a wrong guess becomes an install.
_REFUSED_LAUNCH_SUFFIXES = frozenset({".msi", ".scr", ".cpl", ".reg", ".ps1", ".vbs"})


# --------------------------------------------------------------------------
# Looking around
# --------------------------------------------------------------------------

def running_processes(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """What is running, so a capability can use what is already open."""

    limit = int(payload.get("limit", 40) or 40)
    match = str(payload.get("contains", "")).lower()

    if sys.platform == "win32":
        command = ["tasklist", "/fo", "csv", "/nh"]
    else:
        command = ["ps", "-eo", "comm,pid", "--no-headers"]

    completed = _run(command)
    if not completed["ok"]:
        return completed

    names: list[str] = []
    for line in completed["output"].splitlines():
        line = line.strip()
        if not line:
            continue
        name = line.split(",")[0].strip('"') if sys.platform == "win32" else line.split()[0]
        if match and match not in name.lower():
            continue
        if name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return {"ok": True, "processes": names, "count": len(names)}


def find_applications(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Locate installed programs by name.

    Answers "is there a media player on this machine" without the model having
    to guess an executable name and discover it was wrong by failing to launch
    it.
    """

    names = payload.get("names") or []
    if isinstance(names, str):
        names = [names]
    found: dict[str, str] = {}
    for name in names[:20]:
        name = str(name).strip()
        if not name:
            continue
        located = shutil.which(name)
        if located:
            found[name] = located
            continue
        if sys.platform == "win32":
            located = _find_windows_app(name)
            if located:
                found[name] = located
    return {"ok": True, "found": found, "missing": [n for n in names if n not in found]}


def _find_windows_app(name: str) -> str:
    """Look in the usual install roots for something matching ``name``."""

    stem = name.lower().removesuffix(".exe")
    roots = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    for root in roots:
        if not root or not Path(root).is_dir():
            continue
        try:
            for path in Path(root).glob(f"*/{stem}*.exe"):
                return str(path)
            for path in Path(root).glob(f"*/*/{stem}*.exe"):
                return str(path)
        except OSError:
            continue
    return ""


def media_folders(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """The user's music locations, and whether they actually contain anything."""

    candidates: list[Path] = []
    home = Path.home()
    for name in ("Music", "Musik", "Downloads"):
        candidate = home / name
        if candidate.is_dir():
            candidates.append(candidate)
    extra = payload.get("extra_paths") or []
    for item in extra if isinstance(extra, list) else []:
        path = Path(str(item)).expanduser()
        if path.is_dir():
            candidates.append(path)

    report = []
    for folder in candidates:
        count = 0
        try:
            for path in folder.rglob("*"):
                if path.suffix.lower() in AUDIO_SUFFIXES:
                    count += 1
                    if count >= 500:
                        break
        except OSError:
            continue
        report.append({"path": str(folder), "audio_files": count})
    return {"ok": True, "folders": report}


def find_media(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Search the music folders for playable files matching a query."""

    query = str(payload.get("query", "")).strip().lower()
    limit = int(payload.get("limit", 25) or 25)
    roots = payload.get("paths") or []
    searched: list[Path] = []
    if roots and isinstance(roots, list):
        searched = [Path(str(item)).expanduser() for item in roots]
    else:
        home = Path.home()
        searched = [home / name for name in ("Music", "Musik", "Downloads")]

    matches: list[dict[str, Any]] = []
    for root in searched:
        if not root.is_dir():
            continue
        try:
            for path in root.rglob("*"):
                if path.suffix.lower() not in AUDIO_SUFFIXES:
                    continue
                if query and query not in path.stem.lower() and query not in str(path.parent).lower():
                    continue
                matches.append({"path": str(path), "name": path.stem, "size": path.stat().st_size})
                if len(matches) >= limit:
                    return {"ok": True, "matches": matches, "truncated": True}
        except OSError:
            continue
    return {"ok": True, "matches": matches, "truncated": False}


# --------------------------------------------------------------------------
# Acting
# --------------------------------------------------------------------------

def open_path(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Open a file or folder with whatever the OS associates with it.

    Preferred over naming a specific player: the association is the user's own
    choice, already made, and it works for file types Jarvis knows nothing
    about.
    """

    raw = str(payload.get("path", "")).strip()
    if not raw:
        return {"ok": False, "error": "path is required"}
    target = Path(raw).expanduser()
    if not target.exists():
        return {"ok": False, "error": f"no such path: {target}"}
    if target.suffix.lower() in _REFUSED_LAUNCH_SUFFIXES:
        return {"ok": False, "error": f"refusing to launch {target.suffix} directly"}

    if payload.get("dry_run"):
        return {"ok": True, "dry_run": True, "would_open": str(target)}

    try:
        if sys.platform == "win32":
            os.startfile(str(target))  # noqa: S606 - the OS association is the point
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "opened": str(target)}


def open_url(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Open a URL in the default browser."""

    url = str(payload.get("url", "")).strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        # file:// would turn "open a link" into "read anything on the disk",
        # and the other schemes are program launchers wearing a URL costume.
        return {"ok": False, "error": "only http and https URLs may be opened"}

    if payload.get("dry_run"):
        return {"ok": True, "dry_run": True, "would_open": url}

    import webbrowser

    return {"ok": bool(webbrowser.open(url)), "opened": url}


def launch_application(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Start a program, optionally with arguments."""

    program = str(payload.get("program", "")).strip()
    if not program:
        return {"ok": False, "error": "program is required"}

    resolved = shutil.which(program) or (_find_windows_app(program) if sys.platform == "win32" else "")
    if not resolved and Path(program).is_file():
        resolved = program
    if not resolved:
        return {"ok": False, "error": f"{program!r} was not found on this machine"}
    if Path(resolved).suffix.lower() in _REFUSED_LAUNCH_SUFFIXES:
        return {"ok": False, "error": f"refusing to launch {Path(resolved).suffix} directly"}

    arguments = payload.get("arguments") or []
    if isinstance(arguments, str):
        arguments = [arguments]
    command = [resolved, *[str(item) for item in arguments]]

    if payload.get("dry_run"):
        return {"ok": True, "dry_run": True, "would_run": command}

    try:
        subprocess.Popen(command)
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "launched": command}


def media_control(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Play/pause, next, previous, and volume, via the OS media keys.

    Uses the system-wide media keys rather than talking to a specific player,
    so it works with whatever is actually playing -- Spotify, a browser tab, a
    local player -- without Jarvis needing an integration per application.
    """

    action = str(payload.get("action", "")).strip().lower()
    keys = {
        "play": 0xB3, "pause": 0xB3, "playpause": 0xB3,
        "next": 0xB0, "previous": 0xB1, "stop": 0xB2,
        "mute": 0xAD, "volume_down": 0xAE, "volume_up": 0xAF,
    }
    if action not in keys:
        return {"ok": False, "error": f"unknown action {action!r}", "supported": sorted(keys)}

    if payload.get("dry_run"):
        return {"ok": True, "dry_run": True, "would_send": action}

    if sys.platform != "win32":
        return {"ok": False, "error": "media keys are only implemented for Windows so far"}

    try:
        import ctypes

        user32 = ctypes.windll.user32
        code = keys[action]
        user32.keybd_event(code, 0, 0, 0)          # press
        user32.keybd_event(code, 0, 2, 0)          # release
    except Exception as exc:
        return {"ok": False, "error": f"could not send media key: {exc}"}
    return {"ok": True, "sent": action}


def clipboard_write(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Put text on the clipboard."""

    text = str(payload.get("text", ""))
    if payload.get("dry_run"):
        return {"ok": True, "dry_run": True, "would_write": len(text)}
    if sys.platform != "win32":
        return {"ok": False, "error": "clipboard is only implemented for Windows so far"}
    completed = subprocess.run(["clip"], input=text, text=True, encoding="utf-8")
    return {"ok": completed.returncode == 0, "characters": len(text)}


def notify(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Show a desktop notification."""

    from core.identity import current

    # A desktop notification is about as user-facing as this system gets, so
    # the name on it is the configured one rather than a literal.
    identity = current()
    title = str(payload.get("title") or identity.product_name)
    message = str(payload.get("message", ""))
    if payload.get("dry_run"):
        return {"ok": True, "dry_run": True, "would_notify": f"{title}: {message}"}
    if sys.platform != "win32":
        return {"ok": False, "error": "notifications are only implemented for Windows so far"}

    # PowerShell's toast API needs no dependency and no install.
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
        "ContentType = WindowsRuntime] > $null; "
        "$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
        f"$t.GetElementsByTagName('text')[0].AppendChild($t.CreateTextNode({_ps_quote(title)})) > $null; "
        f"$t.GetElementsByTagName('text')[1].AppendChild($t.CreateTextNode({_ps_quote(message)})) > $null; "
        f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier({_ps_quote(identity.product_name)})"
        ".Show([Windows.UI.Notifications.ToastNotification]::new($t))"
    )
    completed = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=20)
    return {"ok": completed["ok"], "detail": completed.get("output", "")[-300:]}


def screenshot(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Capture the screen to a PNG inside the workspace."""

    relative = str(payload.get("path", "screenshot.png"))
    target = (Path(context.workspace) / relative).resolve()
    try:
        target.relative_to(Path(context.workspace).resolve())
    except ValueError:
        return {"ok": False, "error": "screenshots must be written inside the workspace"}

    if payload.get("dry_run"):
        return {"ok": True, "dry_run": True, "would_write": str(target)}
    if sys.platform != "win32":
        return {"ok": False, "error": "screen capture is only implemented for Windows so far"}

    target.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
        "$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
        "$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height; "
        "$g = [System.Drawing.Graphics]::FromImage($bmp); "
        "$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size); "
        f"$bmp.Save({_ps_quote(str(target))}); $bmp.Dispose()"
    )
    completed = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=60)
    if not completed["ok"] or not target.is_file():
        return {"ok": False, "error": completed.get("output", "capture failed")[-300:]}
    return {"ok": True, "path": str(target), "bytes": target.stat().st_size}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _ps_quote(text: str) -> str:
    """Single-quote for PowerShell, doubling embedded quotes."""

    return "'" + str(text).replace("'", "''") + "'"


def _run(command: list[str], *, timeout: float = 30.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return {"ok": False, "error": f"{command[0]} is not available"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"{command[0]} timed out"}
    return {
        "ok": completed.returncode == 0,
        "output": (completed.stdout or "") + (completed.stderr or ""),
        "exit_code": completed.returncode,
    }


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def desktop_tools() -> list[ToolSpec]:
    """Tools that touch the host.  Risk levels are deliberately pessimistic."""

    return [
        ToolSpec(
            name="running_processes",
            purpose="List running programs, so you can use what is already open.",
            input_schema={
                "type": "object",
                "properties": {"contains": {"type": "string"}, "limit": {"type": "integer"}},
                "required": [],
            },
            adapter=running_processes,
            risk=RiskLevel.SAFE,
            tags=("desktop", "investigate"),
            example='{"name": "running_processes", "arguments": {"contains": "spotify"}}',
        ),
        ToolSpec(
            name="find_applications",
            purpose="Check which of several programs are installed on this machine.",
            input_schema={
                "type": "object",
                "properties": {"names": {"type": "array", "items": {"type": "string"}}},
                "required": ["names"],
            },
            adapter=find_applications,
            risk=RiskLevel.SAFE,
            tags=("desktop", "investigate"),
            example='{"name": "find_applications", "arguments": {"names": ["vlc", "spotify", "wmplayer"]}}',
        ),
        ToolSpec(
            name="media_folders",
            purpose="Find the user's music folders and how many audio files each contains.",
            input_schema={
                "type": "object",
                "properties": {"extra_paths": {"type": "array", "items": {"type": "string"}}},
                "required": [],
            },
            adapter=media_folders,
            risk=RiskLevel.SAFE,
            tags=("desktop", "media", "investigate"),
            example='{"name": "media_folders", "arguments": {}}',
        ),
        ToolSpec(
            name="find_media",
            purpose="Search the user's music folders for playable audio files.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
            adapter=find_media,
            risk=RiskLevel.SAFE,
            tags=("desktop", "media"),
            example='{"name": "find_media", "arguments": {"query": "bach"}}',
        ),
        ToolSpec(
            name="open_path",
            purpose="Open a file or folder with the program the user has associated with it.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "dry_run": {"type": "boolean"}},
                "required": ["path"],
            },
            adapter=open_path,
            risk=RiskLevel.HIGH,
            tags=("desktop", "media"),
            example='{"name": "open_path", "arguments": {"path": "C:/Users/me/Music/track.mp3"}}',
        ),
        ToolSpec(
            name="open_url",
            purpose="Open an http or https URL in the default browser.",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}, "dry_run": {"type": "boolean"}},
                "required": ["url"],
            },
            adapter=open_url,
            risk=RiskLevel.MODERATE,
            tags=("desktop", "web"),
            example='{"name": "open_url", "arguments": {"url": "https://example.com"}}',
        ),
        ToolSpec(
            name="launch_application",
            purpose="Start an installed program, optionally with arguments.",
            input_schema={
                "type": "object",
                "properties": {
                    "program": {"type": "string"},
                    "arguments": {"type": "array", "items": {"type": "string"}},
                    "dry_run": {"type": "boolean"},
                },
                "required": ["program"],
            },
            adapter=launch_application,
            risk=RiskLevel.HIGH,
            tags=("desktop",),
            example='{"name": "launch_application", "arguments": {"program": "notepad"}}',
        ),
        ToolSpec(
            name="media_control",
            purpose="Send a system media key: play, pause, next, previous, volume.",
            input_schema={
                "type": "object",
                "properties": {"action": {"type": "string"}, "dry_run": {"type": "boolean"}},
                "required": ["action"],
            },
            adapter=media_control,
            risk=RiskLevel.MODERATE,
            tags=("desktop", "media"),
            example='{"name": "media_control", "arguments": {"action": "playpause"}}',
        ),
        ToolSpec(
            name="clipboard_write",
            purpose="Put text on the system clipboard.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}, "dry_run": {"type": "boolean"}},
                "required": ["text"],
            },
            adapter=clipboard_write,
            risk=RiskLevel.HIGH,
            tags=("desktop",),
            example='{"name": "clipboard_write", "arguments": {"text": "hello"}}',
        ),
        ToolSpec(
            name="notify",
            purpose="Show a desktop notification.",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "message": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                },
                "required": ["message"],
            },
            adapter=notify,
            risk=RiskLevel.MODERATE,
            tags=("desktop",),
            example='{"name": "notify", "arguments": {"message": "the build finished"}}',
        ),
        ToolSpec(
            name="screenshot",
            purpose="Capture the primary screen to a PNG inside the workspace.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "dry_run": {"type": "boolean"}},
                "required": [],
            },
            adapter=screenshot,
            risk=RiskLevel.HIGH,
            tags=("desktop", "vision"),
            example='{"name": "screenshot", "arguments": {"path": "shot.png"}}',
        ),
    ]
