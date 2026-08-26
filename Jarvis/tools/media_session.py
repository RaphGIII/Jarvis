"""What Windows itself says is playing.

This is the independent observer for music actions.  It is not the thing that
plays anything: it asks the operating system's media session -- the same source
that drives the volume flyout and the keyboard's media keys -- what application
is playing, which track, and whether it is running or paused.

That separation is the whole reason this module exists.  A music capability
that reports its own success is exactly the failure this system has already
been bitten by: an earlier attempt at a music capability was recorded as
*acquired* while every branch returned ``{"message": "Dry run: ..."}`` and
nothing ever played.  A receipt is only worth something if the checking is done
by something other than the doing, so ZEUS reads playback state from the OS and
compares it against what the user asked for.

Implemented through PowerShell rather than a Python WinRT binding because
neither ``winrt`` nor ``winsdk`` is installed on this machine and the brief
forbids adding global packages.  The same trade-off the toast notification in
:mod:`tools.desktop` already makes: PowerShell can project WinRT types, costs
one subprocess, and needs no install.

What this can and cannot prove, stated plainly:

*It can prove* which application holds the media session, the exact title and
artist Windows has been handed, whether the transport is playing or paused, and
the playback position -- all read after the fact, from outside.

*It cannot prove* that sound reached the speakers.  A muted system still
reports ``Playing``.  :mod:`tools.audio_probe` is the check for that, and the
two together are what "it is really playing" means here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Long enough for PowerShell to start and project the WinRT types on a cold
#: run; short enough that a wedged call cannot hold up a conversation turn.
TIMEOUT_SECONDS = 25.0

#: Windows reports the app that owns the session by its model id. Spotify's is
#: literally "Spotify.exe" for both the Store build and the desktop installer.
SPOTIFY_APP_IDS = ("spotify.exe", "spotifyab.spotifymusic")


@dataclass(frozen=True)
class MediaState:
    """A reading of the system media session."""

    ok: bool
    app: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    status: str = ""
    position_seconds: float = 0.0
    duration_seconds: float = 0.0
    error: str = ""
    #: Every session Windows knows about, not just the current one.
    sessions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def playing(self) -> bool:
        return self.status.lower() == "playing"

    @property
    def paused(self) -> bool:
        return self.status.lower() == "paused"

    @property
    def is_spotify(self) -> bool:
        return any(marker in self.app.lower() for marker in SPOTIFY_APP_IDS)

    def describe(self) -> str:
        if not self.ok:
            return f"no media session ({self.error})" if self.error else "no media session"
        where = self.app or "unknown app"
        what = f"{self.title} - {self.artist}".strip(" -") or "unknown track"
        return f"{what} [{self.status}] in {where}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "app": self.app,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "status": self.status,
            "position_seconds": round(self.position_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "error": self.error,
            "sessions": list(self.sessions),
        }


#: The PowerShell side.  ``AsTask`` is fished out of the extension methods
#: because Windows PowerShell 5.1 has no ``await`` -- this is the standard
#: incantation for consuming a WinRT IAsyncOperation from PowerShell.
_SCRIPT = r"""
param([string]$App = "", [string]$Command = "status")
$ErrorActionPreference = "Stop"
# PowerShell writes using the console's output encoding, which on this machine
# is a legacy code page. Python decodes as UTF-8, so every non-ASCII character
# came back as U+FFFD: "Bück dich" was stored in a receipt as "B<?>ck dich" and
# then failed to match the title Windows had actually reported. For a German
# user that is most track titles. Both encodings are set because they govern
# different paths -- $OutputEncoding for the pipeline, [Console]::OutputEncoding
# for what reaches a redirected stdout.
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
try {
  Add-Type -AssemblyName System.Runtime.WindowsRuntime
  $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
      $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
      $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
  })[0]
  function Await($t, $type) {
      $netTask = $asTaskGeneric.MakeGenericMethod($type).Invoke($null, @($t))
      $netTask.Wait(12000) | Out-Null
      $netTask.Result
  }
  $mgrType = [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager, Windows.Media, ContentType = WindowsRuntime]
  $mgr = Await ($mgrType::RequestAsync()) ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager])

  $all = @()
  foreach ($s in $mgr.GetSessions()) { $all += $s.SourceAppUserModelId }

  $session = $null
  if ($App) {
    foreach ($s in $mgr.GetSessions()) {
      if ($s.SourceAppUserModelId -and $s.SourceAppUserModelId.ToLower().Contains($App.ToLower())) { $session = $s; break }
    }
  }
  if ($null -eq $session) { $session = $mgr.GetCurrentSession() }
  if ($null -eq $session) {
    @{ ok = $false; error = "no active media session"; sessions = $all } | ConvertTo-Json -Compress
    exit 0
  }

  $accepted = $true
  switch ($Command) {
    "pause"    { $accepted = Await ($session.TryPauseAsync())        ([bool]) }
    "play"     { $accepted = Await ($session.TryPlayAsync())         ([bool]) }
    "next"     { $accepted = Await ($session.TrySkipNextAsync())     ([bool]) }
    "previous" { $accepted = Await ($session.TrySkipPreviousAsync()) ([bool]) }
  }
  if ($Command -ne "status") { Start-Sleep -Milliseconds 1100 }

  $props = Await ($session.TryGetMediaPropertiesAsync()) ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionMediaProperties])
  $info = $session.GetPlaybackInfo()
  $tl = $session.GetTimelineProperties()
  @{
    ok = $true
    accepted = $accepted
    app = $session.SourceAppUserModelId
    title = $props.Title
    artist = $props.Artist
    album = $props.AlbumTitle
    status = $info.PlaybackStatus.ToString()
    position = $tl.Position.TotalSeconds
    duration = $tl.EndTime.TotalSeconds
    sessions = $all
  } | ConvertTo-Json -Compress
} catch {
  @{ ok = $false; error = $_.Exception.Message } | ConvertTo-Json -Compress
}
"""


def _powershell() -> str | None:
    return shutil.which("powershell") or shutil.which("pwsh")


#: Written once per process, next to the module, so PowerShell parses the
#: script from a file rather than re-parsing it from the command line on every
#: call. Regenerated when the source changes so an edit here cannot be masked
#: by a stale copy on disk.
_SCRIPT_FILE: Path | None = None
_SCRIPT_LOCK = threading.Lock()


def _script_path() -> Path:
    global _SCRIPT_FILE
    with _SCRIPT_LOCK:
        if _SCRIPT_FILE is not None and _SCRIPT_FILE.is_file():
            return _SCRIPT_FILE
        digest = hashlib.sha256(_SCRIPT.encode("utf-8")).hexdigest()[:12]
        path = Path(tempfile.gettempdir()) / f"zeus_media_session_{digest}.ps1"
        if not path.is_file():
            # utf-8-sig: PowerShell reads a BOM-less file as the ANSI code page,
            # which would undo the encoding fix for anything non-ASCII in the
            # script itself.
            path.write_text(_SCRIPT, encoding="utf-8-sig")
        _SCRIPT_FILE = path
        return path


def available() -> bool:
    return _powershell() is not None


def _invoke(command: str, *, app: str = "") -> dict[str, Any]:
    shell = _powershell()
    if shell is None:
        return {"ok": False, "error": "powershell is not on PATH"}
    # Run from a file with real parameters, not from a -Command string.
    #
    # Measured: passing this script inline cost 4.4s per call, while the same
    # work run from a file cost 1.1s including PowerShell's own startup -- and
    # the WinRT part of it is about a millisecond. PowerShell re-parses a long
    # -Command string on every invocation, and this script is long. Three
    # seconds a call, on every music verification.
    #
    # Parameters rather than prepended assignments, because that is what a
    # param() block is for. The earlier defect was not parameters; it was
    # passing a second -Command, which PowerShell reads as an argument to the
    # first. -File takes named parameters correctly.
    try:
        completed = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(_script_path()), "-App", app, "-Command", command],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"the media session did not answer within {TIMEOUT_SECONDS:.0f}s"}
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    text = (completed.stdout or "").strip()
    if not text:
        return {"ok": False, "error": (completed.stderr or "no output from the media session").strip()[:300]}
    try:
        # The script prints one compact JSON object; anything before it is noise.
        return json.loads(text.splitlines()[-1])
    except ValueError:
        return {"ok": False, "error": f"unreadable media session output: {text[:200]}"}


def _state(payload: dict[str, Any]) -> MediaState:
    if not payload.get("ok"):
        return MediaState(
            ok=False,
            error=str(payload.get("error", "")),
            sessions=tuple(str(item) for item in (payload.get("sessions") or [])),
        )
    return MediaState(
        ok=True,
        app=str(payload.get("app", "")),
        title=str(payload.get("title", "")),
        artist=str(payload.get("artist", "")),
        album=str(payload.get("album", "")),
        status=str(payload.get("status", "")),
        position_seconds=float(payload.get("position") or 0.0),
        duration_seconds=float(payload.get("duration") or 0.0),
        sessions=tuple(str(item) for item in (payload.get("sessions") or [])),
    )


def read(*, app: str = "") -> MediaState:
    """What is playing right now, according to Windows."""

    return _state(_invoke("status", app=app))


def control(command: str, *, app: str = "") -> MediaState:
    """Send a transport command and return the state *afterwards*.

    Returning the state rather than a boolean is deliberate.  "The command was
    accepted" is what the earlier music attempts reported and it means very
    little -- Windows accepts a pause for a session that then carries on.  The
    caller needs to know what actually happened, so it gets a fresh reading.
    """

    if command not in {"play", "pause", "next", "previous", "status"}:
        return MediaState(ok=False, error=f"unsupported transport command: {command}")
    return _state(_invoke(command, app=app))


def launch_uri(uri: str) -> tuple[bool, str]:
    """Hand a ``spotify:`` (or other) URI to whatever is registered for it.

    Nothing here decides that the URI played; the caller reads the session
    afterwards and finds out.
    """

    shell = _powershell()
    if shell is None:
        return False, "powershell is not on PATH"
    if not uri or ":" not in uri:
        return False, f"not a URI: {uri!r}"
    try:
        completed = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-Command",
             f"Start-Process '{uri.replace(chr(39), '')}'"],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
            encoding="utf-8", errors="replace",
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return False, (completed.stderr or "the shell refused the URI").strip()[:300]
    return True, ""


def main(argv: list[str] | None = None) -> int:
    """``python -m tools.media_session [status|pause|play|next|previous]``."""

    import argparse

    parser = argparse.ArgumentParser(prog="python -m tools.media_session")
    parser.add_argument("command", nargs="?", default="status",
                        choices=["status", "pause", "play", "next", "previous"])
    parser.add_argument("--app", default="", help="prefer a session whose app id contains this")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    state = control(args.command, app=args.app) if args.command != "status" else read(app=args.app)
    print(json.dumps(state.to_dict(), indent=2) if args.json else state.describe())
    return 0 if state.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
