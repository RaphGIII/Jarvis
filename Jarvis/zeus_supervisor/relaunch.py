"""The relaunch watchdog: start the new ZEUS.exe, and undo it if it never gets READY.

``python -m zeus_supervisor.relaunch --wait-pid <old supervisor> --exe <dist/ZEUS/ZEUS.exe>
--previous <dist/ZEUS.previous> --state <state dir> [--timeout 300]``

Why a separate process: the old supervisor holds the single-instance mutex
until it exits, so it cannot start its successor itself, and a successor that
crashes before READY cannot roll itself back.  This program waits for the old
supervisor to be gone, starts the promoted executable detached, then watches
``/api/health`` for a READY from a *different* supervisor pid.  If that does
not happen within the timeout -- the exe never started, or it started and
held -- the previous release is renamed back into place and started instead.
Everything it does goes into ``<state>/logs/relaunch.log`` and a receipt in
``<state>/releases.jsonl``.

Standard library only: this file is frozen into the executable too, and runs
from the repository when the supervisor is run from source.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _health(port: int, token: str) -> dict:
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{port}/api/health", headers={"X-Jarvis-Token": token})
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _start(exe: Path) -> int:
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    process = subprocess.Popen([str(exe)], cwd=str(exe.parent), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    return process.pid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m zeus_supervisor.relaunch")
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--exe", required=True)
    parser.add_argument("--previous", default="")
    parser.add_argument("--state", required=True)
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--promotion", default="")
    args = parser.parse_args(argv)

    state = Path(args.state)
    logs = state / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "relaunch.log"
    token = ""
    try:
        token = (state / "token").read_text(encoding="utf-8").strip()
    except OSError:
        pass

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{_now()} {message}\n")

    def receipt(outcome: str, **fields) -> None:
        try:
            with (state / "releases.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": "relaunch", "outcome": outcome, "at": _now(), "exe": args.exe,
                                     "promotion": args.promotion, **fields}, default=str) + "\n")
        except OSError:
            pass

    exe = Path(args.exe)
    previous = Path(args.previous) if args.previous else None
    started = time.monotonic()
    log(f"waiting for supervisor pid {args.wait_pid} to exit")
    while _alive(args.wait_pid) and time.monotonic() - started < 120:
        time.sleep(0.25)
    if _alive(args.wait_pid):
        log("the old supervisor did not exit; giving up")
        receipt("old_supervisor_still_running")
        return 2
    waited = round(time.monotonic() - started, 1)

    # A staged promotion (the running exe held its directory open, so the
    # release manager could not swap): the old supervisor is gone now, the
    # locks with it -- swap here, then start what is now known-good.
    swapped = _swap_staged(state, log, receipt)
    if swapped is not None:
        previous = previous or swapped

    if not exe.is_file():
        log(f"{exe} does not exist")
        return _restore(previous, exe, log, receipt, "exe missing", args.port, token)

    try:
        pid = _start(exe)
        log(f"started {exe} as pid {pid} ({waited}s after the old supervisor left)")
    except OSError as exc:
        log(f"could not start {exe}: {exc}")
        return _restore(previous, exe, log, receipt, f"start failed: {exc}", args.port, token)

    deadline = time.monotonic() + args.timeout
    last = ""
    while time.monotonic() < deadline:
        health = _health(args.port, token)
        if health.get("ready") and not health.get("supervisor"):
            elapsed = round(time.monotonic() - started, 1)
            log(f"READY at revision {str(health.get('revision', ''))[:12]} after {elapsed}s; relaunch verified")
            receipt("healthy", seconds=elapsed, revision=health.get("revision", ""))
            return 0
        phase = str(health.get("phase") or health.get("detail") or "")
        if phase and phase != last:
            log(f"  {phase}")
            last = phase
        if health.get("supervisor") and str(health.get("phase", "")).lower() in {"held", "error"}:
            log(f"the new supervisor reports {health.get('phase')}: {health.get('detail', '')}")
            break
        time.sleep(1.0)
    return _restore(previous, exe, log, receipt, f"not READY within {args.timeout:.0f}s", args.port, token)


def _swap_staged(state: Path, log, receipt) -> Path | None:
    """known-good -> previous, staged -> known-good, from ``<state>/control/staged.json``.

    Returns the previous-release path when a swap happened, None otherwise.
    A swap that fails leaves the known-good release in place (the rename is
    attempted before anything is removed) and is recorded; the pointer is
    removed either way so a failed staging is never retried blindly.
    """

    pointer = state / "control" / "staged.json"
    if not pointer.is_file():
        return None
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log(f"staged pointer unreadable: {exc}")
        pointer.unlink(missing_ok=True)
        return None
    staged, known_good, previous = Path(data.get("staged", "")), Path(data.get("known_good", "")), Path(data.get("previous", ""))
    try:
        pointer.unlink(missing_ok=True)
    except OSError:
        pass
    if not (staged / "ZEUS.exe").is_file():
        log(f"staged release missing at {staged}; not swapping")
        receipt("swap_skipped", reason="staged release missing")
        return None
    try:
        if previous.exists():
            shutil.rmtree(previous, ignore_errors=True)
            if previous.exists():
                previous.rename(previous.parent / f"{previous.name}.{int(time.time())}")
        if known_good.exists():
            known_good.rename(previous)
        staged.rename(known_good)
        (known_good / "PROMOTED.json").write_text(json.dumps({
            "candidate": data.get("candidate", ""), "at": _now(), "revision": data.get("revision", ""),
            "fingerprint": data.get("fingerprint", ""), "previous": str(previous) if (previous / "ZEUS.exe").is_file() else "",
            "swapped_by": "relaunch watchdog",
        }, indent=2), encoding="utf-8")
        log(f"swapped staged release into {known_good} (previous kept at {previous})")
        receipt("swapped", revision=data.get("revision", ""), fingerprint=data.get("fingerprint", ""))
        return previous
    except OSError as exc:
        log(f"could not swap the staged release: {exc}")
        receipt("swap_failed", error=str(exc))
        # put things back if the first rename half-happened
        if not known_good.exists() and previous.exists():
            try:
                previous.rename(known_good)
            except OSError:
                pass
        return None


def _restore(previous: Path | None, exe: Path, log, receipt, reason: str, port: int, token: str) -> int:
    """Put the previous release back and start it."""

    log(f"relaunch failed: {reason}")
    if previous is None or not (previous / "ZEUS.exe").is_file():
        receipt("failed_no_previous", reason=reason)
        log("no previous release to restore")
        return 3
    # Stop whatever the failed exe started, so the port and the mutex are free.
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{port}/api/shutdown", data=b"{}",
                                         headers={"X-Jarvis-Token": token, "Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=5).read()
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/IM", "ZEUS.exe"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    time.sleep(2.0)
    current = exe.parent
    failed = current.parent / f"ZEUS.failed.{int(time.time())}"
    try:
        if current.exists():
            current.rename(failed)
        previous.rename(current)
        log(f"restored {current} from {previous}; failed release kept at {failed}")
    except OSError as exc:
        log(f"could not restore the previous release: {exc}")
        receipt("restore_failed", reason=reason, error=str(exc))
        return 4
    try:
        pid = _start(current / "ZEUS.exe")
        log(f"started the restored release as pid {pid}")
        receipt("rolled_back", reason=reason, failed_release=str(failed))
    except OSError as exc:
        log(f"could not start the restored release: {exc}")
        receipt("rollback_start_failed", reason=reason, error=str(exc))
        return 5
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
