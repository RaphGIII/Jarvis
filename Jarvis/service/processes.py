"""What ZEUS-related processes exist on this machine, and how to end them.

The lifecycle rules ("exactly one core, one listener, one worker, one window";
"fully quit means zero related processes") are only rules if something can
count.  This module counts, deterministically, from the operating system's
process table -- never from what a process says about itself.

Windows only for the enumeration (``Get-CimInstance Win32_Process`` gives the
command line and the parent pid, which ``tasklist`` does not); elsewhere the
answers are empty rather than wrong.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    parent: int
    name: str
    command: str

    def to_dict(self) -> dict:
        return {"pid": self.pid, "parent": self.parent, "name": self.name, "command": self.command[:200]}


#: How each ZEUS role shows up in a command line.
ROLES = {
    "core": ("jarvis.serve",),
    "listener": ("speech.listener",),
    "worker": ("speech.worker",),
    "supervisor": ("zeus_supervisor", "ZEUS.exe"),
}


def list_processes(patterns: Iterable[str]) -> list[ProcessInfo]:
    """Processes whose command line contains any of ``patterns`` (case-insensitive)."""

    patterns = [p.lower() for p in patterns if p]
    if sys.platform != "win32" or not patterns:
        return []
    script = ("Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name, CommandLine | "
              "ConvertTo-Json -Compress")
    try:
        completed = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True,
                                   timeout=40, creationflags=_no_window(), encoding="utf-8", errors="replace")
        rows = json.loads(completed.stdout or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    if isinstance(rows, dict):
        rows = [rows]
    out = []
    for row in rows:
        command = str(row.get("CommandLine") or "")
        lowered = command.lower()
        if any(p in lowered for p in patterns):
            out.append(ProcessInfo(int(row.get("ProcessId") or 0), int(row.get("ParentProcessId") or 0),
                                   str(row.get("Name") or ""), command))
    return out


def alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill(pid: int, *, tree: bool = False) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    if sys.platform == "win32":
        command = ["taskkill", "/F", "/PID", str(pid)] + (["/T"] if tree else [])
        completed = subprocess.run(command, capture_output=True, creationflags=_no_window())
        return completed.returncode == 0
    try:
        os.kill(pid, 9)
        return True
    except OSError:
        return False


#: Images that can be a ZEUS process.  A shell whose command string merely
#: mentions ``zeus_supervisor`` is not one.
ZEUS_IMAGES = {"python.exe", "pythonw.exe", "zeus.exe", "python", "python3"}


def zeus_processes() -> dict[str, list[ProcessInfo]]:
    """Every ZEUS role's processes, from the process table.

    A Windows venv's ``Scripts/python.exe`` is a launcher that starts the real
    interpreter as a child with the *same* command line; that pair is one
    process for every purpose here, and is counted once (the parent, whose
    death takes the child with it).
    """

    everything = list_processes([p for pats in ROLES.values() for p in pats])
    by_pid = {info.pid: info for info in everything}
    by_role: dict[str, list[ProcessInfo]] = {role: [] for role in ROLES}
    for info in everything:
        if info.name.lower() not in ZEUS_IMAGES:
            continue
        parent = by_pid.get(info.parent)
        if parent is not None and parent.command == info.command and parent.name.lower() in ZEUS_IMAGES:
            continue  # the launcher's child: same command line, counted with its parent
        lowered = info.command.lower()
        for role, pats in ROLES.items():
            if any(p.lower() in lowered for p in pats):
                by_role[role].append(info)
                break
    return by_role


def counts() -> dict[str, int]:
    return {role: len(rows) for role, rows in zeus_processes().items()}


def kill_orphans(role: str, *, keep: Iterable[int] = ()) -> list[int]:
    """End processes of ``role`` whose parent is gone (a killed core or supervisor
    leaves its speech worker behind), except the pids in ``keep``."""

    keep_set = {int(p) for p in keep}
    killed = []
    for info in zeus_processes().get(role, []):
        if info.pid in keep_set or info.pid == os.getpid():
            continue
        if info.parent and alive(info.parent):
            continue
        if kill(info.pid, tree=True):
            killed.append(info.pid)
    return killed


def kill_role(role: str, *, keep: Iterable[int] = ()) -> list[int]:
    """End every process of ``role`` except ``keep`` (and this process)."""

    keep_set = {int(p) for p in keep}
    killed = []
    for info in zeus_processes().get(role, []):
        if info.pid in keep_set or info.pid == os.getpid():
            continue
        if kill(info.pid, tree=True):
            killed.append(info.pid)
    return killed
