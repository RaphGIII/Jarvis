"""One ZEUS per machine.

Two supervisors would fight over the port, the GPU and the repository.  On
Windows a named mutex is the reliable answer: it is released by the kernel when
the process dies, so a crash cannot leave a stale lock behind.  Elsewhere a
pid file does the job.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MUTEX_NAME = "Global\\ZEUS.Supervisor.Instance"


class InstanceLock:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self._handle = None
        self._pidfile = self.state_dir / "supervisor.pid"
        self.held = False

    def acquire(self) -> bool:
        if sys.platform == "win32":
            return self._acquire_mutex()
        return self._acquire_pidfile()

    def _acquire_mutex(self) -> bool:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
        already = kernel32.GetLastError() == 183  # ERROR_ALREADY_EXISTS
        if not handle or already:
            if handle:
                kernel32.CloseHandle(handle)
            return False
        self._handle = handle
        self.held = True
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self._pidfile.write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            pass
        return True

    def _acquire_pidfile(self) -> bool:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self._pidfile.is_file():
            try:
                pid = int(self._pidfile.read_text(encoding="utf-8").strip() or 0)
                if pid and pid != os.getpid():
                    os.kill(pid, 0)
                    return False
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        self._pidfile.write_text(str(os.getpid()), encoding="utf-8")
        self.held = True
        return True

    def release(self) -> None:
        if self._handle is not None:
            import ctypes

            ctypes.windll.kernel32.CloseHandle(self._handle)  # type: ignore[attr-defined]
            self._handle = None
        if self.held:
            try:
                self._pidfile.unlink()
            except OSError:
                pass
        self.held = False
