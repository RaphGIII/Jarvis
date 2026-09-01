"""The frozen entry point.  PyInstaller runs its script as ``__main__`` with no
parent package, so this file uses absolute imports only and hands straight to
the real main with the crash handler around it."""

from __future__ import annotations

import sys


def run() -> int:
    try:
        from zeus_supervisor.__main__ import _frozen_main, main
    except BaseException:  # noqa: BLE001
        import traceback
        from pathlib import Path

        text = traceback.format_exc()
        try:
            (Path(sys.executable).resolve().parent / "ZEUS-error.log").write_text(text, encoding="utf-8")
        except OSError:
            pass
        try:
            import ctypes

            # A box that closes by itself: a blocking dialog on an unattended
            # machine is indistinguishable from a hang.
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            timeout_box = getattr(user32, "MessageBoxTimeoutW", None)
            if timeout_box is not None:
                timeout_box(None, f"ZEUS could not start.\n\n{text[-900:]}", "ZEUS", 0x10, 0, 30000)
        except Exception:
            pass
        return 1
    return _frozen_main() if getattr(sys, "frozen", False) else main()


if __name__ == "__main__":
    raise SystemExit(run())
