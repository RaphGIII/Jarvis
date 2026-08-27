"""``python -m zeus_supervisor`` -- the same entry point ZEUS.exe is built from.

    run        start ZEUS under supervision (default)
    check      run the preflight and print the report
    status     print the supervisor's status file and the known-good pointer
    stop       ask a running ZEUS to shut down
    restart    ask a running ZEUS to restart
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from . import __version__
from .config import SupervisorConfig, find_repository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zeus", description="Supervise ZEUS.")
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "check", "status", "stop", "quit", "restart", "receipts", "show"])
    parser.add_argument("--repo", default=None, help="the Jarvis directory (default: discovered)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--version", action="version", version=f"ZEUS supervisor {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo).resolve() if args.repo else find_repository()
    try:
        config = SupervisorConfig.load(
            repo, port=args.port,
            open_browser=False if args.no_browser else None,
            voice=False if args.no_voice else None,
        )
    except FileNotFoundError as exc:
        print(f"ZEUS: {exc}", file=sys.stderr)
        return 2

    if args.command == "check":
        from .preflight import Preflight

        print(f"ZEUS supervisor {__version__}")
        print(json.dumps(config.to_dict(), indent=2))
        report = Preflight(config).run()
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ok else 1

    if args.command in {"status", "receipts"}:
        from .control import ControlChannel
        from .known_good import KnownGoodStore

        channel = ControlChannel(config.state_dir)
        store = KnownGoodStore(config.state_dir)
        if args.command == "status":
            print(json.dumps({"status": channel.read_status(), "known_good": store.load().to_dict()}, indent=2))
        else:
            for receipt in store.history(limit=30):
                print(json.dumps(receipt, sort_keys=True))
        return 0

    if args.command in {"stop", "quit", "restart", "show"}:
        token_path = config.state_dir / "token"
        if not token_path.is_file():
            print("ZEUS is not running under the supervisor (no token file)", file=sys.stderr)
            return 1
        token = token_path.read_text(encoding="utf-8").strip()
        endpoint = {"stop": "shutdown", "quit": "quit", "restart": "restart", "show": "window/show"}[args.command]
        url = f"http://{config.host}:{config.port}/api/{endpoint}"
        request = urllib.request.Request(
            url, data=json.dumps({"reason": f"zeus {args.command} from the command line"}).encode("utf-8"),
            headers={"X-Jarvis-Token": token, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                print(response.read().decode("utf-8"))
        except Exception as exc:
            print(f"could not reach ZEUS: {exc}", file=sys.stderr)
            return 1
        return 0

    from .supervisor import Supervisor

    return Supervisor(config, log=print if not getattr(sys, "frozen", False) else None).run()


def _frozen_main() -> int:
    """No console, so a crash must go to a file and a message box, not a dialog
    titled 'Unhandled exception in script' with a traceback nobody can read."""

    import io
    import traceback

    # PyInstaller's --noconsole leaves stdout/stderr as None; anything that
    # prints would then raise. Give them somewhere to go.
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()
    try:
        return main()
    except SystemExit as exc:
        return int(exc.code or 0)
    except BaseException:  # noqa: BLE001 - last resort, must not hide anything
        text = traceback.format_exc()
        log = Path(sys.executable).resolve().parent / "ZEUS-error.log"
        try:
            log.write_text(text, encoding="utf-8")
        except OSError:
            pass
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
                None, f"ZEUS could not start.\n\n{text[-900:]}\n\nSaved to {log}", "ZEUS", 0x10
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(_frozen_main() if getattr(sys, "frozen", False) else main())
