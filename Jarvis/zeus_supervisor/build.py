"""Build ``ZEUS.exe``: ``python -m zeus_supervisor.build``.

What the executable is, honestly: the supervisor, frozen.  It does not contain
ZEUS, Python, Ollama or any model, and it does not pretend to.  It knows where
those are (``install.json`` beside it), checks each of them on every start, and
tells the owner in plain words what is missing.  That is the right shape for a
program that rewrites itself: the part that recovers a broken installation is
the one part the installation cannot rewrite, because it is a compiled file
sitting outside the source tree.

    dist/ZEUS/ZEUS.exe        the launcher (no console window)
    dist/ZEUS/install.json    repository path, interpreter, build metadata
    dist/ZEUS/VERSION.json    version, git revision, build time

``--shortcuts`` also puts "ZEUS" on the Desktop and in the Start Menu.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import find_repository

HERE = Path(__file__).resolve().parent


def build(repository: Path, *, dist: Path | None = None, shortcuts: bool = False, onefile: bool = False) -> Path:
    dist = dist or (repository.parent / "dist")
    work = repository.parent / "build" / "pyinstaller"
    dist.mkdir(parents=True, exist_ok=True)
    entry = HERE / "launch.py"
    command = [
        sys.executable, "-m", "PyInstaller",
        "--name", "ZEUS", "--noconfirm", "--clean", "--noconsole",
        "--onefile" if onefile else "--onedir",
        "--distpath", str(dist), "--workpath", str(work), "--specpath", str(work),
        "--paths", str(repository),
        "--hidden-import", "zeus_supervisor.supervisor",
        "--hidden-import", "zeus_supervisor.preflight",
        "--hidden-import", "zeus_supervisor.build", "--hidden-import", "zeus_supervisor.__main__",
        str(entry),
    ]
    icon = repository / "ui" / "zeus.ico"
    if icon.is_file():
        command[3:3] = ["--icon", str(icon)]
    print("+", " ".join(command))
    subprocess.run(command, check=True, cwd=str(repository))

    out_dir = dist if onefile else dist / "ZEUS"
    exe = out_dir / "ZEUS.exe"
    if not exe.is_file():
        raise FileNotFoundError(f"PyInstaller finished but {exe} does not exist")

    revision = ""
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repository), capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    (out_dir / "install.json").write_text(json.dumps({
        "repository": str(repository),
        "python": sys.executable,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    (out_dir / "VERSION.json").write_text(json.dumps({
        "product": "ZEUS", "supervisor_version": __version__, "revision": revision,
        "python": sys.version.split()[0], "built_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    print(f"built {exe} ({exe.stat().st_size // 1024} KB), revision {revision[:12]}")

    if shortcuts and sys.platform == "win32":
        for name, target_dir in (("Desktop", _desktop_dir()),
                                 ("Start Menu", Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs")):
            if not target_dir.is_dir():
                continue
            link = target_dir / "ZEUS.lnk"
            script = (
                f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{link}');"
                f"$s.TargetPath='{exe}';$s.WorkingDirectory='{out_dir}';"
                f"$s.Description='Start ZEUS';"
                + (f"$s.IconLocation='{icon}';" if icon.is_file() else "")
                + "$s.Save()"
            )
            completed = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True)
            print(f"{name} shortcut: {'ok' if completed.returncode == 0 else completed.stderr.strip()[:200]} -> {link}")
    return exe


def _desktop_dir() -> Path:
    """Where the Desktop really is: OneDrive moves it, and the registry knows."""

    try:
        import winreg

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            value, _ = winreg.QueryValueEx(key, "Desktop")
            return Path(os.path.expandvars(str(value)))
    except Exception:
        return Path.home() / "Desktop"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ZEUS.exe")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--dist", default=None)
    parser.add_argument("--onefile", action="store_true")
    parser.add_argument("--shortcuts", action="store_true", help="create Desktop and Start Menu shortcuts")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve() if args.repo else find_repository()
    if repo is None:
        print("could not find the repository", file=sys.stderr)
        return 2
    if shutil.which("pyinstaller") is None:
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            print("PyInstaller is not installed: python -m pip install pyinstaller", file=sys.stderr)
            return 2
    build(repo, dist=Path(args.dist).resolve() if args.dist else None, shortcuts=args.shortcuts, onefile=args.onefile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
