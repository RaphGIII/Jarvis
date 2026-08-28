"""Does the interface still work after Jarvis edited it?

``python -m jarvis.verify_ui`` -- the health check that gates promoting a
UI change.

The hard part of letting a model edit its own interface is that a broken UI
fails silently.  A syntax error in ``app.js`` does not stop the server, does not
fail a test, and does not appear anywhere except a blank page in a browser
nobody has open.  Promotion needs something that says no.

So the checks run in order of how conclusive they are:

1. *It is served.*  A real server is started and every asset fetched.  This is
   the only check that exercises the actual path the browser takes, and it
   catches a file deleted, renamed, or moved outside the UI root.
2. *The page still has its parts.*  Element ids the client code looks up by
   name -- remove ``#eye`` and ``startJarvis`` throws on load, with no other
   symptom.
3. *The scripts are structurally intact.*  Balanced delimiters outside strings
   and comments. This is not a JavaScript parser and does not pretend to be;
   it catches truncation and mangled edits, which is the failure that actually
   happens when a model rewrites a file.

Nothing here judges whether the change is *good*.  That is the user's call.
This only refuses to promote something that cannot work at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Element ids the client looks up by name.  Losing one is a runtime failure
#: with no other visible symptom.
REQUIRED_IDS = (
    "app", "eye", "stateLabel", "detail", "log", "input",
    "btnSend", "btnMic", "connPill", "panel", "panelBody",
    # the operating environment: workspace, inspector, palette, HUD
    "workspacePane", "workspaceTitle", "inspector", "inspectorBody", "palette", "paletteInput", "hud", "readiness", "rail",
)

#: Files the page cannot render without.  app.js is an ES module and imports
#: the rest; a missing module is a blank page with one console line, which
#: is exactly the failure a verifier exists to catch before promotion.
REQUIRED_ASSETS = (
    "index.html", "eye.js", "graph.js", "app.js", "zeus.css",
    "core/dom.js", "core/api.js", "core/bus.js", "core/state.js", "core/views.js",
    "views/chat.js", "views/activity.js", "views/projects.js", "views/missions.js", "views/knowledge.js",
    "views/corrections.js", "views/diagnostics.js", "views/owner.js", "views/release.js", "views/capabilities.js",
    "views/voice.js", "views/palette.js", "voice/mic.js", "voice/playback.js",
)

#: Names the page depends on existing in its scripts.
#: The last statement of an entry script; without it the page boots nothing.
REQUIRED_TAIL = {"app.js": "startJarvis();"}

REQUIRED_SYMBOLS = {
    "eye.js": ("JarvisEye",),
    "app.js": ("startJarvis",),
}


@dataclass
class UIReport:
    ok: bool = True
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            self.ok = False

    def describe(self) -> str:
        lines = []
        for check in self.checks:
            lines.append(("  ok   " if check["ok"] else "  FAIL ") + check["name"] +
                         (f" -- {check['detail']}" if check["detail"] else ""))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks}


def check_assets(ui_root: Path, report: UIReport) -> None:
    for name in REQUIRED_ASSETS:
        path = ui_root / name
        if not path.is_file():
            report.add(f"{name} exists", False, f"missing from {ui_root}")
        elif path.stat().st_size == 0:
            report.add(f"{name} exists", False, "the file is empty")
        else:
            report.add(f"{name} exists", True, f"{path.stat().st_size} bytes")


def check_page(ui_root: Path, report: UIReport) -> None:
    index = ui_root / "index.html"
    if not index.is_file():
        return
    markup = index.read_text(encoding="utf-8", errors="replace")

    missing = [name for name in REQUIRED_IDS if f'id="{name}"' not in markup]
    report.add(
        "the page still has its parts",
        not missing,
        f"missing element id(s): {', '.join(missing)}" if missing else f"{len(REQUIRED_IDS)} ids present",
    )

    for script in ("eye.js", "app.js"):
        report.add(
            f"{script} is loaded by the page",
            f'src="{script}"' in markup,
            "" if f'src="{script}"' in markup else "the page does not reference it",
        )

    report.add(
        "the token placeholder survives",
        "__JARVIS_TOKEN__" in markup,
        "" if "__JARVIS_TOKEN__" in markup else "the client cannot authenticate without it",
    )


def check_scripts(ui_root: Path, report: UIReport) -> None:
    # The entry point must still be *called* at the end of the entry script:
    # a truncation that lands between two top-level blocks keeps every brace
    # balanced and every definition present, and boots nothing.
    for name, tail in REQUIRED_TAIL.items():
        path = ui_root / name
        source = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        ok = source.rstrip().endswith(tail)
        report.add(f"{name} ends with {tail}", ok, "" if ok else "the entry call is missing: truncated?")
    for name, symbols in REQUIRED_SYMBOLS.items():
        path = ui_root / name
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8", errors="replace")

        balanced, detail = delimiters_balanced(source)
        report.add(f"{name} is structurally intact", balanced, detail)

        missing = [symbol for symbol in symbols if not defines(source, symbol)]
        report.add(
            f"{name} still defines what the page needs",
            not missing,
            f"missing: {', '.join(missing)}" if missing else "",
        )


def defines(source: str, symbol: str) -> bool:
    """Whether the source DEFINES a symbol, not merely mentions it.

    Substring matching is not enough and the difference is not academic:
    renaming `function startJarvis` to `function boot` leaves
    `window.startJarvis = startJarvis` behind, so the name is still present in
    the file while nothing defines it. An adversarial test caught exactly that.
    """

    import re as _re

    escaped = _re.escape(symbol)
    # function startJarvis() / class JarvisEye / const startJarvis = ...
    keywords = ('function', 'class', 'const', 'let', 'var')
    lead = r'(?<![A-Za-z0-9_$])'
    trail = r'(?![A-Za-z0-9_$])'
    forms = [lead + word + r'\s+' + escaped + trail for word in keywords]
    return any(_re.search(pattern, source) for pattern in forms)


def delimiters_balanced(source: str) -> tuple[bool, str]:
    """Balanced braces, brackets and parentheses, ignoring strings and comments.

    Not a JavaScript parser, and it does not pretend to be one -- template
    literals with nested expressions and regex literals are approximated. It
    exists to catch truncation and mangled edits, which is the failure mode that
    actually occurs when a model rewrites a file, and for that it is reliable.
    """

    pairs = {"}": "{", "]": "[", ")": "("}
    stack: list[tuple[str, int]] = []
    line = 1
    index = 0
    length = len(source)

    while index < length:
        char = source[index]
        if char == "\n":
            line += 1
            index += 1
            continue

        # Comments
        if char == "/" and index + 1 < length:
            following = source[index + 1]
            if following == "/":
                while index < length and source[index] != "\n":
                    index += 1
                continue
            if following == "*":
                index += 2
                while index + 1 < length and not (source[index] == "*" and source[index + 1] == "/"):
                    if source[index] == "\n":
                        line += 1
                    index += 1
                index += 2
                continue

        # Strings and template literals
        if char in "\"'`":
            quote = char
            index += 1
            while index < length:
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == "\n":
                    line += 1
                    if quote != "`":
                        break          # an unterminated ordinary string ends at the line
                if source[index] == quote:
                    index += 1
                    break
                index += 1
            continue

        if char in "{[(":
            stack.append((char, line))
        elif char in "}])":
            if not stack:
                return False, f"unexpected '{char}' at line {line}"
            opener, opened_at = stack.pop()
            if opener != pairs[char]:
                return False, f"'{opener}' opened at line {opened_at} closed by '{char}' at line {line}"
        index += 1

    if stack:
        opener, opened_at = stack[-1]
        return False, f"'{opener}' opened at line {opened_at} is never closed"
    return True, "balanced"


def check_modules(ui_root: Path, report: UIReport) -> None:
    """Every module's delimiters balance and every relative import resolves.

    The page is an ES module graph with no build step, so a typo in one
    import is a blank page with a single console line.  Break any module on
    purpose and this is the check that turns red.
    """

    import re as _re

    pattern = _re.compile(r"(?:import|export)\s+(?:[^;]*?\s+from\s+)?[\"'](\.{1,2}/[^\"']+)[\"']")
    for name in [n for n in REQUIRED_ASSETS if n.endswith(".js")]:
        path = ui_root / name
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        ok, detail = delimiters_balanced(source)
        report.add(f"{name} is well-formed", ok, detail)
        for match in pattern.finditer(source):
            target = (path.parent / match.group(1)).resolve()
            report.add(f"{name} imports {match.group(1)}", target.is_file(), "" if target.is_file() else f"missing: {target}")


def check_served(ui_root: Path, report: UIReport) -> None:
    """Start a real server and fetch every asset the browser would."""

    try:
        from service.core import JarvisCore
        from service.http import JarvisHTTPServer
    except ImportError as exc:
        report.add("the UI is served", False, f"cannot import the service: {exc}")
        return

    class _Kernel:
        """Enough kernel to serve files, with nothing that touches a model."""

        state_root = Path(ui_root).parent / "data" / "ui-health"
        catalog = type("C", (), {"get": staticmethod(lambda tier: type("S", (), {"model": ""})())})()

        def provider(self, tier):
            raise RuntimeError("the health check does not generate")

    server = None
    try:
        core = JarvisCore(kernel=_Kernel())
        server = JarvisHTTPServer(core, port=0, token="health-check", ui_root=ui_root)
        server.start()
        base = f"http://{server.host}:{server.port}"

        for name in REQUIRED_ASSETS:
            path = "/?token=health-check" if name == "index.html" else f"/{name}"
            try:
                with urllib.request.urlopen(base + path, timeout=15) as response:
                    body = response.read()
                    ok = response.status == 200 and len(body) > 0
                    detail = f"{response.status}, {len(body)} bytes"
            except Exception as exc:
                ok, detail = False, str(exc)
            report.add(f"{name} is served", ok, detail)

        # The token must actually be substituted, or the page loads and then
        # fails to open the event stream -- which looks like "Jarvis is down".
        try:
            with urllib.request.urlopen(base + "/?token=health-check", timeout=15) as response:
                page = response.read().decode("utf-8", errors="replace")
            substituted = "__JARVIS_TOKEN__" not in page and "health-check" in page
            report.add("the served page carries a usable token", substituted,
                       "" if substituted else "the placeholder was not replaced")
        except Exception as exc:
            report.add("the served page carries a usable token", False, str(exc))
    finally:
        if server is not None:
            server.stop()


def verify(ui_root: str | Path, *, serve: bool = True) -> UIReport:
    root = Path(ui_root).resolve()
    report = UIReport()
    if not root.is_dir():
        report.add("the UI directory exists", False, str(root))
        return report

    check_assets(root, report)
    check_page(root, report)
    check_scripts(root, report)
    check_modules(root, report)
    if serve:
        check_served(root, report)
    return report


def default_ui_root() -> Path:
    return Path(__file__).resolve().parent.parent / "ui"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m jarvis.verify_ui",
        description="Check that the Jarvis interface can still load and be served.",
    )
    parser.add_argument("--ui-root", default="", help="defaults to this repository's ui/ directory")
    parser.add_argument("--no-serve", action="store_true", help="skip the live serving check")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = verify(args.ui_root or default_ui_root(), serve=not args.no_serve)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.describe())
        print("\nUI_OK" if report.ok else "\nUI_BROKEN")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
