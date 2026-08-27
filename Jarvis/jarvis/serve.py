"""Start Jarvis: ``python -m jarvis.serve``.

One command, as the brief requires.  Prints the URL with its token, opens the
interface in its own desktop window unless told not to, and stays in the
foreground so Ctrl-C stops it.

Deliberately thin.  Everything it does is available programmatically through
:class:`~service.core.JarvisCore` and :class:`~service.http.JarvisHTTPServer`,
because a future Windows service, a login task or a headless server deployment
must be able to start Jarvis without going through an entry point designed for
a terminal.

The interface is an application window rather than a browser tab (see
:mod:`jarvis.window`), because that is what the owner asked for and because the
supervisor -- which starts this process for ``ZEUS.exe`` -- passes
``--no-browser`` and therefore has to be able to say "no browser" without also
saying "no interface".  Hence two independent switches: ``--no-browser``
declines the browser, ``--no-window`` declines the window, and only both
together mean nothing is opened.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

from jarvis.window import open_window
from service.core import JarvisCore
from service.http import JarvisHTTPServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jarvis.serve", description="Run the Jarvis core service and web interface."
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: loopback only)")
    parser.add_argument("--port", type=int, default=8420, help="port (0 picks a free one)")
    parser.add_argument("--token", default="", help="shared token; generated when omitted")
    parser.add_argument("--token-file", default="", help="read the shared token from this file (the supervisor's way)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not use a web browser (the desktop window still opens)")
    parser.add_argument("--no-window", action="store_true",
                        help="do not open the desktop window (falls back to a browser unless --no-browser)")
    parser.add_argument("--browser", action="store_true",
                        help="use the ordinary web browser instead of a desktop window")
    # Empty rather than a name: the assistant's identity comes from
    # config/identity.json, and a default here silently overrode it.
    parser.add_argument("--persona", default="", help="persona name (default: the configured identity)")
    parser.add_argument("--no-warm", action="store_true", help="do not preload models at startup")
    parser.add_argument("--no-speech", action="store_true", help="do not preload the speech stack")
    return parser


def interface_plan(args: argparse.Namespace, environ: dict[str, str] | None = None) -> tuple[str, bool]:
    """What to open -- ``("window"|"browser"|"none", fall back to a browser)``.

    ``ZEUS_UI`` (``window``, ``browser`` or ``none``) decides the same thing
    from the environment, for the callers that cannot pass arguments: a login
    task, a service wrapper, or a headless box where opening anything at all
    would be wrong.  An explicit flag still wins over it.
    """

    env = str((environ if environ is not None else os.environ).get("ZEUS_UI", "")).strip().lower()
    if env in {"none", "off", "0", "false", "headless"}:
        return "none", False

    no_window = bool(args.no_window or args.browser or env == "browser")
    no_browser = bool(args.no_browser or env == "window")
    if no_window:
        return ("none", False) if no_browser else ("browser", False)
    return "window", not no_browser


def show_interface(url: str, plan: tuple[str, bool], core: Any = None) -> str:
    """Open the interface as planned and report what the owner will see.

    With a ``core``, the window is a managed :class:`service.desktop.DesktopWindow`
    owned by its lifecycle: found again after a restart, hidden and shown on
    request, given its own Windows identity.  Without one (tests, the bare
    CLI) it is the plain launch it always was.
    """

    mode, fallback = plan
    if mode == "none":
        return ""
    if mode == "browser":
        try:
            return "in the default browser" if webbrowser.open(url) else "nowhere -- no browser would open"
        except Exception as exc:  # noqa: BLE001 - a missing browser is not a reason to abort the boot
            return f"nowhere -- the browser would not open: {exc}"
    if core is not None:
        desktop = core.lifecycle.attach_desktop(url)
        if desktop is not None:
            shown = desktop.show(reason="startup")
            desktop.start_watcher()
            if shown.get("ok"):
                return f"in its own window ({desktop.status().get('engine', 'window')}, {shown.get('action')} in {shown.get('seconds')}s)"
            if not fallback:
                return f"{shown.get('detail', 'the window did not open')} -- point ZEUS_WINDOW_BROWSER at one, or open the URL above"
    launch = open_window(url, fallback=fallback)
    if not launch.ok:
        # Said out loud rather than swallowed: the service is up and reachable,
        # so this is a "nothing appeared" the owner would otherwise diagnose as
        # "ZEUS is down".
        return f"{launch.detail} -- point ZEUS_WINDOW_BROWSER at one, or open the URL above"
    return launch.describe()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        # Not forbidden -- a home server is a stated goal -- but it must be a
        # decision rather than a default. This process exposes file access,
        # shell tools and a model that writes code.
        print(
            f"WARNING: binding to {args.host} exposes Jarvis beyond this machine.\n"
            "         The token is the only thing protecting it. Use a firewall or a tunnel.\n",
            file=sys.stderr,
        )

    token = args.token
    if args.token_file:
        token = Path(args.token_file).read_text(encoding="utf-8").strip()

    # Process hygiene before anything starts: a worker or listener whose
    # parent died with the last core would otherwise hold the microphone and
    # the GPU next to the new ones.
    try:
        from service.processes import kill_orphans

        swept = kill_orphans("worker") + kill_orphans("listener")
    except Exception:  # noqa: BLE001
        swept = []

    core = JarvisCore(persona_name=args.persona)
    # A planned restart saved the transcript; a fresh start finds nothing.
    resumed = core.lifecycle.restore_conversation()
    server = JarvisHTTPServer(core, host=args.host, port=args.port, token=token)
    url = server.start()
    core.lifecycle.mark("http", True, url)

    # Before warming, not after: warming loads a 4B model and the speech stack,
    # and the owner should be looking at the interface while that happens
    # rather than at nothing. The page renders its own loading state.
    shown = show_interface(url, interface_plan(args), core)
    if swept:
        print(f"  Swept {len(swept)} orphaned speech process(es): {swept}\n")

    # Start loading models immediately. The page is served either way; this
    # only decides whether the first question takes one second or fifty.
    if not args.no_warm:
        core.warm(speech=not args.no_speech)

    identity = core.identity
    print(f"\n  {identity.product_name} is running.\n\n    {url}\n")
    if shown:
        print(f"  Interface: {shown}\n")
    note = identity.wake_word_note()
    if note:
        # Said at startup rather than discovered at the microphone: the spoken
        # wake word and the trained model can differ, and finding that out by
        # talking to something that is not listening is the worst way to learn it.
        print(f"  {note}\n")
    if resumed:
        print(f"  Resumed {resumed['turns']} turn(s) from before the restart ({resumed['reason']}).\n")
    if core.lifecycle.supervised:
        print("  Running under the ZEUS supervisor.\n")
    print("  Ctrl-C to stop.\n")

    try:
        # Sleeps until a restart or shutdown is requested through the API;
        # the exit code tells the supervisor which.
        while not core.lifecycle.exit_event.wait(timeout=3600):
            pass
        if core.lifecycle.exit_reason:
            print(f"\n  exiting: {core.lifecycle.exit_reason}")
    except KeyboardInterrupt:
        print("\n  stopping…")
    finally:
        # A restart keeps the window (the page reconnects on its own); a
        # shutdown ends everything this process owns: window, speech worker.
        core.lifecycle.leave(final=core.lifecycle.exit_code == 0)
        server.stop()
    return core.lifecycle.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
