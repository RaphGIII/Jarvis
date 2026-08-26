"""Start Jarvis: ``python -m jarvis.serve``.

One command, as the brief requires.  Prints the URL with its token, opens a
browser unless told not to, and stays in the foreground so Ctrl-C stops it.

Deliberately thin.  Everything it does is available programmatically through
:class:`~service.core.JarvisCore` and :class:`~service.http.JarvisHTTPServer`,
because a future Windows service, a login task or a headless server deployment
must be able to start Jarvis without going through an entry point designed for
a terminal.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

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
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    # Empty rather than a name: the assistant's identity comes from
    # config/identity.json, and a default here silently overrode it.
    parser.add_argument("--persona", default="", help="persona name (default: the configured identity)")
    parser.add_argument("--no-warm", action="store_true", help="do not preload models at startup")
    parser.add_argument("--no-speech", action="store_true", help="do not preload the speech stack")
    return parser


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

    core = JarvisCore(persona_name=args.persona)
    # A planned restart saved the transcript; a fresh start finds nothing.
    resumed = core.lifecycle.restore_conversation()
    server = JarvisHTTPServer(core, host=args.host, port=args.port, token=token)
    url = server.start()

    # Start loading models immediately. The page is served either way; this
    # only decides whether the first question takes one second or fifty.
    if not args.no_warm:
        core.warm(speech=not args.no_speech)

    identity = core.identity
    print(f"\n  {identity.product_name} is running.\n\n    {url}\n")
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

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

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
        server.stop()
    return core.lifecycle.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
