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
import time
import webbrowser

from service.core import JarvisCore
from service.http import JarvisHTTPServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jarvis.serve", description="Run the Jarvis core service and web interface."
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: loopback only)")
    parser.add_argument("--port", type=int, default=8420, help="port (0 picks a free one)")
    parser.add_argument("--token", default="", help="shared token; generated when omitted")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument("--persona", default="Jarvis", help="persona name")
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

    core = JarvisCore(persona_name=args.persona)
    server = JarvisHTTPServer(core, host=args.host, port=args.port, token=args.token)
    url = server.start()

    # Start loading models immediately. The page is served either way; this
    # only decides whether the first question takes one second or fifty.
    if not args.no_warm:
        core.warm(speech=not args.no_speech)

    print(f"\n  Jarvis is running.\n\n    {url}\n")
    print("  Ctrl-C to stop.\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n  stopping…")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
