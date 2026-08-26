"""What the browser sees while ZEUS itself is not answering.

The UI's address is the core's port.  When the core is down -- booting, being
rolled back, held after a boot loop -- a browser pointed there would get a
connection error and nothing else, which tells the owner nothing.  So the
supervisor holds the port itself in those moments and serves one page: what it
is doing, what failed, and what would fix it.  The page refreshes itself and
disappears the moment the real interface is back.

It is released *before* the core is launched, so the two never compete for the
address.
"""

from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ZEUS</title>
<meta http-equiv="refresh" content="3">
<style>
body{margin:0;background:#05080f;color:#c8d6e5;font:15px/1.5 system-ui,Segoe UI,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh}
.card{max-width:640px;padding:32px 40px;border:1px solid #1b2a3d;border-radius:12px;background:#0a1220}
h1{margin:0 0 4px;font-size:22px;letter-spacing:.2em;color:#8fd3ff}
.phase{color:#6f8aa5;font-size:12px;text-transform:uppercase;letter-spacing:.15em;margin-bottom:16px}
.eye{width:64px;height:64px;border-radius:50%%;margin:0 auto 20px;background:radial-gradient(circle,%(colour)s 0%%,#05080f 70%%);animation:b 2.4s ease-in-out infinite}
@keyframes b{0%%,100%%{transform:scale(.92);opacity:.7}50%%{transform:scale(1);opacity:1}}
.detail{white-space:pre-wrap;color:#e6eef7}
.remedy{margin-top:14px;padding:10px 12px;background:#101b2b;border-left:3px solid #ffb454;color:#ffd9a0;white-space:pre-wrap}
.log{margin-top:18px;font:12px/1.4 ui-monospace,Consolas,monospace;color:#6f8aa5;white-space:pre-wrap;max-height:200px;overflow:auto}
</style></head><body><div class="card"><div class="eye"></div>
<h1>ZEUS</h1><div class="phase">%(phase)s</div>
<div class="detail">%(detail)s</div>%(remedy)s<div class="log">%(log)s</div></div></body></html>"""


class StatusPage:
    def __init__(self, host: str, port: int, snapshot: Callable[[], dict[str, Any]]) -> None:
        self.host = host
        self.port = port
        self.snapshot = snapshot
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> bool:
        if self._server is not None:
            return True
        page = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                state = page.snapshot()
                if self.path.startswith("/api/") or self.path.startswith("/events"):
                    body = json.dumps({"supervisor": True, "ready": False, **state}, default=str).encode("utf-8")
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                else:
                    colour = {"error": "#ff5a5a", "held": "#ff5a5a"}.get(str(state.get("phase", "")).lower(), "#3fa9ff")
                    remedy = state.get("remedy", "")
                    body = (PAGE % {
                        "colour": colour,
                        "phase": html.escape(str(state.get("phase", "starting"))),
                        "detail": html.escape(str(state.get("detail", ""))),
                        "remedy": f'<div class="remedy">{html.escape(str(remedy))}</div>' if remedy else "",
                        "log": html.escape("\n".join(str(l) for l in state.get("log", [])[-12:])),
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            do_POST = do_GET  # noqa: N815

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError:
            self._server = None
            return False
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="zeus-status-page")
        self._thread.start()
        return True

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
