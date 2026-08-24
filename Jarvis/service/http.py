"""The local HTTP + Server-Sent Events interface to Jarvis Core.

Why SSE and not WebSockets: nothing web-related is installed on this machine and
the brief forbids modifying global packages, so the choice was between adding a
dependency, hand-rolling RFC 6455 framing, or using the streaming transport the
standard library and every browser already support.  What the UI actually needs
is a *server-to-client event stream* -- state changes, tokens, progress, tool
activity -- with client input arriving as ordinary requests.  That is exactly
the shape SSE has, so the dependency would have bought bidirectionality nobody
is using.  The transport is isolated behind :class:`JarvisHTTPServer` and
:mod:`service.events`, so swapping in WebSockets later touches this file only.

Two properties are enforced here rather than left to the caller:

*Local only, with a token.*  The server binds to loopback and requires a shared
token on every request except the UI itself.  It exposes file access, shell
tools and a model that can write code; reachable-from-the-LAN-by-default would
be indefensible.

*A dead client cannot wedge the core.*  Each SSE connection drains its own
bounded subscription and is dropped when the socket breaks.
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from service.events import EventBus, EventType

#: Where the built-in UI lives.
UI_ROOT = Path(__file__).resolve().parent.parent / "ui"


class JarvisHTTPServer:
    """Serves the UI, the event stream, and a small JSON API."""

    def __init__(
        self,
        core: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 8420,
        token: str = "",
        ui_root: Path | None = None,
    ) -> None:
        self.core = core
        self.host = host
        self.port = port
        self.token = token or secrets.token_urlsafe(24)
        self.ui_root = Path(ui_root or UI_ROOT)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------

    @property
    def bus(self) -> EventBus:
        return self.core.bus

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/?token={self.token}"

    def start(self) -> str:
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._server.daemon_threads = True
        # Port 0 means "any free port"; report the one actually bound.
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="jarvis-http")
        self._thread.start()
        return self.url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> "JarvisHTTPServer":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # -- routing ---------------------------------------------------------

    def authorised(self, headers: Any, query: dict[str, list[str]]) -> bool:
        supplied = headers.get("X-Jarvis-Token") or (query.get("token") or [""])[0]
        return secrets.compare_digest(str(supplied), self.token)

    def handle_api(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Dispatch a JSON API call to the core."""

        routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "/api/message": lambda body: self.core.send_message(
                str(body.get("text", "")), scope=str(body.get("scope", ""))
            ),
            "/api/status": lambda _: self.core.status(),
            "/api/state": lambda _: self.core.state.snapshot.to_dict(),
            "/api/projects": lambda _: {"projects": self.core.list_projects()},
            "/api/project": lambda body: self.core.project_detail(str(body.get("id", ""))),
            "/api/capabilities": lambda _: {"capabilities": self.core.list_capabilities()},
            "/api/knowledge/graph": lambda body: self.core.knowledge_graph(
                query=str(body.get("query", "")), limit=int(body.get("limit", 300) or 300)
            ),
            "/api/knowledge/node": lambda body: self.core.knowledge_node(str(body.get("id", ""))),
            "/api/diagnostics": lambda _: self.core.diagnostics(),
            "/api/voice": lambda body: self.core.voice_settings(
                **{
                    key: body[key]
                    for key in ("enabled", "language", "voice_id", "speak_replies")
                    if key in body
                }
            ),
            "/api/personas": lambda _: self.core.list_personas(),
            "/api/persona": lambda body: self.core.set_persona(str(body.get("name", ""))),
            "/api/language": lambda body: self.core.set_language(str(body.get("language", ""))),
            "/api/stop": lambda _: self.core.stop_current(),
        }
        handler = routes.get(path)
        if handler is None:
            return 404, {"error": f"no such endpoint: {path}"}
        try:
            return 200, handler(payload)
        except Exception as exc:  # surfaced to the client rather than swallowed
            return 500, {"error": f"{type(exc).__name__}: {exc}"}


def _make_handler(app: JarvisHTTPServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Jarvis"
        protocol_version = "HTTP/1.1"

        # -- plumbing ---------------------------------------------------

        def log_message(self, *_: Any) -> None:
            """Silence the default stderr access log."""

        def _send(self, status: int, body: bytes, content_type: str, *, extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        # -- GET --------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            path = parsed.path

            if path in {"", "/"}:
                self._serve_ui("index.html", query)
                return
            if path == "/events":
                if not app.authorised(self.headers, query):
                    self._send_json(401, {"error": "unauthorised"})
                    return
                self._stream_events(query)
                return
            if path.startswith("/api/voice/audio/"):
                if not app.authorised(self.headers, query):
                    self._send_json(401, {"error": "unauthorised"})
                    return
                self._serve_audio(path.rsplit("/", 1)[-1])
                return
            if path.startswith("/api/"):
                if not app.authorised(self.headers, query):
                    self._send_json(401, {"error": "unauthorised"})
                    return
                status, payload = app.handle_api(path, {key: value[0] for key, value in query.items()})
                self._send_json(status, payload)
                return
            self._serve_ui(path.lstrip("/"), query)

        # -- POST -------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if not app.authorised(self.headers, query):
                self._send_json(401, {"error": "unauthorised"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length else b"{}"

            if parsed.path == "/api/voice/utterance":
                # Audio is posted as bytes rather than base64 in JSON: a 30
                # second utterance is about a megabyte, and encoding it would
                # cost a third more bandwidth and a copy on both sides.
                answer = (query.get("answer") or ["1"])[0] not in {"0", "false", "no"}
                language = (query.get("language") or [""])[0]
                try:
                    result = app.core.hear(raw, language=language, answer=answer)
                except Exception as exc:
                    self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                    return
                self._send_json(200, result)
                return

            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except ValueError:
                self._send_json(400, {"error": "body must be JSON"})
                return
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "body must be a JSON object"})
                return
            status, response = app.handle_api(parsed.path, payload)
            self._send_json(status, response)

        def _serve_audio(self, name: str) -> None:
            audio = app.core.voice.store.get(name.removesuffix(".wav"))
            if audio is None:
                # Expected rather than exceptional: the store is bounded, so a
                # client that comes back for old audio finds it gone.
                self._send_json(404, {"error": "audio expired"})
                return
            self._send(200, audio.to_wav(), "audio/wav")

        # -- static UI --------------------------------------------------

        def _serve_ui(self, relative: str, query: dict[str, list[str]]) -> None:
            root = app.ui_root.resolve()
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                self._send_json(403, {"error": "path escapes the UI root"})
                return
            if not target.is_file():
                self._send_json(404, {"error": f"not found: {relative}"})
                return
            body = target.read_bytes()
            if target.name == "index.html":
                # The page needs the token to open the event stream. It is
                # already in the URL the user was given; inlining it avoids a
                # second round trip and keeps the token out of any file.
                supplied = (query.get("token") or [""])[0]
                body = body.replace(b"__JARVIS_TOKEN__", supplied.encode("utf-8"))
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type == "application/javascript":
                content_type = f"{content_type}; charset=utf-8"
            self._send(200, body, content_type)

        # -- SSE --------------------------------------------------------

        def _stream_events(self, query: dict[str, list[str]]) -> None:
            try:
                since = int((query.get("since") or ["0"])[0])
            except ValueError:
                since = 0

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            subscription = app.bus.subscribe(replay=False)
            try:
                for event in app.bus.history(since=since):
                    if not self._write_event(event.to_dict()):
                        return
                # Tell the client the current state immediately, so a page
                # opened mid-mission renders the truth rather than "idle".
                self._write_event(
                    {
                        "type": EventType.STATE.value,
                        "payload": app.core.state.snapshot.to_dict(),
                        "seq": app.bus.sequence,
                        "at": "",
                        "scope": "",
                    }
                )
                while True:
                    event = subscription.get(timeout=15.0)
                    if event is None:
                        # A comment frame keeps proxies and idle sockets alive
                        # and detects a client that has gone away.
                        if not self._write_raw(b": keepalive\n\n"):
                            return
                        continue
                    if not self._write_event(event.to_dict()):
                        return
            finally:
                subscription.close()

        def _write_event(self, payload: dict[str, Any]) -> bool:
            data = json.dumps(payload)
            return self._write_raw(f"event: {payload['type']}\ndata: {data}\n\n".encode("utf-8"))

        def _write_raw(self, chunk: bytes) -> bool:
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                return False

    return Handler
