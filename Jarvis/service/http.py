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
        """The core token, or a paired device's own credential.

        Two ways in rather than one, because a device must be revocable without
        changing the secret every other client is using -- which is the whole
        reason per-device tokens exist.
        """

        supplied = headers.get("X-Jarvis-Token") or (query.get("token") or [""])[0]
        if supplied and secrets.compare_digest(str(supplied), self.token):
            return True

        device_id = headers.get("X-Jarvis-Device") or (query.get("device") or [""])[0]
        if device_id and supplied:
            try:
                return self.core.gateway.authenticate(str(device_id), str(supplied)) is not None
            except Exception:
                return False
        return False

    def handle_api(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Dispatch a JSON API call to the core."""

        routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "/api/message": lambda body: self.core.send_message(
                str(body.get("text", "")), scope=str(body.get("scope", ""))
            ),
            "/api/status": lambda _: self.core.status(),
            # Polled far more often than /api/status, because the whole value
            # of a live load readout is that it is live. Answered from a
            # cached background reading, so the extra polling costs nothing.
            "/api/gpu": lambda _: self.core.gpu_usage(),
            "/api/state": lambda _: self.core.state.snapshot.to_dict(),
            "/api/projects": lambda _: {"projects": self.core.list_projects()},
            "/api/project": lambda body: self.core.project_detail(str(body.get("id", ""))),
            "/api/capabilities": lambda _: {"capabilities": self.core.list_capabilities()},
            # Receipts are the record of everything that actually changed. They
            # are on the API rather than only in the event stream because a
            # client that was not connected at the time still has to be able to
            # ask what this system has done.
            # What the Activity view reads. Durable, so a client that was not
            # connected at the time still sees everything that happened.
            "/api/activity": lambda body: self.core.list_activity(int(body.get("limit", 200) or 200)),
            "/api/receipts": lambda body: self.core.list_receipts(int(body.get("limit", 50) or 50)),
            "/api/receipt": lambda body: self.core.receipt(str(body.get("id", ""))),
            "/api/knowledge/graph": lambda body: self.core.knowledge_graph(
                query=str(body.get("query", "")), limit=int(body.get("limit", 300) or 300)
            ),
            "/api/knowledge/node": lambda body: self.core.knowledge_node(str(body.get("id", ""))),
            # Pairing is the only device endpoint reachable without a device
            # credential, because a device that has none is exactly what it is
            # for. It still requires the core token, so only something already
            # trusted enough to reach the API may ask.
            "/api/device/pair": lambda body: self.core.device_pair_request(
                str(body.get("name", "")),
                str(body.get("kind", "generic")),
                list(body.get("capabilities") or []),
            ),
            "/api/device/collect": lambda body: self.core.device_pair_collect(str(body.get("code", ""))),
            "/api/device/approve": lambda body: self.core.device_approve(str(body.get("code", ""))),
            "/api/device/deny": lambda body: self.core.device_deny(str(body.get("code", ""))),
            "/api/device/list": lambda _: self.core.device_list(),
            "/api/device/revoke": lambda body: self.core.device_revoke(str(body.get("device_id", ""))),
            "/api/device/heartbeat": lambda body: self.core.device_heartbeat(str(body.get("device_id", ""))),
            "/api/device/display": lambda body: self.core.device_display(
                str(body.get("device_id", "")), str(body.get("command", "")), dict(body.get("payload") or {})
            ),
            "/api/research": lambda body: self.core.research(
                str(body.get("question", "")), max_sources=int(body.get("max_sources", 3) or 3)
            ),
            "/api/knowledge/do": lambda body: self.core.graph_operation(
                str(body.get("request", "")),
                selected=str(body.get("selected", "")),
                confirm=bool(body.get("confirm", False)),
            ),
            "/api/knowledge/ingest": lambda body: self.core.ingest(
                str(body.get("path", "")),
                text=str(body.get("text", "")),
                title=str(body.get("title", "")),
                recursive=bool(body.get("recursive", True)),
                max_files=int(body.get("max_files", 500) or 500),
            ),
            # refresh=true asks for live probes, which cost a real generation
            # on every tier. Opt-in, because the default must not slow down the
            # conversation it is reporting on.
            "/api/diagnostics": lambda body: self.core.diagnostics(
                refresh=str(body.get("refresh", "")).lower() in {"1", "true", "yes"}
            ),
            "/api/voice": lambda body: self.core.voice_settings(
                **{
                    key: body[key]
                    for key in ("enabled", "language", "voice_id", "speak_replies", "microphone", "output", "voice",
                                "wake_sensitivity", "volume")
                    if key in body
                }
            ),
            "/api/personas": lambda _: self.core.list_personas(),
            "/api/persona": lambda body: self.core.set_persona(str(body.get("name", ""))),
            "/api/language": lambda body: self.core.set_language(str(body.get("language", ""))),
            "/api/stop": lambda _: self.core.stop_current(),
            "/api/new": lambda _: self.core.new_conversation(),
            # The supervisor's contract. READY here means the conversation
            # model produced real text in this process -- the port being open
            # is what the supervisor already knows.
            # The owner core. Reading is open to any authenticated client;
            # approve and rollback are the only writers the five documents
            # have, and they exist only here -- no model, capability, expert
            # or ingestion path holds a reference to them.
            # Korrigieren. Saving a correction is an owner act from the live
            # interface; nothing that reads documents or runs models can reach
            # these routes.
            "/api/correction/context": lambda body: self.core.correction_context(str(body.get("receipt_id", ""))),
            "/api/correction/classify": lambda body: self.core.correction_classify(
                str(body.get("what_was_wrong", "")), receipt_id=str(body.get("receipt_id", ""))
            ),
            "/api/correction/save": lambda body: self.core.correction_save(
                str(body.get("what_was_wrong", "")), receipt_id=str(body.get("receipt_id", "")),
                classification=str(body.get("classification", "")), scope=str(body.get("scope", "")),
                original_request=str(body.get("original_request", "")), rerun=bool(body.get("rerun", False)),
            ),
            "/api/corrections": lambda _: self.core.list_corrections(),
            "/api/correction/update": lambda body: self.core.update_correction(
                str(body.get("correction_id", "")), dict(body.get("changes") or {})
            ),
            "/api/correction/delete": lambda body: self.core.delete_correction(str(body.get("correction_id", ""))),
            "/api/selfdev": lambda _: self.core.list_selfdev(),
            "/api/selfdev/cancel": lambda body: self.core.cancel_selfdev(str(body.get("mission_id", ""))),
            "/api/selfdev/resume": lambda body: self.core.resume_selfdev(str(body.get("mission_id", ""))),
            "/api/owner": lambda _: self.core.owner_view(),
            "/api/owner/propose": lambda body: self.core.owner_propose(
                dict(body.get("changes") or {}), reason=str(body.get("reason", "")), origin="ui"
            ),
            "/api/owner/approve": lambda body: self.core.owner_approve(
                str(body.get("transaction_id", "")), confirm=bool(body.get("confirm", False))
            ),
            "/api/owner/reject": lambda body: self.core.owner_reject(str(body.get("transaction_id", ""))),
            "/api/owner/rollback": lambda body: self.core.owner_rollback(
                str(body.get("audit_id", "")), confirm=bool(body.get("confirm", False))
            ),
            "/api/health": lambda _: self.core.lifecycle.health(),
            "/api/window": lambda body: self.core.lifecycle.window(
                str(body.get("action", "status")), reason=str(body.get("reason", ""))
            ),
            "/api/window/show": lambda body: self.core.lifecycle.window("show", reason=str(body.get("reason", "second launch"))),
            "/api/window/hide": lambda body: self.core.lifecycle.window("hide", reason=str(body.get("reason", ""))),
            "/api/processes": lambda _: self.core.lifecycle.process_counts(),
            "/api/doctor": lambda _: self.core.doctor(),
            "/api/search": lambda body: self.core.universal_search(str(body.get("q", body.get("query", ""))), limit=int(body.get("limit", 30) or 30),
                                                                   types=[t for t in str(body.get("types", "")).split(",") if t]),
            "/api/selfdev/diff": lambda body: self.core.selfdev_diff(str(body.get("mission_id", ""))),
            "/api/capabilities/report": lambda _: self.core.capability_report(),
            "/api/voice/wake": lambda _: self.core.wake_status(),
            "/api/voice/wake/train": lambda _: self.core.wake_train(),
            "/api/release": lambda _: self.core.release_status(),
            "/api/release/build": lambda body: self.core.release_build(verify=bool(body.get("verify", True))),
            "/api/release/verify": lambda body: self.core.release_verify(str(body.get("candidate", ""))),
            "/api/release/promote": lambda body: self.core.release_promote(
                str(body.get("candidate", "")), relaunch=bool(body.get("relaunch", True))
            ),
            "/api/release/rollback": lambda body: self.core.release_rollback(confirm=bool(body.get("confirm", False))),
            "/api/quit": lambda body: self.core.lifecycle.request_quit(
                str(body.get("reason", "owner asked ZEUS to quit completely")), requested_by=str(body.get("requested_by", "ui"))
            ),
            "/api/supervisor": lambda _: self.core.lifecycle.supervisor_status(),
            "/api/restart": lambda body: self.core.lifecycle.request_restart(
                str(body.get("reason", "restart requested")),
                expected_revision=str(body.get("expected_revision", "")),
                promotion_id=str(body.get("promotion_id", "")),
                requested_by=str(body.get("requested_by", "api")),
            ),
            "/api/shutdown": lambda body: self.core.lifecycle.request_shutdown(
                str(body.get("reason", "shutdown requested")), requested_by=str(body.get("requested_by", "api"))
            ),
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

            if parsed.path in {"/api/voice/wake/record", "/api/voice/wake/test"}:
                try:
                    if parsed.path.endswith("record"):
                        result = app.core.wake_record(raw, kind=(query.get("kind") or ["positive"])[0])
                    else:
                        result = app.core.wake_test(raw)
                except Exception as exc:
                    self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                    return
                self._send_json(200, result)
                return

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
                # The product name is substituted here rather than baked into
                # the file, so renaming is configuration and the page stays a
                # single document with no build step.
                identity = getattr(app.core, "identity", None)
                if identity is not None:
                    body = body.replace(b"__PRODUCT_NAME__", identity.product_name.encode("utf-8"))
                    body = body.replace(b"__ASSISTANT_NAME__", identity.assistant_name.encode("utf-8"))
                    injected = (
                        'window.ASSISTANT_NAME = "'
                        + identity.assistant_name.replace('"', "")
                        + '";\n  window.JARVIS_TOKEN ='
                    )
                    body = body.replace(b"window.JARVIS_TOKEN =", injected.encode("utf-8"))
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
