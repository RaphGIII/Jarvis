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
        # the desktop observer resumes with the server IF the owner enabled
        # it; constructing it is cheap and its thread runs only when opted in
        try:
            self.core.observer
        except Exception:  # noqa: BLE001 - a stub core in tests has no state root
            pass
        self._start_reminders()
        return self.url

    def _start_reminders(self) -> None:
        """A minute-beat that turns due calendar reminders into notifications."""

        if getattr(self, "_reminder_thread", None) is not None:
            return

        def beat() -> None:
            import time as _time

            while self._server is not None:
                try:
                    for event in self.core.calendar.due_reminders():
                        start = str(event.get("start", ""))[11:16]
                        self.core.emit(EventType.NOTIFICATION,
                                       {"kind": "reminder", "event": event,
                                        "text": f"Erinnerung: „{event.get('title', '')}“ um {start} Uhr."})
                except Exception:  # noqa: BLE001 - a reminder must never kill the beat
                    pass
                _time.sleep(60)

        self._reminder_thread = threading.Thread(target=beat, daemon=True, name="calendar-reminders")
        self._reminder_thread.start()

    @staticmethod
    def _corpus_phrases() -> dict[str, Any]:
        from speech.corpus import PHRASES

        return {"ok": True, "phrases": [{"category": c, "text": t} for c, t in PHRASES]}

    def _calendar_export(self, path: str) -> dict[str, Any]:
        from pathlib import Path as _P

        target = _P(path) if path else _P(self.core.kernel.state_root) / "calendar" / "zeus-kalender.ics"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.core.calendar.export_ics(), encoding="utf-8")
        return {"ok": True, "path": str(target)}

    def _calendar_import(self, path: str) -> dict[str, Any]:
        from pathlib import Path as _P

        source = _P(path)
        if not source.is_file():
            return {"ok": False, "error": f"keine Datei: {path}"}
        count = self.core.calendar.import_ics(source.read_text(encoding="utf-8", errors="replace"))
        return {"ok": True, "imported": count}

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
                str(body.get("text", "")), scope=str(body.get("scope", "")),
                meta={"source": str(body.get("source") or "text")}, request_id=str(body.get("request_id", "")),
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
            "/api/knowledge/create": lambda body: self.core.knowledge_create(
                str(body.get("title", "")), str(body.get("text", body.get("body", ""))), type=str(body.get("type", "note")),
                tags=list(body.get("tags") or []), links=body.get("links") or [], provenance=str(body.get("provenance", "owner")),
            ),
            "/api/knowledge/link": lambda body: self.core.knowledge_link(
                str(body.get("source", "")), str(body.get("target", "")), str(body.get("relation", "relates_to"))
            ),
            "/api/knowledge/read": lambda body: self.core.knowledge_read(str(body.get("id", body.get("title", "")))),
            "/api/knowledge/backlinks": lambda body: self.core.knowledge_backlinks(str(body.get("id", body.get("title", "")))),
            "/api/knowledge/delete": lambda body: self.core.knowledge_delete(str(body.get("id", body.get("title", ""))), confirm=bool(body.get("confirm", False))),
            "/api/knowledge/stats": lambda _: self.core.knowledge_stats(),
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
            "/api/stop": lambda body: self.core.stop_current(reason=str(body.get("reason", "owner")), session=str(body.get("session", ""))),
            "/api/voice/interrupt": lambda body: self.core.voice_interrupt(session=str(body.get("session", "")), wake=float(body.get("wake", 0) or 0)),
            "/api/voice/session": lambda body: self.core.voice_session_event(
                str(body.get("session", "")), str(body.get("state", "")), str(body.get("reason", "")), wake=float(body.get("wake", 0) or 0)
            ),
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
                category=str(body.get("category", "")),
            ),
            # The knowledge library: REAL files under one owner-visible root.
            "/api/library/tree": lambda _: self.core.library.tree(),
            "/api/library/folder": lambda body: self.core.library.create_folder(str(body.get("path", ""))),
            "/api/library/note": lambda body: self.core.library.write_note(
                str(body.get("folder", "")), str(body.get("title", "")), str(body.get("text", ""))),
            "/api/library/import": lambda body: self.core.library.import_file(
                str(body.get("source", "")), folder=str(body.get("folder", ""))),
            "/api/library/move": lambda body: self.core.library.move(str(body.get("path", "")), str(body.get("into", ""))),
            "/api/library/read": lambda body: self.core.library.read_note(str(body.get("path", ""))),
            "/api/pdf/extract": lambda body: self.core.pdf_extract(str(body.get("path", ""))),
            "/api/pdf/summarize": lambda body: self.core.pdf_summarize(
                str(body.get("path", "")), save=bool(body.get("save", True)), to_knowledge=bool(body.get("to_knowledge", True))),
            "/api/app/open": lambda body: self.core.apps.launch(str(body.get("name", ""))),
            # Desktop observation: opt-in, owner-controlled, fully auditable.
            "/api/observer/status": lambda _: self.core.observer.status(),
            "/api/observer/enable": lambda body: self.core.observer.set_enabled(bool(body.get("enabled", False))),
            "/api/observer/patterns": lambda body: self.core.observer.patterns(
                since_hours=float(body.get("since_hours", 72) or 72)),
            "/api/fs/roots": lambda _: self.core.fs.roots(),
            "/api/fs/list": lambda body: self.core.fs.list(str(body.get("path", "")), hidden=bool(body.get("hidden", False)),
                                                            files=bool(body.get("files", True))),
            "/api/fs/open": lambda body: self.core.fs.open_in_explorer(str(body.get("path", ""))),
            "/api/fs/watch": lambda body: self.core.fs.watch(str(body.get("path", ""))),
            "/api/fs/unwatch": lambda body: self.core.fs.unwatch(str(body.get("path", ""))),
            "/api/fs/status": lambda _: {"ok": True, **self.core.fs.status()},
            # The calendar: local-first, every mutation persisted immediately.
            "/api/calendar/list": lambda body: {"ok": True, "events": self.core.calendar.list(
                start=str(body.get("start", "")), end=str(body.get("end", "")), query=str(body.get("query", "")))},
            "/api/calendar/create": lambda body: {"ok": True, "event": self.core.calendar.create(
                title=str(body.get("title", "")), start=str(body.get("start", "")), end=str(body.get("end", "")),
                timezone=str(body.get("timezone", "")), location=str(body.get("location", "")),
                notes=str(body.get("notes", "")), project_id=str(body.get("project_id", "")),
                reminder_minutes=(int(body["reminder_minutes"]) if str(body.get("reminder_minutes", "")).strip() not in {"", "None"} else None),
                source=str(body.get("source", "ui")))},
            "/api/calendar/update": lambda body: (lambda ev: {"ok": ev is not None, "event": ev})(
                self.core.calendar.update(str(body.get("id", "")), **{k: v for k, v in (body.get("changes") or {}).items()})),
            "/api/calendar/delete": lambda body: {"ok": self.core.calendar.delete(str(body.get("id", "")))},
            "/api/calendar/export": lambda body: self._calendar_export(str(body.get("path", ""))),
            "/api/calendar/import": lambda body: self._calendar_import(str(body.get("path", ""))),
            # The owner speech corpus: verified recordings, and the benchmark
            # that turns them into WER/CER/latency numbers on this machine.
            "/api/corpus/phrases": lambda _: self._corpus_phrases(),
            "/api/corpus/list": lambda _: {"ok": True, **self.core.speech_corpus.stats(),
                                           "entries": [{k: v for k, v in e.items() if k != "audio"} | {"audio_name": Path(e["audio"]).name}
                                                       for e in self.core.speech_corpus.list()]},
            "/api/corpus/add": lambda body: {"ok": True, "entry": {
                **self.core.speech_corpus.add_base64(str(body.get("audio", "")), ext=str(body.get("ext", "webm")),
                                                     ground_truth=str(body.get("ground_truth", "")),
                                                     category=str(body.get("category", "")),
                                                     device=str(body.get("device", "")),
                                                     conditions=str(body.get("conditions", "")),
                                                     held_out=bool(body.get("held_out", False)))}},
            "/api/corpus/delete": lambda body: {"ok": self.core.speech_corpus.delete(str(body.get("id", "")))},
            "/api/corpus/benchmark": lambda body: self.core.corpus_benchmark(
                models=str(body.get("models", "small")), limit=int(body.get("limit", 0) or 0),
                held_out_only=bool(body.get("held_out_only", False))),
            "/api/corpus/reports": lambda _: self.core.corpus_reports(),
            # local image generation: a real file or an honest error
            "/api/image/generate": lambda body: self.core.imagegen.generate(
                str(body.get("prompt", "")), negative=str(body.get("negative", "")),
                size=str(body.get("size", "512x512")), steps=int(body.get("steps", 2) or 2),
                seed=int(body.get("seed", -1) if str(body.get("seed", "")).strip() not in {"", "None"} else -1)),
            "/api/image/status": lambda _: {**self.core.imagegen.available(), "busy": self.core.imagegen.busy},
            "/api/feedback": lambda body: self.core.feedback(
                str(body.get("kind", "response")), rating=str(body.get("rating", "")), category=str(body.get("category", "")),
                text=str(body.get("text", "")), request_id=str(body.get("request_id", "")),
                receipt_id=str(body.get("receipt_id", "")), session=str(body.get("session", "")),
            ),
            "/api/adaptation": lambda _: self.core.adaptation_rules(),
            "/api/adaptation/rule": lambda body: self.core.adaptation_rule(
                rule_id=str(body.get("rule_id", "")), action=str(body.get("action", "update")), text=str(body.get("text", "")),
                domain=str(body.get("domain", "STYLE")), scope=body.get("scope") if isinstance(body.get("scope"), dict) else None,
                changes=body.get("changes") if isinstance(body.get("changes"), dict) else None,
            ),
            # The password strings live only inside this call; handle_api and
            # the activity log never see or record these bodies.
            "/api/auth/status": lambda _: self.core.auth_status(),
            "/api/auth/setup": lambda body: self.core.auth_setup(str(body.get("password", "")), current=str(body.get("current", ""))),
            "/api/auth/unlock": lambda body: self.core.auth_unlock(str(body.get("password", "")), str(body.get("scope", "")),
                                                                   seconds=float(body.get("seconds", 0) or 0)),
            "/api/auth/lock": lambda body: self.core.auth_lock(str(body.get("scope", ""))),
            "/api/corrections": lambda _: self.core.list_corrections(),
            "/api/correction/update": lambda body: self.core.update_correction(
                str(body.get("correction_id", "")), dict(body.get("changes") or {})
            ),
            "/api/correction/delete": lambda body: self.core.delete_correction(str(body.get("correction_id", ""))),
            "/api/selfdev": lambda _: self.core.list_selfdev(),
            "/api/selfdev/cancel": lambda body: self.core.cancel_selfdev(str(body.get("mission_id", ""))),
            "/api/selfdev/resume": lambda body: self.core.resume_selfdev(str(body.get("mission_id", ""))),
            "/api/project/delete": lambda body: self.core.project_delete(
                str(body.get("id", "")), authorization=str(body.get("authorization", ""))),
            "/api/activity/correct": lambda body: self.core.activity_correct(
                request_id=str(body.get("request_id", "")), seq=int(body.get("seq", 0) or 0),
                correction_type=str(body.get("type", "TRANSCRIPT")), corrected_text=str(body.get("corrected", "")),
                original_text=str(body.get("original", "")), note=str(body.get("note", "")), rerun=bool(body.get("rerun", False))),
            "/api/activity/corrections": lambda body: self.core.activity_corrections(int(body.get("limit", 200) or 200)),
            "/api/owner": lambda _: self.core.owner_view(),
            "/api/owner/propose": lambda body: self.core.owner_propose(
                dict(body.get("changes") or {}), reason=str(body.get("reason", "")), origin="ui",
                unlock_core=bool(body.get("unlock_core", False)), authorization=str(body.get("authorization", "")),
            ),
            "/api/owner/personality": lambda _: self.core.owner_personality(),
            "/api/thoughts": lambda body: self.core.list_thoughts(status=str(body.get("status", ""))),
            "/api/thoughts/think": lambda body: self.core.think(str(body.get("trigger", "manual")), force=True, background=False),
            "/api/thoughts/act": lambda body: self.core.thought_action(str(body.get("id", body.get("thought_id", ""))), str(body.get("action", ""))),
            "/api/owner/approve": lambda body: self.core.owner_approve(
                str(body.get("transaction_id", "")), confirm=bool(body.get("confirm", False)),
                authorization=str(body.get("authorization", "")),
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
            "/api/missions": lambda body: self.core.list_missions(status=str(body.get("status", ""))),
            "/api/projects/overview": lambda _: self.core.projects_overview(),
            "/api/projects/graph": lambda body: self.core.project_graph(everything=bool(body.get("everything", False))),
            "/api/project/update": lambda body: self.core.project_update(
                str(body.get("id", "")), importance=str(body.get("importance", "")), hidden=body.get("hidden"),
                layout=dict(body["layout"]) if isinstance(body.get("layout"), dict) else None, note=str(body.get("note", ""))),
            "/api/project/timeline": lambda body: self.core.project_timeline(str(body.get("id", "")), limit=int(body.get("limit", 200) or 200)),
            "/api/timers": lambda _: self.core.list_timers(),
            "/api/backup": lambda _: {"backups": self.core.backups.list()},
            "/api/backup/create": lambda body: self.core.backup_create(str(body.get("label", ""))),
            "/api/backup/verify": lambda body: self.core.backups.verify(str(body.get("path", ""))),
            "/api/backup/restore": lambda body: self.core.backups.restore(str(body.get("path", "")), confirm=bool(body.get("confirm", False))),
            "/api/experience": lambda body: self.core.experience_view(str(body.get("goal", ""))),
            "/api/device/context/set": lambda body: __import__("runtime.device_context", fromlist=["set_context"]).set_context(
                self.core, **{k: v for k, v in body.items() if k in ("room", "name", "device_type", "speaker", "microphone", "inputs", "outputs")}).to_dict(),
            "/api/device/context": lambda _: self.core.device_context(),
            "/api/compose": lambda body: self.core.compose_preview(str(body.get("goal", ""))),
            "/api/mission": lambda body: self.core.mission_detail(str(body.get("id", body.get("mission_id", "")))),
            "/api/mission/cancel": lambda body: self.core.mission_control(str(body.get("mission_id", "")), "cancel"),
            "/api/mission/pause": lambda body: self.core.mission_control(str(body.get("mission_id", "")), "pause"),
            "/api/mission/resume": lambda body: self.core.mission_control(str(body.get("mission_id", "")), "resume"),
            "/api/capabilities/report": lambda _: self.core.capability_report(),
            "/api/tools/chess": lambda _: self.core.chess_tool_status(),
            "/api/tools/chess/start": lambda _: self.core.chess_tool_start(),
            "/api/tools/chess/stop": lambda _: self.core.chess_tool_stop(),
            "/api/voice/pronunciation": lambda body: self.core.pronunciation(str(body.get("text", "")), language=str(body.get("language", ""))),
            "/api/voice/pronunciation/set": lambda body: self.core.pronunciation_set(
                str(body.get("surface", "")), str(body.get("spoken", "")), language=str(body.get("language", "")), note=str(body.get("note", "")),
                test=bool(body.get("test", True))),
            "/api/voice/pronunciation/remove": lambda body: self.core.pronunciation_remove(str(body.get("surface", "")), language=str(body.get("language", ""))),
            "/api/voice/wake": lambda _: self.core.wake_status(),
            "/api/voice/wake/train": lambda _: self.core.wake_train(),
            "/api/voice/wake/evaluate": lambda _: self.core.wake_evaluate(),
            "/api/voice/wake/listener": lambda body: self.core.wake_listener_report(dict(body or {})),
            "/api/release": lambda _: self.core.release_status(),
            "/api/release/build": lambda body: self.core.release_build(verify=bool(body.get("verify", True))),
            "/api/release/verify": lambda body: self.core.release_verify(str(body.get("candidate", ""))),
            "/api/release/promote": lambda body: self.core.release_promote(
                str(body.get("candidate", "")), relaunch=bool(body.get("relaunch", True)),
                authorization=str(body.get("authorization", "")),
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
                # Who opened this listening session: the listener sends the
                # wake score that fired, the browser's microphone button says
                # "ui".  Audio with neither is transcribed but never acted on.
                wake = (query.get("wake") or [self.headers.get("X-Jarvis-Wake") or ""])[0] or None
                session = (query.get("session") or [self.headers.get("X-Jarvis-Session") or ""])[0]
                origin = (query.get("origin") or [self.headers.get("X-Jarvis-Origin") or ""])[0]
                # What the device measured while recording (X-Jarvis-Utterance,
                # -Speech-Seconds, -Noise-Floor, -Interrupted, ...): evidence for
                # the acceptance gate, and the utterance's identity.
                evidence: dict[str, Any] = {}
                for name in ("Utterance", "Source", "Speech-Seconds", "Noise-Floor", "Threshold", "Elapsed", "Started", "Interrupted", "Wake-At"):
                    value = self.headers.get(f"X-Jarvis-{name}")
                    if value is not None and value != "":
                        evidence[name.lower().replace("-", "_")] = value
                for key in ("utterance", "source"):
                    if query.get(key):
                        evidence[key] = query[key][0]
                try:
                    result = app.core.hear(raw, language=language, answer=answer, wake=wake, session=session, origin=origin, evidence=evidence)
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
                    # History is context, not something that is happening
                    # now: a client must render it as the past (no speech
                    # playback, no "new" user turn), so it is marked.
                    data = event.to_dict()
                    data["replay"] = True
                    if not self._write_event(data):
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
