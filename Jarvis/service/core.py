"""Jarvis Core: the system, with no user interface attached to it.

Everything a client can ask for is a method here, and every client -- browser,
CLI, TV, future device -- goes through the same ones.  The rule that keeps this
honest is that :mod:`service.core` may not import anything from a UI, and a
method may not format anything for display.  It returns data and publishes
events; how that looks is somebody else's problem.

The persona sits at this layer rather than in a provider, because identity must
survive a backend change.  When a question is answered by the 4B model and the
next by a subscription expert, the user is still talking to Jarvis, and nothing
should announce "I am Qwen" because the router happened to pick a different
tier.  Diagnostics tell the truth about the backend when asked; ordinary
conversation does not volunteer it.
"""

from __future__ import annotations

import subprocess
import sys
import json
from datetime import datetime, timezone
import re
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from service.events import EventBus, EventType
from service.state import JarvisState, StateMachine


@dataclass
class ConversationTurn:
    role: str
    text: str
    at: str = ""
    #: Which tier answered.  Recorded always, shown only in diagnostics.
    backend: str = ""
    #: What this turn contributes to the *next* prompt, when that should differ
    #: from what the user was shown.  An action turn displays its full evidence
    #: block -- paths, byte counts, every check -- and that is right for a
    #: reader and wrong for a transcript: a model shown three receipts saying
    #: "erstellt / geschrieben / ok" starts producing that language itself, and
    #: the next ordinary answer trips the claim guard. Empty means "same text".
    context_text: str = ""
    #: How the turn arrived (a wake session, a press): metadata, never content.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {"role": self.role, "text": self.text, "at": self.at, "backend": self.backend}
        if self.meta:
            data["meta"] = dict(self.meta)
        return data

    def for_prompt(self, *, limit: int = 400) -> str:
        return (self.context_text or self.text)[:limit]


class JarvisCore:
    """The service object.  Owns state, events, and the long-lived subsystems.

    Subsystems are created lazily and never in ``__init__``.  Constructing the
    core must stay cheap and side-effect free: the HTTP server, the CLI and the
    tests all build one, and probing a model or opening a database on
    construction would make starting the UI depend on a GPU being warm.
    """

    #: How long a health probe's answer is trusted.  A real generation is the
    #: only honest health check and costs ~80 s cold, so it cannot be on the
    #: request path; this is how stale the UI's badge may be.
    HEALTH_TTL_SECONDS = 120.0
    #: Expert availability costs a subprocess, not a generation.
    EXPERT_TTL_SECONDS = 60.0

    def __init__(
        self,
        *,
        kernel: Any = None,
        bus: EventBus | None = None,
        persona_name: str = "",
        identity: Any = None,
    ) -> None:
        from core.identity import current as current_identity

        self.bus = bus or EventBus()
        self.state = StateMachine(on_change=self._publish_state)
        # The product name is a setting, not a spelling. Internal names stay as
        # they are -- this class is still JarvisCore -- because none of that is
        # user-facing and churning it would be risk spent on nothing.
        self.identity = identity or current_identity()
        self.persona_name = persona_name or self.identity.assistant_name
        self._kernel = kernel
        self._expert_gateway: Any = None
        self._lock = threading.Lock()
        self._history: list[ConversationTurn] = []
        self._current_work: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._started_at = time.time()
        self._health_ok = False
        self._health_detail = ""
        self._health_checked_at = 0.0
        self._probe_running = threading.Event()
        self._expert_status: dict[str, Any] = {"expert_available": False, "quota_exhausted": False}
        self._expert_checked_at = 0.0
        self._expert_probe_running = threading.Event()
        #: Live GPU load, sampled off the request path.  Created lazily like
        #: every other subsystem, so constructing the core still touches no
        #: hardware.
        self._gpu_usage: Any = None
        self._voice: Any = None
        self._personas: Any = None
        self._gateway: Any = None
        self._receipts: Any = None
        self._actions: Any = None
        self._activity: Any = None
        self._preferences: Any = None
        self._secrets: Any = None
        self._capabilities: Any = None
        self._music: Any = None
        #: Guards capability acquisition: two missions for the same capability
        #: running at once would fight over the same workspace.
        self._acquiring = threading.Lock()
        #: Consecutive failures per capability. One failure is an incident;
        #: two in a row is a defect worth rebuilding for. Reset by any success.
        self._defects: dict[str, int] = {}
        #: Receipts produced during this conversation, in memory.  The claim
        #: guard consults them on every streamed chunk, and re-reading the
        #: ledger file that often would cost more than the check saves.
        self._session_receipts: list[Any] = []
        #: The language the conversation is currently in.  Sticky: it changes
        #: only on a confident detection, because flipping mid-conversation
        #: changes the recogniser hint and the voice, which sounds worse than
        #: occasionally answering in the wrong language.
        self.language = ""
        #: Readiness, restart and shutdown -- the supervisor's view of this
        #: process. Constructed here because it must exist before warm().
        from service.lifecycle import Lifecycle

        self.lifecycle = Lifecycle(self)

    # ------------------------------------------------------------------
    # Lazy subsystems
    # ------------------------------------------------------------------

    @property
    def kernel(self) -> Any:
        if self._kernel is None:
            from core.kernel import JarvisKernel

            self._kernel = JarvisKernel()
        return self._kernel

    @property
    def personas(self) -> Any:
        if self._personas is None:
            from persona.profiles import PersonaStore

            self._personas = PersonaStore(self.kernel.state_root / "personas.json")
        return self._personas

    @property
    def voice(self) -> Any:
        if self._voice is None:
            from service.voice import VoiceService

            try:
                settings_path = Path(self.kernel.state_root) / "voice" / "settings.json"
            except Exception:  # noqa: BLE001 - a stub kernel without a state root
                settings_path = None
            self._voice = VoiceService(self.bus, settings_path=settings_path)
        return self._voice

    @property
    def gateway(self) -> Any:
        if self._gateway is None:
            from devices.gateway import DeviceGateway

            self._gateway = DeviceGateway(self.kernel.state_root / "devices.json")
        return self._gateway

    @property
    def preferences(self) -> Any:
        """What the user prefers -- provider choice, output device, and so on."""

        if self._preferences is None:
            from runtime.preferences import Preferences

            self._preferences = Preferences(self.kernel.state_root / "preferences.json")
        return self._preferences

    @property
    def secrets(self) -> Any:
        if self._secrets is None:
            from runtime.secrets import SecretStore

            self._secrets = SecretStore(self.kernel.state_root / "secrets")
        return self._secrets

    @property
    def capabilities(self) -> Any:
        """The capability service: what ZEUS can do, and how it learns more."""

        if self._capabilities is None:
            from capabilities.registry import CapabilityRegistry
            from capabilities.service import CapabilityService

            root = self.kernel.state_root / "capabilities"
            graph = None
            try:
                from knowledge.graph import KnowledgeGraph

                graph = self.graph
            except Exception:
                # The graph is what lets "play something" reach a capability
                # named after a provider. Without it lookup degrades to lexical
                # matching, which is worse but still works.
                graph = None
            from projects.engine import EngineHooks

            # Every step of a build is published. Without this an acquisition
            # is forty silent minutes followed by a verdict -- which is exactly
            # what Activity is supposed to stop being.
            def on_step(project: Any, step: Any) -> None:
                self.emit(
                    EventType.PROGRESS,
                    {
                        "summary": f"{getattr(step.phase, 'value', step.phase)}: {step.summary[:160]}",
                        "project": project.id,
                        "ok": bool(step.success),
                        "productive": bool(getattr(step, "productive", False)),
                        "step": getattr(step, "index", 0),
                    },
                )

            self._capabilities = CapabilityService(
                registry=CapabilityRegistry(root / "registry.json"),
                engine=self.kernel.engine(hooks=EngineHooks(on_step=on_step)),
                graph=graph,
                root=root / "installed",
                # A cold call starts PowerShell, fetches a token and searches over
                # the network. 120s was tight enough that a busy machine looked
                # like a broken capability.
                execution_timeout=240.0,
            )
        return self._capabilities

    @property
    def music(self) -> Any:
        if self._music is None:
            from service.music import MusicService

            self._music = MusicService(
                preferences=self.preferences,
                capabilities=self.capabilities,
                secrets=self.secrets,
            )
        return self._music

    @property
    def activity(self) -> Any:
        """The durable record of what happened, fed only by the event bus.

        Attached on first access rather than in ``__init__`` because touching
        it builds the kernel, and constructing the core must stay free.  The
        cost of that is that nothing is recorded until something asks -- so
        :meth:`warm` attaches it at startup, which is where the first events
        appear anyway.
        """

        if self._activity is None:
            from runtime.activity import ActivityLog

            self._activity = ActivityLog(self.kernel.state_root / "activity.jsonl").attach(self.bus)
        return self._activity

    @property
    def receipts(self) -> Any:
        """The durable record of every side effect this system has performed."""

        if self._receipts is None:
            from runtime.receipts import ReceiptLedger

            self._receipts = ReceiptLedger(self.kernel.state_root / "receipts.jsonl")
        return self._receipts

    @property
    def actions(self) -> Any:
        """The only thing here that may change the world."""

        if self._actions is None:
            from service.actions import ActionExecutor

            self._actions = ActionExecutor(self.kernel)
        return self._actions

    @property
    def experts(self) -> Any:
        if self._expert_gateway is None:
            from experts.claude_code import ClaudeCodeExpert
            from experts.gateway import ExpertGateway

            self._expert_gateway = ExpertGateway([ClaudeCodeExpert()])
        return self._expert_gateway

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _publish_state(self, snapshot: Any) -> None:
        self.bus.publish(EventType.STATE, snapshot.to_dict())

    def emit(self, type: EventType, payload: dict[str, Any] | None = None, *, scope: str = "") -> None:
        self.bus.publish(type, payload or {}, scope=scope)

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[ConversationTurn]:
        with self._lock:
            return list(self._history)

    #: Where a USER turn may come from.  A message with no provenance never
    #: enters the conversation: every visible owner sentence must be traceable
    #: to a typed message, a listening session, a press, or an owner action.
    USER_SOURCES = frozenset({"text", "microphone", "ui_mic", "thought_inbox", "correction_rerun", "api", "cli", "test", "galaxy", "palette"})

    def send_message(self, text: str, *, scope: str = "", meta: dict[str, Any] | None = None, request_id: str = "") -> dict[str, Any]:
        """Accept user input and answer it, streaming tokens as events.

        Idempotent by ``request_id``: a websocket retry, a UI refresh or a
        replayed utterance carrying an id that was already accepted is
        refused rather than answered twice.
        """

        import uuid

        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "empty message"}
        meta = dict(meta or {})
        meta.setdefault("source", "text")
        if meta["source"] not in self.USER_SOURCES:
            return {"ok": False, "error": f"no provenance: {meta['source']!r} may not enter the conversation as the owner"}
        request_id = str(request_id or meta.get("request_id") or "") or uuid.uuid4().hex[:12]
        with self._lock:
            seen = getattr(self, "_requests_seen", None)
            if seen is None:
                seen = self._requests_seen = {}
            if request_id in seen:
                self.emit(EventType.DIAGNOSTIC, {"request": "duplicate refused", "request_id": request_id, "text": text[:80], "source": meta["source"]}, scope=scope)
                return {"ok": False, "duplicate": True, "request_id": request_id, "accepted": text}
            seen[request_id] = time.monotonic()
            for key in list(seen)[:-400]:
                del seen[key]
        meta["request_id"] = request_id

        # Touched before the first event is published, so the request that
        # started everything is in the record rather than missing from it.
        # `warm()` normally pays this cost at startup; here it is the guarantee.
        try:
            self.activity
        except Exception:
            pass

        from persona.language import stable_language

        self.language = stable_language(text, current=self.language)

        turn = ConversationTurn(role="user", text=text, at=_now(), meta=meta)
        with self._lock:
            self._history.append(turn)
        self.emit(EventType.USER_MESSAGE, turn.to_dict(), scope=scope)

        # Answering happens off the request thread so the HTTP call returns at
        # once and the client watches the event stream, which is what makes
        # "Jarvis starts speaking before the answer is finished" possible.
        thread = threading.Thread(
            target=self._answer_guarded, args=(text, scope), daemon=True, name="jarvis-answer"
        )
        with self._lock:
            self._current_work = thread
        self._stop_requested.clear()
        thread.start()
        return {"ok": True, "accepted": text, "request_id": request_id}

    def _answer_guarded(self, text: str, scope: str) -> None:
        """Run :meth:`_answer`, and never let it fail in silence.

        ``say()`` returns ``{"ok": True, "accepted": text}`` the moment the
        thread starts, because that is what lets Jarvis begin speaking before it
        has finished thinking. The consequence is that everything after that
        point is out of the request's reach: an exception here reaches a daemon
        thread's default handler, which prints to stderr nobody is reading, and
        the user simply never gets an answer. Accepted, no reply, no error --
        the request reported success and the work did not happen, which is the
        exact failure this system exists to prevent, in the one place nothing
        was watching for it.

        Found by a voice test whose stub kernel lacked ``state_root``: the
        preferences lookup raised inside this thread, nothing was spoken, and
        the only visible symptom was silence.

        The message goes out as an ERROR event *and* as a normal reply, because
        a client watching the stream should see the failure and a person waiting
        for an answer should be told there is not going to be one.
        """

        import traceback

        try:
            self._answer(text, scope)
        except BaseException as exc:  # noqa: BLE001 - a thread boundary swallows everything
            detail = f"{type(exc).__name__}: {exc}"
            # The traceback travels with the event rather than being re-raised.
            # Re-raising would put it on a stderr nobody is reading and end the
            # thread the same way it used to end; carrying it here means the
            # one place that can see the failure also carries what is needed to
            # diagnose it.
            try:
                self.emit(
                    EventType.ERROR,
                    {"error": detail, "traceback": traceback.format_exc()[-4000:]},
                    scope=scope,
                )
            except Exception:
                pass
            try:
                self.state.set(JarvisState.ERROR, detail=detail[:200])
                self._deliver(
                    f"Something went wrong while I was answering that: {detail}",
                    scope=scope, backend="error", final_state=JarvisState.ERROR,
                )
            except Exception:
                pass

    def _answer(self, text: str, scope: str) -> None:
        """Route the request, then let the right machinery answer it.

        This branch is the fix.  Before it, every message went to the model and
        the model's prose was the product's output -- so "create this file"
        produced a confident description of a file that did not exist, complete
        with an invented path.  Nothing was executed because there was nothing
        here that could execute.

        Conversation keeps exactly the path it had, which is what keeps it
        instant.  Everything with a truth condition goes somewhere that can
        establish it.
        """

        from service.intent import Intent, classify

        # Owner corrections are read BEFORE the route is chosen, so a route the
        # owner has corrected once is not taken again; the registry's names let
        # "repair your zip capability" find the thing it names.
        started = time.perf_counter()
        try:
            prior = self.corrections.relevant(text)
        except Exception:  # noqa: BLE001 - a broken store must not block routing
            prior = []
        try:
            names = [str(m.capability_id) for m in self.capabilities.registry.all()]
        except Exception:  # noqa: BLE001
            names = []
        classification = classify(text, corrections=prior, capability_names=names)
        route = classification.route
        self.emit(
            EventType.DIAGNOSTIC,
            {"classified": classification.to_dict(), "text": text[:120],
             "router_ms": round((time.perf_counter() - started) * 1000, 1)},
            scope=scope,
        )
        if route is not None:
            # Routing evidence in Activity: what was decided, on what, and what
            # was overruled.  A wrong route is then correctable rather than
            # invisible.
            self.emit(
                EventType.TOOL,
                {"summary": f"routed: {route.intent.value} ({route.confidence}) -> {classification.intent.value}",
                 "routing": route.to_dict(), "intent": classification.intent.value,
                 "source": "router", "text": text[:160]},
                scope=scope,
            )
            if route.corrections:
                try:
                    self.corrections.note_applied([c for c in prior if c.correction_id in route.corrections])
                except Exception:  # noqa: BLE001
                    pass

        # Semantic purpose before any domain parser: one top-level intent and,
        # where the operation is deterministic (projects, views, stop, the
        # owner's corrections), a typed action contract that is executed
        # without a model.  Word overlap never selects an operation here.
        from service.intents import TopIntent, understand

        try:
            titles = [str(p.get("title") or "") for p in self.list_projects()]
        except Exception:  # noqa: BLE001
            titles = []
        understanding = understand(text, route=route, project_titles=titles, capability_names=names)
        self.emit(EventType.TOOL, {"summary": f"understood: {understanding.top.value}" + (f" -> {understanding.action.operation}" if understanding.action else ""),
                                   "understanding": understanding.to_dict(), "source": "intents", "text": text[:160]}, scope=scope)
        if self._handle_pending_confirmation(text, scope):
            return
        if self._handle_coach(text, scope):
            return
        if self._resume_open_target(text, scope):
            return
        if self._resume_fs(text, scope):
            return
        if self._resume_calendar(text, scope):
            return
        if self._handle_alias_teach(text, scope):
            return
        if understanding.top is TopIntent.CONVERSATION and self._resume_clarification(text, scope):
            return
        if understanding.top is TopIntent.CORRECTION:
            self._answer_correction(text, scope, understanding.action)
            return
        if understanding.top is TopIntent.SELF_DEVELOPMENT and understanding.reason.startswith("asks ZEUS to find"):
            # "Finde den Fehler und repariere dich": the action router must
            # never swallow an explicit self-improvement request.
            self._answer_by_self_development(text, scope, classification=classification)
            return
        if understanding.top is TopIntent.CLARIFICATION and understanding.action is not None:
            self._ask_clarification(understanding.action, understanding.question, text, scope)
            return
        if understanding.top in {TopIntent.PROJECT_OPERATION, TopIntent.SYSTEM_CONTROL} and understanding.action is not None:
            if self._needs_confirmation(understanding.action, text, scope):
                return
            if understanding.top is TopIntent.SYSTEM_CONTROL:
                self._answer_by_system_control(understanding.action, text, scope)
            else:
                self._answer_by_project_operation(understanding.action, text, scope)
            return

        # CAPABILITY joins READ rather than going to the executor, because
        # "learn to do X" cannot be executed from a chat turn in this system.
        # Answering it conversationally invites "I can do that now" -- a
        # present-tense capability claim, which the claim guard does not catch
        # because nothing was claimed to have been *done*. The registry knows
        # what is actually installed; the model does not.
        if classification.intent is Intent.CAPABILITY and not text.rstrip().endswith("?"):
            self._answer_by_acquisition(text, scope)
            return
        if classification.intent in {Intent.READ, Intent.CAPABILITY}:
            self._answer_from_records(text, scope, classification)
            return
        if classification.intent is Intent.MUSIC:
            if self._needs_confirmation(None, text, scope, side_effect=True):
                return
            self._answer_musically(text, scope, classification)
            return
        if classification.intent is Intent.SELF_DEVELOPMENT:
            self._answer_by_self_development(text, scope, classification=classification)
            return
        if classification.intent is Intent.OWNER_CONFIG:
            self._answer_owner_config(text, scope, classification)
            return
        if classification.intent is Intent.CORRECTION:
            self._answer_correction(text, scope, None)
            return
        if route is not None and route.intent.value == "research":
            self._answer_by_research(text, scope)
            return
        if classification.intent.has_side_effect or understanding.is_action_request:
            if self._needs_confirmation(None, text, scope, side_effect=True):
                return
            if not classification.intent.has_side_effect:
                # The legacy classifier saw conversation; the request is an
                # imperative with an action verb.  It goes to the executor,
                # which may decline honestly -- but never to prose.
                from service.intent import Classification

                classification = Classification(Intent.ACTION, understanding.reason, matched="action-request", route=route)
            self._answer_by_executing(text, scope, classification, action_request=understanding.is_action_request)
            return
        # freshness needs sources, not the local model's memory: "Was ist
        # heute passiert?" is a question, but not one memory can answer
        if self._FRESHNESS.search(text):
            self._answer_by_research(text, scope)
            return
        # "Ich brauche Wikipedia." carries no action verb, but it names an
        # openable thing with an intent cue: the semantic executor decides,
        # never a routing table
        if self._openable_wish(text):
            if self._needs_confirmation(None, text, scope, side_effect=True):
                return
            from service.intent import Classification

            wish = Classification(Intent.ACTION, "names an openable thing with an intent cue", matched="openable-wish", route=route)
            self._answer_by_executing(text, scope, wish, action_request=True)
            return
        self._answer_conversationally(text, scope)

    _FRESHNESS = re.compile(
        r"\b(was\s+ist\s+(?:heute|gerade|aktuell)\s+(?:in\s+der\s+welt\s+)?(?:passiert|los)"
        r"|was\s+(?:heute|gerade)\s+in\s+der\s+welt\s+passiert"
        r"|nachrichten\s+von\s+heute|neuigkeiten\s+von\s+heute"
        r"|what\s+happened\s+today|today'?s\s+news)\b", re.I)

    _WISH_CUE = re.compile(
        r"\b(ich\s+brauche|ich\s+will|ich\s+moechte|ich\s+möchte|bring\s+mich|geh\s+auf|geh\s+zu|ab\s+zu"
        r"|ich\s+muss\s+(?:auf|zu|in)|i\s+need|take\s+me\s+to|go\s+to)\b", re.I)

    def _openable_wish(self, text: str) -> bool:
        """A short non-question with an intent cue that names something openable."""

        from service.aliases import fold
        from service.intents import is_question
        from service.websearch import known_site

        if len(text.split()) > 9 or is_question(text) or not self._WISH_CUE.search(text):
            return False
        probe = fold(text).replace("-", " ")
        try:
            if self.aliases.matches(text):
                return True
        except Exception:  # noqa: BLE001
            pass
        for token in probe.split():
            if len(token) >= 4 and known_site(token):
                return True
        launcher = getattr(self, "_apps", None)
        if launcher is not None and getattr(launcher, "_index", None):
            for folded in launcher._index:
                if len(folded) >= 4 and folded in probe:
                    return True
        return False

    # -- typed operations ------------------------------------------------

    def _last_user_meta(self) -> dict[str, Any]:
        with self._lock:
            for turn in reversed(self._history):
                if turn.role == "user":
                    return dict(turn.meta or {})
        return {}

    def _needs_confirmation(self, action: Any, text: str, scope: str, *, side_effect: bool = False) -> bool:
        """Confidence + consequence: uncertain speech never executes; irreversible actions ask once.

        Returns True when a question was asked (and the intent parked) instead of executing.
        """

        from service.intents import Consequence

        meta = self._last_user_meta()
        level = str(meta.get("speech_level") or "")
        spoken = meta.get("source") in {"microphone", "ui_mic"}
        de = self.language.startswith("de")
        irreversible = action is not None and action.consequence is Consequence.IRREVERSIBLE
        if irreversible:
            self._pending = {"action": action, "text": text, "kind": "irreversible"}
            target = action.target or (action.arguments or {}).get("title") or ""
            self._deliver((f"Soll ich „{target}“ wirklich löschen? Das lässt sich nicht rückgängig machen – ja oder nein?" if de
                           else f"Do you really want me to delete “{target}”? This cannot be undone – yes or no?"),
                          scope=scope, backend="policy", final_state=JarvisState.WAITING,
                          context_text="[confirmation requested for an irreversible action]")
            return True
        if spoken and level == "low" and (side_effect or action is not None):
            self._pending = {"action": action, "text": text, "kind": "low_confidence"}
            heard = str(meta.get("normalized") or text)
            self._deliver((f"Ich bin nicht sicher, ob ich dich richtig verstanden habe: „{heard}“ – soll ich das machen?" if de
                           else f"I am not sure I understood you correctly: “{heard}” – should I do that?"),
                          scope=scope, backend="policy", final_state=JarvisState.WAITING,
                          context_text="[confirmation requested: low speech confidence]")
            return True
        return False

    _YES = re.compile(r"^\s*(ja|jawohl|genau|mach|mach das|mach es|bitte|ok|okay|yes|yep|do it|go ahead|sure)\b[.!\s]*$", re.I)
    _NO = re.compile(r"^\s*(nein|nee|nö|nicht|lass|lass es|abbrechen|no|nope|cancel|stop)\b[.!\s]*$", re.I)

    def _handle_pending_confirmation(self, text: str, scope: str) -> bool:
        pending = getattr(self, "_pending", None)
        if not pending:
            return False
        self._pending = None
        de = self.language.startswith("de")
        if self._YES.match(text):
            action, original = pending.get("action"), str(pending.get("text", ""))
            if action is not None and action.operation == "project.delete" and self.security.configured:
                # Confirmed by voice/text -- but a permanent deletion is a
                # protected change once a password exists.  The dialog is the
                # interface's; the password never passes through a model.
                target = action.target or ""
                self.emit(EventType.NOTIFICATION, {"kind": "needs_auth", "scope": "PROJECT_DELETE",
                                                   "text": f"Löschen von „{target}“ wartet auf deine Passwort-Freigabe.",
                                                   "retry": {"operation": "project.delete", "target": target, "request": original}})
                self._deliver("Das ist eine geschützte Änderung – bitte gib dein Passwort im Dialog ein, dann lösche ich es.",
                              scope=scope, backend="policy", final_state=JarvisState.WAITING,
                              context_text="[protected deletion: awaiting the owner's password]")
                return True
            if action is not None and str(action.operation).startswith("project."):
                self._answer_by_project_operation(action, original, scope, confirmed=True)
            elif action is not None and action.operation in {"system.open_view", "system.stop"}:
                self._answer_by_system_control(action, original, scope)
            else:
                # a spoken side-effect request the router handles: run the original words again, confirmed
                self.send_message(original, scope=scope, meta={"source": "correction_rerun", "confirmed": True})
            return True
        if self._NO.match(text):
            self._deliver("Okay, nichts gemacht." if de else "Okay, nothing done.", scope=scope, backend="policy",
                          context_text="[confirmation declined; nothing executed]")
            return True
        # anything else: the owner moved on; the parked intent is dropped, the new text is handled normally
        return False

    def _ask_clarification(self, action: Any, question: str, text: str, scope: str) -> None:
        """One concise question for genuinely missing information; the intent waits for the answer."""

        self._pending_clarification = {"action": action, "text": text}
        self.emit(EventType.TOOL, {"summary": f"clarification needed: {', '.join(action.missing)}", "action": action.to_dict(), "source": "intents"}, scope=scope)
        self._deliver(question, scope=scope, backend="intents", final_state=JarvisState.WAITING,
                      context_text=f"[clarification asked for {action.operation}: {', '.join(action.missing)}]")

    def _resume_clarification(self, text: str, scope: str) -> bool:
        """The owner answered the question: fill the missing slot and execute."""

        pending = getattr(self, "_pending_clarification", None)
        if not pending:
            return False
        self._pending_clarification = None
        action = pending["action"]
        from service.intents import _clean_title, is_action_request

        if is_action_request(text) or len(text.split()) > 8:
            return False  # a new request, not an answer
        value = _clean_title(text)
        if not value:
            return False
        if "title" in action.missing:
            action.arguments["title"] = value
            action.arguments.setdefault("goal", value)
            action.target = value
        elif "target" in action.missing:
            action.target = value
        elif "tasks" in action.missing:
            from service.intents import _split_list

            action.arguments["tasks"] = _split_list(text)
        action.missing = []
        self._answer_by_project_operation(action, f"{pending['text']} → {text}", scope)
        return True

    def _answer_by_project_operation(self, action: Any, text: str, scope: str, *, confirmed: bool = False) -> None:
        """Execute a typed project operation deterministically and verify the goal independently."""

        from service.project_ops import ProjectOperations, compose_concise

        de = self.language.startswith("de")
        if action.operation == "project.list":
            rows = [p for p in self.list_projects() if p.get("origin") == "owner" and not p.get("hidden") and p.get("importance") not in {"ARCHIVED", "TEST"}]
            self.emit(EventType.NOTIFICATION, {"kind": "open_view", "view": "projects", "params": {}, "text": ""}, scope=scope)
            names = ", ".join(f"„{p['title'] or p['goal'][:30]}“" for p in rows[:8])
            more = f" und {len(rows) - 8} weitere" if len(rows) > 8 else ""
            answer = ((f"Du hast {len(rows)} Projekte: {names}{more}. Die Projektansicht ist offen." if rows else "Du hast noch keine Projekte. Die Projektansicht ist offen.") if de
                      else (f"You have {len(rows)} projects: {names}{more}. The Projects view is open." if rows else "You have no projects yet. The Projects view is open."))
            self._deliver(answer, scope=scope, backend="projects", context_text=f"[listed {len(rows)} owner projects]")
            return
        if action.operation == "project.open":
            ops = ProjectOperations(self)
            project = ops._find(action.target)
            if project is None:
                self._deliver((f"Ich finde kein Projekt namens „{action.target}“." if action.target != "__last__" else "Ich weiß nicht, welches Projekt du meinst – nenn mir den Namen.") if de
                              else (f"I cannot find a project called “{action.target}”." if action.target != "__last__" else "I do not know which project you mean – give me its name."),
                              scope=scope, backend="projects", final_state=JarvisState.WAITING, context_text="[project.open: target not found]")
                return
            self._last_project_id = project.id
            self.emit(EventType.NOTIFICATION, {"kind": "open_view", "view": "projects", "params": {"id": project.id}, "text": ""}, scope=scope)
            self._deliver((f"Projekt „{project.title}“ ist offen." if de else f"Project “{project.title}” is open."), scope=scope, backend="projects",
                          context_text=f"[opened project {project.id}]")
            return

        self.state.set(JarvisState.WORKING, detail=action.operation, scope=scope)
        self.emit(EventType.TOOL, {"summary": f"executing {action.operation}", "action": action.to_dict(), "source": "project_ops"}, scope=scope)
        receipt = ProjectOperations(self).execute(action, request=text)
        self.state.set(JarvisState.VERIFYING, detail=receipt.kind, scope=scope)
        self.receipts.record(receipt)
        self._session_receipts.append(receipt)
        self.emit(EventType.TOOL, {"summary": receipt.summary(), "receipt_id": receipt.id, "receipt": receipt.to_dict()}, scope=scope)
        # GOAL_SATISFIED is the contract, not the return code: every success
        # criterion of the intent has a verification behind it.
        satisfied = receipt.verified
        reasons = [v.check for v in receipt.failures] or (["every success criterion verified"] if satisfied else [receipt.detail])
        self.emit(EventType.TOOL, {"summary": f"goal: {'SATISFIED' if satisfied else 'NOT satisfied'} — {action.operation} {action.target or ''}".strip(),
                                   "goal": {"ACTION_EXECUTED": receipt.ok, "EXECUTION_VERIFIED": receipt.verified, "GOAL_SATISFIED": satisfied, "reasons": reasons},
                                   "receipt_id": receipt.id, "source": "project_ops"}, scope=scope)
        if satisfied:
            try:
                self.think("mission_finished")
            except Exception:  # noqa: BLE001
                pass
        self._deliver(compose_concise(receipt, language=self.language), scope=scope, backend=receipt.executor,
                      final_state=JarvisState.IDLE if satisfied else JarvisState.ERROR,
                      context_text=f"[executed {receipt.kind}: {'verified' if receipt.verified else 'not verified'}, receipt {receipt.id}]")

    def _answer_by_system_control(self, action: Any, text: str, scope: str) -> None:
        de = self.language.startswith("de")
        if action.operation == "system.stop":
            self.stop_current(reason="owner")
            self.state.set(JarvisState.IDLE)
            return
        if action.operation == "system.tell_time":
            now = datetime.now()
            answer = (f"Es ist {now.strftime('%H:%M')} Uhr." if de else f"It is {now.strftime('%H:%M')}.")
            self._deliver(answer, scope=scope, backend="clock", context_text=f"[told the time: {now.strftime('%H:%M:%S')}]")
            return
        if action.operation == "system.tell_date":
            now = datetime.now()
            days = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
            months = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]
            answer = (f"Heute ist {days[now.weekday()]}, der {now.day}. {months[now.month - 1]} {now.year}." if de
                      else f"Today is {now.strftime('%A, %B %d, %Y')}.")
            self._deliver(answer, scope=scope, backend="clock", context_text=f"[told the date: {now.date().isoformat()}]")
            return
        if action.operation == "app.open":
            # an owner-taught alias outranks the app index: "Uni-Planer" may be
            # a file, a URL or a project even when the sentence said "öffne"
            try:
                alias = self.aliases.get(action.target)
            except Exception:  # noqa: BLE001
                alias = None
            if alias and alias.get("kind") != "app":
                self._open_alias(alias, text, scope)
                return
            if alias and alias.get("kind") == "app":
                action.target = str(alias.get("value") or action.target)
            self.state.set(JarvisState.WORKING, detail=f"öffne {action.target}", scope=scope)
            result = self.apps.launch(action.target)
            self.emit(EventType.TOOL, {"summary": f"app.open {action.target}: {'ok' if result.get('ok') else result.get('error', '')}",
                                       "result": result, "source": "apps"}, scope=scope)
            if not result.get("ok"):
                # observe → replan, not dead-end: no such app, but the name may
                # be a website everyone knows ("Öffne Wikipedia"), and if it is
                # neither, ONE question turns the answer into a lasting alias.
                from service.websearch import resolve_site

                url, how = resolve_site(action.target)
                if how in {"known", "url"}:
                    self.emit(EventType.TOOL, {"summary": f"app.open «{action.target}» -> web.open {url} (recovered)",
                                               "source": "semantic"}, scope=scope)
                    self._open_web_target(action.target, text, scope)
                    return
                hint = result.get("candidates") or []
                if not hint:
                    self._ask_open_target(action.target, text, scope)
                    return
                answer = ((f"Ich finde keine App namens „{action.target}“." + (f" Meintest du: {', '.join(hint[:3])}?" if hint else "")) if de
                          else f"I cannot find an app called “{action.target}”." + (f" Did you mean: {', '.join(hint[:3])}?" if hint else ""))
                self._deliver(answer, scope=scope, backend="apps", final_state=JarvisState.IDLE, context_text=f"[app.open failed: {result.get('error', '')}]")
                return
            app = result.get("app", action.target)
            if result.get("already_running"):
                answer = (f"{app} läuft bereits – ich habe es in den Vordergrund geholt." if de else f"{app} is already running – brought to the front.")
            elif result.get("process_verified"):
                answer = (f"{app} ist geöffnet ({result.get('process')})." if de else f"{app} is open ({result.get('process')}).")
            else:
                answer = (f"{app} wurde gestartet; einen passenden Prozess sehe ich noch nicht – manche Apps laufen unter anderem Namen." if de
                          else f"{app} was launched; I do not see a matching process yet – some apps run under a different name.")
            self._deliver(answer, scope=scope, backend="apps", context_text=f"[app.open {app}: verified={result.get('process_verified')}, {result.get('seconds')}s]")
            return
        if action.operation == "web.open":
            # the target may be a URL, a spoken site name ("Wikipedia") or an
            # owner alias; _open_web_target resolves canonically and never
            # leaves the owner on a dead end
            try:
                alias = self.aliases.get(action.target)
            except Exception:  # noqa: BLE001
                alias = None
            if alias and alias.get("kind") not in {None, "url"}:
                self._open_alias(alias, text, scope)
                return
            target = str(alias.get("value")) if alias else action.target
            self._open_web_target(target, text, scope)
            return
        if action.operation in {"file.open", "folder.open"}:
            try:
                alias = self.aliases.get(action.target)
            except Exception:  # noqa: BLE001
                alias = None
            if alias and alias.get("kind") in {"file", "folder"}:
                self._open_alias(alias, text, scope)
                return
            if alias and alias.get("kind") == "url":
                self._open_web_target(str(alias.get("value")), text, scope)
                return
            if self._open_path_target(action.target, scope=scope):
                return
            self._ask_open_target(action.target, text, scope)
            return
        if action.operation == "web.search":
            from service.websearch import search as web_search

            self.state.set(JarvisState.RESEARCHING, detail=f"suche: {action.target}", scope=scope)
            result = web_search(action.target)
            self.emit(EventType.TOOL, {"summary": f"web.search „{action.target}“: {len(result.get('results', []))} Treffer" if result.get("ok") else f"web.search fehlgeschlagen: {result.get('error')}",
                                       "result": result, "source": "websearch"}, scope=scope)
            if not result.get("ok"):
                answer = ((f"Die Internetsuche ist gerade nicht erreichbar: {result.get('error')}" if result.get("offline")
                           else f"Ich habe dazu nichts gefunden: {result.get('error')}") if de
                          else f"Web search failed: {result.get('error')}")
                self._deliver(answer, scope=scope, backend="websearch", final_state=JarvisState.IDLE,
                              context_text="[web.search failed; said so instead of guessing]")
                return
            rows = result.get("results", [])
            # the follow-up context: "fass mir den Inhalt zusammen" now knows
            # exactly which results "der Inhalt" refers to
            self._web_context = {"query": action.target, "results": rows, "at": time.time()}
            lines = [f"• {r['title']} — {r['snippet'][:140]}".rstrip(" —") for r in rows[:4]]
            sources = ", ".join(r["url"].split("/")[2] for r in rows[:4])
            answer = ((f"Dazu habe ich im Internet gefunden:\n" + "\n".join(lines) + f"\n\nQuellen: {sources}") if de
                      else ("Found on the web:\n" + "\n".join(lines) + f"\n\nSources: {sources}"))
            self._deliver(answer, scope=scope, backend="websearch",
                          context_text=f"[web.search {action.target}: {len(rows)} real results, sources {sources}]")
            return
        if action.operation == "system.open_view":
            self.emit(EventType.NOTIFICATION, {"kind": "open_view", "view": action.target, "params": {}, "text": ""}, scope=scope)
            names = {"projects": "Projekte", "missions": "Mission Control", "activity": "Activity", "knowledge": "Knowledge", "corrections": "Korrekturen",
                     "diagnostics": "Diagnose", "owner": "Einstellungen", "voice": "Voice Studio", "thoughts": "Gedanken", "capabilities": "Fähigkeiten", "release": "Release"}
            label = names.get(action.target, action.target)
            self._deliver((f"{label} ist offen." if de else f"{label} is open."), scope=scope, backend="ui", context_text=f"[opened view {action.target}]")
            return
        if action.operation == "image.generate":
            self._answer_image_generate(action.target or text, scope)
            return
        if action.operation in {"fs.count", "fs.largest", "fs.list", "fs.open", "fs.search", "fs.info"}:
            self._answer_fs_operation(action, text, scope)
            return
        if action.operation == "web.read_summary":
            self._answer_web_summary(action, text, scope)
            return
        if action.operation == "tv.control":
            self._answer_tv_control(action, scope)
            return
        if action.operation == "job.cancel":
            self._answer_job_cancel(action, scope)
            return
        if action.operation == "calendar.create":
            self._answer_calendar_create(text or action.target, scope)
            return
        if action.operation == "calendar.query":
            self._answer_calendar_query(text or action.target, scope)
            return
        self._deliver("Das kann ich nicht steuern." if de else "I cannot control that.", scope=scope, backend="ui")

    def _answer_job_cancel(self, action: Any, scope: str) -> None:
        """"Stop die Bilderzeugung": distinct from stopping speech (§55)."""

        de = self.language.startswith("de")
        wanted = action.target or "image"
        active = self.jobs.active()
        matching = [j for j in active if wanted == "all" or j.get("kind") == wanted]
        if not matching:
            self._deliver(("Da läuft gerade nichts, was ich abbrechen könnte." if de
                           else "Nothing matching is running."), scope=scope, backend="jobs")
            return
        if wanted == "all" and len(matching) > 1:
            names = ", ".join(j["title"] for j in matching[:4])
            self._pending = {"action": None, "text": "Stopp die Bilderzeugung"}
            self._deliver((f"Es laufen {len(matching)} Arbeiten ({names}). Wirklich alle abbrechen?" if de
                           else f"{len(matching)} jobs are running. Cancel all of them?"),
                          scope=scope, backend="jobs", final_state=JarvisState.WAITING)
            return
        cancelled, uncancellable = [], []
        for j in matching:
            if self.jobs.cancel(j["job_id"]):
                cancelled.append(j["title"])
            else:
                uncancellable.append(j)
        lines = []
        if cancelled:
            lines.append(("Abgebrochen: " if de else "Cancelled: ") + ", ".join(cancelled))
        for j in uncancellable:
            lines.append((f"„{j['title']}“ steckt mitten in „{j.get('phase', '')}“ und lässt sich nicht mehr sauber stoppen — es ist gleich fertig." if de
                          else f"“{j['title']}” is mid-{j.get('phase', '')} and will finish shortly."))
        self._deliver("\n".join(lines), scope=scope, backend="jobs",
                      context_text=f"[job cancel: {len(cancelled)} cancelled, {len(uncancellable)} running out]")

    # ------------------------------------------------------------------
    # The living-room TV
    # ------------------------------------------------------------------

    def _answer_tv_control(self, action: Any, scope: str) -> None:
        de = self.language.startswith("de")
        sub = action.target
        status = self.tv.status()
        if not status.get("paired") and sub != "power_on":
            self._deliver(("Es ist noch kein Fernseher gekoppelt. Unter Owner → Geräte kannst du deinen LG suchen und koppeln." if de
                           else "No TV is paired yet. Pair your LG under Owner → Devices."),
                          scope=scope, backend="tv", final_state=JarvisState.WAITING)
            return
        self.state.set(JarvisState.WORKING, detail=f"TV: {sub}", scope=scope)
        apps = {"youtube": "youtube.leanback.v4", "netflix": "netflix", "spotify": "spotify-beehive",
                "amazon": "amazon", "prime": "amazon", "disney": "com.disney.disneyplus-prod"}
        if sub == "show_zeus":
            # the TV needs the PC's LAN address — and the server must actually
            # be LAN-bound (opt-in via ZEUS_LAN=1; loopback is the default for
            # a reason).  No fake success against a dead URL.
            url = str(self.lifecycle.stages.get("http", {}).get("detail", "")) or "http://127.0.0.1:8420"
            if "127.0.0.1" in url and os.environ.get("ZEUS_LAN", "").strip() not in {"1", "true"}:
                self._deliver(("ZEUS ist gerade nur auf diesem PC erreichbar (127.0.0.1) — der Fernseher käme nicht dran. "
                               "Setze die Umgebungsvariable ZEUS_LAN=1 und starte ZEUS neu, dann funktioniert die TV-Anzeige im Heimnetz." if de
                               else "ZEUS is bound to this PC only (127.0.0.1). Set ZEUS_LAN=1 and restart to enable the LAN TV display."),
                              scope=scope, backend="tv", final_state=JarvisState.WAITING,
                              context_text="[tv show_zeus refused: server is loopback-only]")
                return
            try:
                import socket as _s

                probe = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
                probe.connect(("8.8.8.8", 80))
                lan_ip = probe.getsockname()[0]
                probe.close()
                url = url.replace("127.0.0.1", lan_ip).replace("localhost", lan_ip)
            except OSError:
                pass
            result = self.tv.show_zeus(url)
        elif sub == "power_on":
            result = self.tv.power_on()
        elif sub == "power_off":
            result = self.tv.power_off()
        elif sub in {"volume_up", "volume_down"}:
            result = self.tv.volume_step(sub == "volume_up")
        elif sub == "mute":
            result = self.tv.mute(True)
        elif sub == "open_app":
            wanted = str(action.arguments.get("app", "")).strip()
            app_id = apps.get(wanted.lower())
            result = self.tv.launch_app(app_id) if app_id else self.tv.open_url("https://" + wanted.lower().replace(" ", "") + ".com")
        else:
            result = {"ok": False, "error": f"unbekannt: {sub}"}
        self.emit(EventType.TOOL, {"summary": f"tv.{sub}: {'ok' if result.get('ok') else result.get('error', '')[:100]}",
                                   "result": {k: v for k, v in result.items() if k != "payload"}, "source": "tv"}, scope=scope)
        if result.get("ok"):
            words = {"show_zeus": "ZEUS ist auf dem Fernseher.", "power_off": "Fernseher ist aus.",
                     "power_on": "Einschalt-Signal gesendet — ob er angeht, hängt vom Netzwerk-Standby des TVs ab.",
                     "volume_up": "Lauter.", "volume_down": "Leiser.", "mute": "Stumm.",
                     "open_app": "Läuft auf dem Fernseher."}
            self._deliver(words.get(sub, "Erledigt."), scope=scope, backend="tv",
                          context_text=f"[tv {sub}: ok]")
        else:
            self._deliver((f"Der Fernseher hat nicht reagiert: {result.get('error', '')[:120]}" if de
                           else f"The TV did not respond: {result.get('error', '')[:120]}"),
                          scope=scope, backend="tv", final_state=JarvisState.ERROR)

    # ------------------------------------------------------------------
    # Web reading: the follow-up knows what "davon" means
    # ------------------------------------------------------------------

    def _answer_web_summary(self, action: Any, text: str, scope: str) -> None:
        """Fetch the referenced article FOR REAL and summarize it with a source."""

        de = self.language.startswith("de")
        m = re.search(r"https?://\S+", text)
        explicit_url = m.group(0).rstrip(".,)") if m else ""
        ctx = getattr(self, "_web_context", None)
        candidates: list[dict[str, Any]] = []
        if explicit_url:
            candidates = [{"url": explicit_url, "title": explicit_url}]
        elif ctx and ctx.get("results"):
            from service.aliases import fold

            probe = fold(text)
            rows = list(ctx["results"])
            named = [r for r in rows if fold(str(r.get("url", "")).split("/")[2] if "://" in str(r.get("url", "")) else "") and
                     fold(str(r.get("url", "")).split("/")[2].replace("www.", "").split(".")[0]) in probe]
            candidates = (named or rows)[:4]
        if not candidates:
            self._deliver(("Worauf beziehst du dich? Ich habe gerade keine Suchergebnisse oder Artikel offen — such erst etwas, oder gib mir einen Link." if de
                           else "What are you referring to? I have no open search results — search first, or give me a link."),
                          scope=scope, backend="web", final_state=JarvisState.WAITING,
                          context_text="[web summary asked without any web context]")
            return

        job = self.jobs.create(f"Artikel zusammenfassen: {candidates[0].get('title', '')[:50]}", kind="web", scope=scope)
        self._deliver(("Bin dran — ich lese den Artikel und fasse ihn zusammen." if de
                       else "On it — reading the article and summarizing."),
                      scope=scope, backend="web", final_state=JarvisState.RESEARCHING,
                      context_text=f"[web summary job {job.job_id}]")

        def work() -> None:
            from brain.tiers import ModelTier
            from service.webread import fetch_readable, summarize_with_retry

            article = None
            tried: list[str] = []
            for candidate in candidates[:3]:
                url = str(candidate.get("url", ""))
                self.jobs.phase(job.job_id, f"Lese {url.split('/')[2] if '://' in url else url}")
                self.state.set(JarvisState.RESEARCHING, detail=f"lese {url[:80]}", scope=scope)
                fetched = fetch_readable(url)
                self.emit(EventType.TOOL, {"summary": f"web.read {url}: {'ok, ' + str(len(fetched.get('text', ''))) + ' Zeichen' if fetched.get('ok') else fetched.get('error', '')[:120]}",
                                           "source": "webread"}, scope=scope)
                if fetched.get("ok"):
                    article = fetched
                    break
                tried.append(f"{url}: {fetched.get('error', '')[:80]}")
            if article is None:
                self.jobs.fail(job.job_id, "keine Quelle lesbar", detail="; ".join(tried)[:280])
                self._deliver(("Keine der Quellen ließ sich lesen (blockiert oder leer). Details stehen in Activity." if de
                               else "None of the sources could be read. Details are in Activity."),
                              scope=scope, backend="web", final_state=JarvisState.ERROR)
                return
            self.jobs.phase(job.job_id, "Fasse zusammen", progress=0.7)
            self.state.set(JarvisState.THINKING, detail="fasse den Artikel zusammen", scope=scope)
            try:
                provider = self.kernel.provider(ModelTier.FAST_LOCAL)
            except Exception as exc:  # noqa: BLE001
                self.jobs.fail(job.job_id, f"kein Modell: {exc}")
                self._deliver("Die lokale KI ist gerade nicht erreichbar — gleich nochmal versuchen.", scope=scope,
                              backend="web", final_state=JarvisState.ERROR)
                return
            summary = summarize_with_retry(provider, title=article.get("title", ""), text=article.get("text", ""))
            if not summary.get("ok"):
                self.jobs.fail(job.job_id, summary.get("error", "Zusammenfassung fehlgeschlagen"))
                self._deliver(("Die Zusammenfassung ist auch nach mehreren Anläufen fehlgeschlagen — die lokale KI antwortet nicht. Details in Activity." if de
                               else "Summarization failed after several attempts. Details in Activity."),
                              scope=scope, backend="web", final_state=JarvisState.ERROR)
                return
            self._web_context = {**(getattr(self, "_web_context", None) or {}),
                                 "last_article": {"url": article["url"], "title": article.get("title", "")}}
            self.jobs.complete(job.job_id, {"url": article["url"], "title": article.get("title", ""),
                                            "context": summary.get("context")})
            answer = summary["summary"] + f"\n\nQuelle: {article.get('title') or article['url']} — {article['url']}"
            self._deliver(answer, scope=scope, backend="webread",
                          context_text=f"[summarized {article['url']} ({summary.get('context')} context)]")

        threading.Thread(target=work, daemon=True, name=f"web-summary-{job.job_id[-6:]}").start()

    # ------------------------------------------------------------------
    # Filesystem intelligence: count/find/inspect are TOOL work, never prose
    # ------------------------------------------------------------------

    _FS_SEARCH_ROOTS = (r"D:\\", r"C:\\Users\\")

    def _fs_candidates(self, name: str, *, drive: str = "", limit: int = 6) -> list[str]:
        """Real directories whose name contains ``name`` — bounded, two levels."""

        from service.aliases import fold

        wanted = fold(name).replace("-", " ").strip()
        if not wanted:
            return []
        roots = [drive] if drive else [r.replace("\\\\", "\\") for r in self._FS_SEARCH_ROOTS]
        hits: list[str] = []
        for root in roots:
            try:
                level1 = [e for e in os.scandir(root) if e.is_dir()]
            except OSError:
                continue
            for entry in level1:
                if wanted in fold(entry.name).replace("-", " "):
                    hits.append(entry.path)
            if len(hits) >= limit:
                break
            for entry in level1[:80]:
                if len(hits) >= limit:
                    break
                try:
                    for sub in os.scandir(entry.path):
                        if sub.is_dir() and wanted in fold(sub.name).replace("-", " "):
                            hits.append(sub.path)
                            if len(hits) >= limit:
                                break
                except OSError:
                    continue
        # exact folds first, shortest paths first: D:\Jarvis beats deep matches
        hits.sort(key=lambda p: (fold(Path(p).name).replace("-", " ") != wanted, len(p)))
        return hits[:limit]

    def _fs_count(self, path: str, what: str) -> dict[str, Any]:
        p = Path(path)
        if not p.is_dir():
            return {"ok": False, "error": f"{path} ist kein Ordner"}
        dirs = files = 0
        try:
            for entry in os.scandir(p):
                if entry.is_dir():
                    dirs += 1
                else:
                    files += 1
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": str(p), "dirs": dirs, "files": files,
                "count": dirs if what == "dirs" else files}

    @property
    def system_context(self) -> Any:
        """Deterministic self-knowledge: repo/data/model roots ZEUS resolves by."""

        if getattr(self, "_system_context", None) is None:
            from service.system_context import SystemContext

            self._system_context = SystemContext(state_root=Path(self.kernel.state_root))
        return self._system_context

    def _answer_fs_operation(self, action: Any, text: str, scope: str) -> None:
        de = self.language.startswith("de")
        args = dict(action.arguments or {})
        op = action.operation
        what = args.get("what") or "dirs"
        path = str(args.get("path") or "")
        name = str(args.get("name") or "")
        drive = str(args.get("drive") or "")
        self_ref = bool(args.get("self_ref"))
        self.state.set(JarvisState.WORKING, detail=f"Dateisystem: {op}", scope=scope)

        # 1) a "dein Repo / deine Modelle" phrase resolves to a real path directly
        if not path and self_ref:
            key, resolved = self.system_context.resolve_self_reference(text)
            if resolved:
                path = resolved
                self.emit(EventType.TOOL, {"summary": f"self-reference {key} -> {resolved}", "source": "fs"}, scope=scope)

        # 2) an owner-taught alias wins for a named folder
        if not path and name:
            try:
                alias = self.aliases.get(name)
            except Exception:  # noqa: BLE001
                alias = None
            if alias and alias.get("kind") == "folder":
                path = str(alias.get("value"))

        # 3) resolve a bare name against the real filesystem (search + clarify)
        if not path and name:
            candidates = self._fs_candidates(name, drive=drive)
            self.emit(EventType.TOOL, {"summary": f"fs resolve „{name}“: {len(candidates)} Kandidat(en)",
                                       "candidates": candidates, "source": "fs"}, scope=scope)
            if not candidates:
                self._deliver((f"Ich finde keinen Ordner namens „{name}“" + (f" auf {drive}" if drive else "") + ". Wo liegt er?" if de
                               else f"I cannot find a folder called “{name}”. Where is it?"),
                              scope=scope, backend="fs", final_state=JarvisState.WAITING)
                self._pending_open = {"name": name, "text": text}
                return
            if len(candidates) > 1:
                self._pending_fs = {"action": action, "candidates": candidates, "text": text}
                listing = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates[:5]))
                self._deliver((f"Ich finde {len(candidates)} passende Ordner:\n{listing}\nWelchen meinst du?" if de
                               else f"I find {len(candidates)} matching folders:\n{listing}\nWhich one?"),
                              scope=scope, backend="fs", final_state=JarvisState.WAITING,
                              context_text=f"[fs disambiguation among {len(candidates)}]")
                return
            path = candidates[0]

        # 4) a drive root with no name ("größter Ordner auf D:")
        if not path and drive:
            path = drive
        if not path:
            self._deliver(("Welchen Ordner meinst du? Nenn mir einen Pfad, einen Namen oder „dein Repo“." if de
                           else "Which folder? Give me a path, a name, or say “your repo”."),
                          scope=scope, backend="fs", final_state=JarvisState.WAITING)
            return

        if op == "fs.open":
            if self._open_path_target(path, name=name or Path(path).name, scope=scope):
                return
            self._deliver((f"{path} existiert nicht (mehr)." if de else f"{path} does not exist."),
                          scope=scope, backend="fs", final_state=JarvisState.WAITING)
            return
        if op == "fs.largest":
            self._answer_fs_largest(path, what, scope)
            return
        if op == "fs.list":
            self._answer_fs_list(path, scope)
            return

        # fs.count
        result = self._fs_count(path, what)
        self.emit(EventType.TOOL, {"summary": f"fs.count {path}: {result}", "source": "fs"}, scope=scope)
        if not result.get("ok"):
            self._deliver((f"Das konnte ich nicht zählen: {result.get('error')}" if de
                           else f"I could not count that: {result.get('error')}"),
                          scope=scope, backend="fs", final_state=JarvisState.ERROR)
            return
        label = ("Unterordner" if what == "dirs" else "Dateien") if de else ("subfolders" if what == "dirs" else "files")
        other = (f" (und {result['files']} Dateien)" if what == "dirs" else f" (und {result['dirs']} Ordner)") if de else ""
        self._deliver((f"„{result['path']}“ hat {result['count']} direkte {label}{other}." if de
                       else f"“{result['path']}” has {result['count']} direct {label}."),
                      scope=scope, backend="fs",
                      context_text=f"[fs.count {result['path']}: {result['dirs']} dirs, {result['files']} files]")

    def _answer_fs_list(self, path: str, scope: str) -> None:
        de = self.language.startswith("de")
        try:
            entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError as exc:
            self._deliver((f"Ich kann „{path}“ nicht lesen: {exc}" if de else f"I cannot read “{path}”: {exc}"),
                          scope=scope, backend="fs", final_state=JarvisState.ERROR)
            return
        dirs = [e.name for e in entries if e.is_dir()]
        files = [e.name for e in entries if e.is_file()]
        lines = [(f"Direkt in „{path}“:" if de else f"Directly in “{path}”:")]
        if dirs:
            lines.append(("Ordner: " if de else "Folders: ") + ", ".join(dirs[:20]) + (" …" if len(dirs) > 20 else ""))
        if files:
            lines.append(("Dateien: " if de else "Files: ") + ", ".join(files[:20]) + (" …" if len(files) > 20 else ""))
        if not dirs and not files:
            lines.append("(leer)" if de else "(empty)")
        self.emit(EventType.NOTIFICATION, {"kind": "open_view", "view": "files", "params": {"path": path}, "text": ""}, scope=scope)
        self._deliver("\n".join(lines), scope=scope, backend="fs",
                      context_text=f"[fs.list {path}: {len(dirs)} dirs, {len(files)} files]")

    def _answer_fs_largest(self, path: str, what: str, scope: str) -> None:
        """The largest direct child of a folder — recursive sizing runs as a job."""

        de = self.language.startswith("de")
        try:
            children = [e for e in os.scandir(path) if (e.is_dir() if what == "dirs" else e.is_file())]
        except OSError as exc:
            self._deliver((f"Ich kann „{path}“ nicht lesen: {exc}" if de else f"I cannot read “{path}”: {exc}"),
                          scope=scope, backend="fs", final_state=JarvisState.ERROR)
            return
        if not children:
            self._deliver((f"In „{path}“ liegen keine {'Ordner' if what == 'dirs' else 'Dateien'}." if de
                           else f"“{path}” has no {'folders' if what == 'dirs' else 'files'}."),
                          scope=scope, backend="fs")
            return
        job = self.jobs.create(f"Größe berechnen: {path}", kind="index", scope=scope)
        self._deliver((f"Bin dran — ich rechne die Größe der {len(children)} Einträge in „{path}“ aus. Bei vielen Dateien dauert das einen Moment." if de
                       else f"On it — measuring the {len(children)} entries in “{path}”. This can take a moment."),
                      scope=scope, backend="fs", final_state=JarvisState.WORKING)

        def work() -> None:
            def dir_size(root: str) -> int:
                total = 0
                stack = [root]
                seen = 0
                while stack:
                    if self.jobs.cancelled(job.job_id):
                        return total
                    cur = stack.pop()
                    try:
                        with os.scandir(cur) as it:
                            for e in it:
                                try:
                                    if e.is_symlink():
                                        continue
                                    if e.is_dir(follow_symlinks=False):
                                        stack.append(e.path)
                                    else:
                                        total += e.stat(follow_symlinks=False).st_size
                                except OSError:
                                    continue
                    except OSError:
                        continue
                    seen += 1
                    if seen % 200 == 0:
                        self.jobs.phase(job.job_id, f"gemessen: {seen} Ordner", progress=None)
                return total
            sizes = []
            for i, e in enumerate(children):
                if self.jobs.cancelled(job.job_id):
                    self.jobs.fail(job.job_id, "abgebrochen")
                    return
                self.jobs.phase(job.job_id, f"{e.name}", progress=(i + 1) / max(1, len(children)))
                try:
                    size = dir_size(e.path) if what == "dirs" else e.stat().st_size
                except OSError:
                    size = 0
                sizes.append((e.name, e.path, size))
            sizes.sort(key=lambda t: t[2], reverse=True)
            self.jobs.complete(job.job_id, {"top": [{"name": n, "bytes": s} for n, _p, s in sizes[:5]]})
            top = sizes[0]
            gib = top[2] / (1 << 30)
            unit = f"{gib:.1f} GB" if gib >= 1 else f"{top[2] / (1 << 20):.0f} MB"
            rest = ", ".join(f"{n} ({(s / (1 << 30)):.1f} GB)" if s >= (1 << 30) else f"{n} ({(s / (1 << 20)):.0f} MB)"
                             for n, _p, s in sizes[1:4])
            kind = ("Ordner" if what == "dirs" else "Datei") if de else ("folder" if what == "dirs" else "file")
            self.emit(EventType.TOOL, {"summary": f"fs.largest {path}: {top[0]} = {unit}", "source": "fs"}, scope=scope)
            self._deliver((f"Der größte direkte {kind} in „{path}“ ist „{top[0]}“ mit {unit}." + (f"\nDahinter: {rest}." if rest else "") if de
                           else f"The largest {kind} in “{path}” is “{top[0]}” at {unit}." + (f"\nNext: {rest}." if rest else "")),
                          scope=scope, backend="fs",
                          context_text=f"[fs.largest {path}: {top[0]} {unit}]")

        threading.Thread(target=work, daemon=True, name=f"fs-largest-{job.job_id[-6:]}").start()

    def _resume_fs(self, text: str, scope: str) -> bool:
        """The owner picked one of the offered folders (by name, path or number)."""

        pending = getattr(self, "_pending_fs", None)
        if not pending:
            return False
        self._pending_fs = None
        from service.aliases import fold

        reply = fold(text).strip(" .!?")
        candidates = pending.get("candidates") or []
        chosen = ""
        m = re.search(r"\b([1-9])\b", reply)
        if m and int(m.group(1)) <= len(candidates):
            chosen = candidates[int(m.group(1)) - 1]
        if not chosen:
            for c in candidates:
                if fold(c) == reply or fold(Path(c).name) in reply or reply in fold(c):
                    chosen = c
                    break
        if not chosen:
            pm = re.search(r"[A-Za-z]:[\\/][^\s\"']*", text)
            if pm:
                chosen = pm.group(0)
        if not chosen:
            return False  # not an answer to the question; handle normally
        action = pending["action"]
        action.arguments = dict(action.arguments or {})
        action.arguments["path"] = chosen
        self._answer_fs_operation(action, pending.get("text", text), scope)
        return True

    # ------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------

    def _answer_calendar_create(self, text: str, scope: str) -> None:
        """Enter a real event; a missing piece becomes ONE question, not a guess."""

        from service.calendar import parse_event

        de = self.language.startswith("de")
        proposal = parse_event(text)
        self.emit(EventType.TOOL, {"summary": f"calendar parse: {proposal['title'] or '?'} @ {proposal['start'] or '?'}"
                                              + (f" (missing: {', '.join(proposal['missing'])})" if proposal["missing"] else ""),
                                   "proposal": proposal, "source": "calendar"}, scope=scope)
        if proposal["missing"]:
            need = proposal["missing"]
            if "date" in need or "time" in need:
                question = ("Wann soll der Termin sein — Tag und Uhrzeit?" if de else "When should the event be — day and time?")
            else:
                question = ("Wie soll der Termin heißen?" if de else "What should the event be called?")
            self._pending_calendar = {"text": text}
            self._deliver(question, scope=scope, backend="calendar", final_state=JarvisState.WAITING,
                          context_text=f"[calendar: asked for {', '.join(need)}]")
            return
        event = self.calendar.create(title=proposal["title"], start=proposal["start"], end=proposal["end"],
                                     timezone=proposal["timezone"], source="chat")
        stored = self.calendar.get(event["id"])  # verified: read back from disk
        satisfied = stored is not None and stored["title"] == proposal["title"]
        self.emit(EventType.TOOL, {"summary": f"goal: {'SATISFIED' if satisfied else 'NOT satisfied'} — calendar.create {event['id']}",
                                   "event": event, "source": "calendar"}, scope=scope)
        self.emit(EventType.NOTIFICATION, {"kind": "calendar", "event": event, "text": ""}, scope=scope)
        start_dt = datetime.fromisoformat(event["start"])
        minutes = int((datetime.fromisoformat(event["end"]) - start_dt).total_seconds() // 60)
        when = start_dt.strftime("%d.%m.%Y um %H:%M")
        self._deliver((f"Eingetragen: „{event['title']}“ am {when} ({minutes} Min.)." if de
                       else f"Entered: “{event['title']}” on {when} ({minutes} min)."),
                      scope=scope, backend="calendar",
                      final_state=JarvisState.IDLE if satisfied else JarvisState.ERROR,
                      context_text=f"[calendar event {event['id']} persisted]")

    def _answer_calendar_query(self, text: str, scope: str) -> None:
        from datetime import timedelta

        de = self.language.startswith("de")
        now = datetime.now().astimezone()
        lowered = text.lower()
        if "heute" in lowered or "today" in lowered:
            lo, hi, label = now.replace(hour=0, minute=0, second=0, microsecond=0), now.replace(hour=23, minute=59, second=59), ("heute" if de else "today")
        elif "morgen" in lowered or "tomorrow" in lowered:
            d = now + timedelta(days=1)
            lo, hi, label = d.replace(hour=0, minute=0, second=0, microsecond=0), d.replace(hour=23, minute=59, second=59), ("morgen" if de else "tomorrow")
        else:
            lo, hi, label = now, now + timedelta(days=7), ("in den nächsten 7 Tagen" if de else "in the next 7 days")
        events = self.calendar.list(start=lo.isoformat(), end=hi.isoformat())
        if not events:
            self._deliver((f"Keine Termine {label}." if de else f"No events {label}."), scope=scope, backend="calendar",
                          context_text=f"[calendar query: 0 events {label}]")
            return
        lines = [(f"Deine Termine {label}:" if de else f"Your events {label}:")]
        for e in events[:8]:
            start = datetime.fromisoformat(e["start"])
            lines.append(f"• {start.strftime('%a %d.%m. %H:%M')} — {e['title']}" + (f" ({e['location']})" if e.get("location") else ""))
        self.emit(EventType.NOTIFICATION, {"kind": "open_view", "view": "calendar", "params": {}, "text": ""}, scope=scope)
        self._deliver("\n".join(lines), scope=scope, backend="calendar",
                      context_text=f"[calendar query: {len(events)} events {label}]")

    def _resume_calendar(self, text: str, scope: str) -> bool:
        """The owner answered the calendar question: combine and enter."""

        pending = getattr(self, "_pending_calendar", None)
        if not pending:
            return False
        self._pending_calendar = None
        from service.calendar import parse_event
        from service.intents import is_action_request

        if is_action_request(text) and "uhr" not in text.lower():
            return False  # a new request, not an answer
        combined = f"{pending['text']} {text}"
        if parse_event(combined)["missing"]:
            return False  # still not enough: handle the message normally
        self._answer_calendar_create(combined, scope)
        return True

    # ------------------------------------------------------------------
    # The semantic control plane
    # ------------------------------------------------------------------

    def _semantic_goal(self, text: str, scope: str, *, guidance: str = "") -> Any:
        """One structured FAST_LOCAL call: the goal behind the words, or None.

        The context the model sees is deterministic and small: installed apps,
        project titles and owner aliases that lexically touch the request.
        The model chooses from a CLOSED operation set (see service.semantic);
        it can be wrong about the goal but it cannot invent a tool.
        """

        from brain.tiers import ModelTier
        from service.aliases import fold

        self.state.set(JarvisState.THINKING, detail="verstehe die Absicht", scope=scope)
        probe = fold(text).replace("-", " ")
        words = {w for w in probe.split() if len(w) >= 4}
        apps_hint: list[str] = []
        try:
            for display in self.apps.names():
                folded = fold(display)
                if any(w in folded or folded in probe for w in words):
                    apps_hint.append(display)
                if len(apps_hint) >= 8:
                    break
        except Exception:  # noqa: BLE001
            apps_hint = []
        projects_hint: list[str] = []
        try:
            for p in self.list_projects():
                title = str(p.get("title") or "")
                folded = fold(title)
                if folded and any(w in folded or folded in probe for w in words):
                    projects_hint.append(title)
                if len(projects_hint) >= 8:
                    break
        except Exception:  # noqa: BLE001
            projects_hint = []
        try:
            aliases_hint = self.aliases.matches(text)
        except Exception:  # noqa: BLE001
            aliases_hint = []
        provider = self.kernel.provider(ModelTier.FAST_LOCAL)
        goal = self.semantic.plan(text, provider, apps=apps_hint, projects=projects_hint,
                                  aliases=aliases_hint, guidance=guidance)
        if goal is not None:
            self.emit(EventType.TOOL,
                      {"summary": f"semantic goal: {goal.operation} „{goal.target}“ ({goal.confidence:.2f}, {goal.elapsed_ms:.0f}ms)",
                       "goal": goal.to_dict(), "source": "semantic", "text": text[:160]}, scope=scope)
        return goal

    def _dispatch_semantic_goal(self, goal: Any, text: str, scope: str, classification: Any) -> bool:
        """Route one semantic goal to the typed dispatcher that owns it."""

        from service.intents import ActionIntent

        op = goal.operation
        if op in {"delegate", "conversation"}:
            return False
        # low confidence must not silently fall back to lexical guessing: for
        # capability.missing the uncertainty IS the finding ("I have no tool
        # for this") — offer to build it instead of dropping to the legacy
        # planner, which is where the Spotify-for-a-light-switch answers live
        if goal.confidence < 0.5 and op not in {"clarify", "capability.missing"}:
            return False
        de = self.language.startswith("de")
        if op == "clarify":
            question = goal.question or ("Was genau soll ich tun?" if de else "What exactly should I do?")
            self._deliver(question, scope=scope, backend="semantic", final_state=JarvisState.WAITING,
                          context_text=f"[semantic clarification: {goal.reason[:120]}]")
            return True
        # a target the model INVENTED (not in the owner's words, not an alias,
        # not resolvable) must never be acted on: asking about the invention
        # ("Wo finde ich 'notizen'?") is worse than asking about the request
        if op in {"web.open", "app.open", "file.open", "folder.open", "project.open"} and goal.target:
            if not self._target_grounded(op, goal.target, text):
                self.emit(EventType.TOOL, {"summary": f"semantic target „{goal.target}“ is not grounded in the request; asking",
                                           "source": "semantic"}, scope=scope)
                what = {"web.open": "Welche Seite", "app.open": "Welches Programm", "file.open": "Welche Datei",
                        "folder.open": "Welchen Ordner", "project.open": "Welches Projekt"}[op]
                self._deliver((f"{what} genau soll ich öffnen?" if de else "What exactly should I open?"),
                              scope=scope, backend="semantic", final_state=JarvisState.WAITING,
                              context_text=f"[ungrounded semantic target {goal.target!r} for {op}]")
                return True
        if op in {"fs.count", "fs.largest", "fs.list", "fs.open"}:
            # prefer the deterministic parser's rich arguments; fall back to the
            # planner's target, resolved the same way (path / name / self-ref)
            from service.intents import parse_fs_operation

            parsed = parse_fs_operation(text)
            if parsed is not None and parsed.operation == op:
                self._answer_fs_operation(parsed, text, scope)
                return True
            tgt = goal.target or ""
            self_ref = bool(re.search(r"\b(dein|deine|your|own)\b", tgt, re.I)) or bool(re.search(r"\b(dein|deine|your|own)\b", text, re.I))
            is_path = bool(re.match(r"^[A-Za-z]:[\\/]", tgt))
            args = {"what": "files" if re.search(r"\bdatei|file\b", text, re.I) else "dirs",
                    "path": tgt if is_path else "", "name": "" if (is_path or self_ref) else tgt,
                    "drive": (tgt if re.match(r"^[A-Za-z]:\\?$", tgt) else ""), "self_ref": self_ref}
            intent = ActionIntent(op, verb="read", object_type="fs", target=tgt,
                                  arguments=args, confidence=goal.confidence, reason=f"semantic: {goal.reason[:120]}")
            self._answer_fs_operation(intent, text, scope)
            return True
        if op in {"web.open", "web.search", "app.open", "file.open", "folder.open",
                  "system.open_view", "system.tell_time", "system.tell_date",
                  "calendar.create", "calendar.query", "image.generate",
                  "fs.count", "web.read_summary"}:
            intent = ActionIntent(op, verb="open" if op.endswith(".open") else "read",
                                  object_type=op.split(".", 1)[0], target=goal.target,
                                  confidence=goal.confidence,
                                  success_criteria=["the goal is observably reached"],
                                  reason=f"semantic: {goal.reason[:120]}")
            self._answer_by_system_control(intent, text, scope)
            return True
        if op == "project.open":
            intent = ActionIntent("project.open", verb="open", object_type="project",
                                  target=goal.target or "__last__", confidence=goal.confidence,
                                  reason=f"semantic: {goal.reason[:120]}")
            self._answer_by_project_operation(intent, text, scope)
            return True
        if op == "music.control":
            self._answer_musically(text, scope, classification)
            return True
        if op == "knowledge.search":
            self._answer_from_records(text, scope, classification)
            return True
        if op == "research":
            self._answer_by_research(goal.target or text, scope)
            return True
        if op == "capability.missing":
            if goal.confidence >= 0.5:
                self._answer_by_acquisition(text, scope)
                return True
            # honest and forward-looking, never a dead end: name the gap and
            # offer the acquisition; a yes re-enters as an explicit "lerne"
            target = goal.target or text
            self._pending = {"action": None, "text": f"Lerne: {target}"}
            self._deliver((f"Dafür habe ich noch keine Fähigkeit ({target}). Soll ich versuchen, sie zu lernen?" if de
                           else f"I have no capability for that yet ({target}). Should I try to learn it?"),
                          scope=scope, backend="semantic", final_state=JarvisState.WAITING,
                          context_text=f"[missing capability offered for acquisition: {target[:120]}]")
            return True
        return False

    def _target_grounded(self, op: str, target: str, text: str) -> bool:
        """Whether a semantic target is anchored in reality rather than invented."""

        from service.aliases import fold

        folded_target, folded_text = fold(target).replace("-", " "), fold(text).replace("-", " ")
        if folded_target and folded_target in folded_text:
            return True
        try:
            if self.aliases.get(target):
                return True
        except Exception:  # noqa: BLE001
            pass
        if op == "web.open":
            from service.websearch import known_site

            return known_site(target)
        if op == "app.open":
            try:
                return bool(self.apps.resolve(target)[1])
            except Exception:  # noqa: BLE001
                return False
        if op == "project.open":
            try:
                titles = {str(p.get("title") or "").lower() for p in self.list_projects()}
            except Exception:  # noqa: BLE001
                titles = set()
            return target.lower() in titles
        if op in {"file.open", "folder.open"}:
            from pathlib import Path as _P

            try:
                return _P(target).exists()
            except OSError:
                return False
        return False

    # -- resolving "open X" when X is not what it first seemed -----------

    def _open_web_target(self, name: str, text: str, scope: str) -> None:
        """Open a site by NAME: canonical URL first, a results page as last resort."""

        from service.websearch import resolve_site

        de = self.language.startswith("de")
        url, how = resolve_site(name)
        if not url:
            self._ask_open_target(name, text, scope)
            return
        try:
            import webbrowser

            opened = webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001
            opened = False
            self.emit(EventType.TOOL, {"summary": f"web.open failed: {exc}", "source": "web"}, scope=scope)
        self.emit(EventType.TOOL, {"summary": f"web.open {url} ({how}): {opened}", "source": "web"}, scope=scope)
        if not opened:
            self._deliver((f"Ich konnte {url} nicht öffnen." if de else f"I could not open {url}."),
                          scope=scope, backend="web", final_state=JarvisState.ERROR,
                          context_text=f"[web.open {url}: failed]")
            return
        if how == "search":
            answer = (f"„{name}“ kenne ich nicht als Adresse – ich habe dir die Suchergebnisse dazu geöffnet." if de
                      else f"I do not know “{name}” as an address – I opened the search results for it.")
        else:
            answer = (f"{url} ist im Browser geöffnet." if de else f"{url} is open in the browser.")
        self._deliver(answer, scope=scope, backend="web", context_text=f"[web.open {url} ({how}): ok]")

    def _open_path_target(self, path: str, *, name: str = "", scope: str = "") -> bool:
        """Open a real file or folder in the OS shell; True only if it exists."""

        from pathlib import Path as _P

        de = self.language.startswith("de")
        p = _P(str(path))
        if not p.exists():
            return False
        try:
            os.startfile(str(p))  # noqa: S606 - deliberate: open in the associated app
        except OSError as exc:
            self._deliver((f"Öffnen von {p} ist fehlgeschlagen: {exc}" if de else f"Opening {p} failed: {exc}"),
                          scope=scope, backend="fs", final_state=JarvisState.ERROR,
                          context_text=f"[open {p}: {exc}]")
            return True
        kind = "Ordner" if p.is_dir() else "Datei"
        label = name or p.name
        self.emit(EventType.TOOL, {"summary": f"{'folder' if p.is_dir() else 'file'}.open {p}", "source": "fs"}, scope=scope)
        self._deliver((f"{kind} „{label}“ ist geöffnet ({p})." if de else f"Opened “{label}” ({p})."),
                      scope=scope, backend="fs", context_text=f"[opened {p}]")
        return True

    def _open_alias(self, alias: dict[str, Any], text: str, scope: str) -> None:
        """Open whatever an owner-taught alias points at."""

        kind, value, name = alias.get("kind"), str(alias.get("value") or ""), str(alias.get("name") or "")
        if kind == "url":
            self._open_web_target(value, text, scope)
            return
        if kind in {"file", "folder"}:
            if not self._open_path_target(value, name=name, scope=scope):
                de = self.language.startswith("de")
                self._deliver((f"„{name}“ zeigt auf {value}, aber das existiert nicht mehr. Wo liegt es jetzt?" if de
                               else f"“{name}” points at {value}, but it no longer exists. Where is it now?"),
                              scope=scope, backend="fs", final_state=JarvisState.WAITING)
                self._pending_open = {"name": name, "text": text}
            return
        if kind == "app":
            from service.intents import ActionIntent

            self._answer_by_system_control(ActionIntent("app.open", verb="open", object_type="app", target=value,
                                                        confidence=0.9, reason=f"alias {name}"), text, scope)
            return
        if kind == "project":
            from service.intents import ActionIntent

            self._answer_by_project_operation(ActionIntent("project.open", verb="open", object_type="project",
                                                           target=value, confidence=0.9, reason=f"alias {name}"), text, scope)
            return

    def _ask_open_target(self, name: str, text: str, scope: str) -> None:
        """No dead end: one question, and the answer is remembered as an alias."""

        de = self.language.startswith("de")
        self._pending_open = {"name": name, "text": text}
        self._deliver((f"Ich finde „{name}“ nicht eindeutig. Wo finde ich es – ein Pfad oder eine Webadresse?" if de
                       else f"I cannot resolve “{name}”. Where do I find it – a path or a web address?"),
                      scope=scope, backend="semantic", final_state=JarvisState.WAITING,
                      context_text=f"[asked where to find {name}; the answer becomes an alias]")

    def _resume_open_target(self, text: str, scope: str) -> bool:
        """The owner answered "where is X": open it and remember the alias."""

        pending = getattr(self, "_pending_open", None)
        if not pending:
            return False
        self._pending_open = None
        from service.aliases import classify_target

        kind, value = classify_target(text.strip())
        if not kind:
            return False  # not a location: handle the message normally
        name = str(pending.get("name") or "")
        de = self.language.startswith("de")
        if kind in {"file", "folder"}:
            from pathlib import Path as _P

            if not _P(value).exists():
                self._deliver((f"{value} existiert nicht – hast du dich vertippt?" if de
                               else f"{value} does not exist – a typo?"),
                              scope=scope, backend="fs", final_state=JarvisState.WAITING)
                self._pending_open = pending
                return True
            kind = "folder" if _P(value).is_dir() else "file"
        try:
            self.aliases.learn(name, kind, value)
            learned = True
        except Exception:  # noqa: BLE001
            learned = False
        self.emit(EventType.TOOL, {"summary": f"alias learned: „{name}“ = {kind} {value}" if learned else "alias not learned",
                                   "source": "aliases"}, scope=scope)
        if kind == "url":
            self._open_web_target(value, text, scope)
        else:
            self._open_path_target(value, name=name, scope=scope)
        if learned:
            self._deliver((f"Gemerkt: „{name}“ heißt ab jetzt {value}." if de
                           else f"Remembered: “{name}” now means {value}."),
                          scope=scope, backend="aliases", context_text=f"[alias {name} -> {kind} {value}]")
        return True

    def _handle_alias_teach(self, text: str, scope: str) -> bool:
        """"Wenn ich 'Lernplan' sage, meine ich D:\\...": persist the lesson."""

        from service.aliases import classify_target, parse_teach

        pair = parse_teach(text)
        if not pair:
            return False
        name, value = pair
        de = self.language.startswith("de")
        kind, norm = classify_target(value)
        if not kind:
            # not a path or URL: an installed app or a project?
            try:
                _, app_id, _ = self.apps.resolve(value)
            except Exception:  # noqa: BLE001
                app_id = ""
            if app_id:
                kind, norm = "app", value
            else:
                titles = {str(p.get("title") or "").lower() for p in self.list_projects()}
                if value.lower() in titles:
                    kind, norm = "project", value
        if not kind:
            self._deliver((f"Ich kann „{value}“ nicht zuordnen. Gib mir einen Pfad, eine Webadresse, einen App- oder Projektnamen." if de
                           else f"I cannot classify “{value}”. Give me a path, a web address, an app or a project name."),
                          scope=scope, backend="aliases", final_state=JarvisState.WAITING)
            return True
        if kind in {"file", "folder"}:
            from pathlib import Path as _P

            if not _P(norm).exists():
                self._deliver((f"{norm} existiert nicht – so merke ich es mir nicht." if de
                               else f"{norm} does not exist – I will not remember it like that."),
                              scope=scope, backend="aliases", final_state=JarvisState.WAITING)
                return True
        self.aliases.learn(name, kind, norm)
        self.emit(EventType.TOOL, {"summary": f"alias learned: „{name}“ = {kind} {norm}", "source": "aliases"}, scope=scope)
        self._deliver((f"Gemerkt: „{name}“ = {norm}." if de else f"Remembered: “{name}” = {norm}."),
                      scope=scope, backend="aliases", context_text=f"[alias {name} -> {kind} {norm}]")
        return True

    def _answer_correction(self, text: str, scope: str, action: Any) -> None:
        """"Nein, ich meinte Stockfish": classify the correction and act on it.

        STT corrections (the previous spoken request contained a look-alike of
        what the owner meant) become a bounded vocabulary rule and the
        corrected request runs again.  Everything else is attached to the last
        receipt through the correction memory, as before.
        """

        de = self.language.startswith("de")
        meant = str((action.arguments if action is not None else {}).get("meant") or "").strip() if action is not None else ""
        last_user = None
        with self._lock:
            users = [t for t in self._history if t.role == "user"]
            last_user = users[-2] if len(users) >= 2 else None
        heard_text = ""
        if last_user is not None:
            heard_text = str(last_user.meta.get("normalized") or last_user.text or "")
        # STT correction: which heard token does the meant word replace?
        if meant and last_user is not None and last_user.meta.get("source") in {"microphone", "ui_mic"}:
            from difflib import SequenceMatcher

            tokens = re.findall(r"[^\W\d_]+", heard_text, re.UNICODE)
            best, score = "", 0.0
            for token in tokens:
                if token.lower() == meant.lower():
                    continue
                ratio = SequenceMatcher(None, token.lower(), meant.lower()).ratio()
                if ratio > score:
                    best, score = token, ratio
            if best and score >= 0.5 and len(best) >= 3:
                learned = self.voice.vocabulary.learn(best, meant, note=text[:120])
                corrected = re.sub(r"(?<![\w'])" + re.escape(best) + r"(?![\w'])", meant, heard_text)
                self.emit(EventType.NOTIFICATION, {"kind": "correction", "text": f"STT correction: „{best}“ → „{meant}“", "classification": "STT_CORRECTION",
                                                   "heard": best, "meant": meant, "learned": learned}, scope=scope)
                try:
                    self.corrections.add(__import__("service.corrections", fromlist=["OwnerCorrection"]).OwnerCorrection(
                        original_request=heard_text, what_was_wrong=text.strip(), classification="STT_CORRECTION", scope="ENTITY_SPECIFIC",
                        parsed_intent="", entities={"heard": best, "meant": meant}, executed_action="", observed_result="",
                        receipt_id=getattr(self._session_receipts[-1], "id", "") if self._session_receipts else "",
                        when={"terms": [best.lower()]}, then={"note": f"{best} means {meant}", "overrides": {}}, provenance="owner-chat"))
                except Exception:  # noqa: BLE001 - the vocabulary rule is the durable part
                    pass
                # The mis-heard word already produced something durable?  A
                # project created as "Sprachtist" is renamed, not created twice.
                last_receipt = self._session_receipts[-1] if self._session_receipts else None
                if last_receipt is not None and last_receipt.kind == "project.create" and last_receipt.ok:
                    made = str(last_receipt.evidence.get("title") or "")
                    if best.lower() in made.lower():
                        from service.intents import ActionIntent

                        new_title = re.sub(re.escape(best), meant, made, flags=re.I)
                        intent = ActionIntent("project.rename", verb="rename", object_type="project", target=made, arguments={"title": new_title},
                                              confidence=0.9, success_criteria=[f"a project titled {new_title!r} exists"], reason="STT correction after a create")
                        self._deliver((f"Verstanden – {meant}. Ich merke mir das und benenne das Projekt um." if de
                                       else f"Understood – {meant}. I will remember that and rename the project."),
                                      scope=scope, backend="corrections", context_text=f"[STT correction {best} -> {meant}; project renamed]")
                        self._answer_by_project_operation(intent, text, scope)
                        return
                self._deliver((f"Verstanden – {meant}. Ich merke mir das und führe es korrigiert aus." if de
                               else f"Understood – {meant}. I will remember that and run it corrected."),
                              scope=scope, backend="corrections", context_text=f"[STT correction {best} -> {meant}; corrected request re-run]")
                self.send_message(corrected, scope=scope, meta={"source": "correction_rerun", "corrected_from": best, "meant": meant})
                return
        # Not a transcription error: an intent/entity/result correction attached to the last receipt.
        last = self._session_receipts[-1] if getattr(self, "_session_receipts", None) else None
        receipt_id = getattr(last, "id", "") if last is not None else ""
        classification = "ENTITY_RESOLUTION_ERROR" if meant else "INTENT_ERROR"
        try:
            if receipt_id or heard_text:
                self.correction_save(text.strip(), receipt_id=receipt_id, classification=classification, original_request=heard_text)
        except Exception:  # noqa: BLE001
            pass
        self.emit(EventType.NOTIFICATION, {"kind": "correction", "text": text[:160], "receipt_id": receipt_id}, scope=scope)
        if meant and heard_text:
            self._deliver((f"Verstanden – {meant}, nicht das, was ich verstanden hatte. Ich habe es notiert; sag es noch einmal ganz, dann führe ich es richtig aus." if de
                           else f"Understood – {meant}, not what I had understood. Noted; say it once more in full and I will do it right."),
                          scope=scope, backend="corrections", context_text="[owner correction recorded]")
            return
        self._deliver(
            (f"Verstanden – das war falsch. Ich habe es notiert{' (' + receipt_id + ')' if receipt_id else ''}. Sag mir, was du meintest, oder nutze „Korrigieren“ am letzten Ergebnis.")
            if de else
            (f"Understood – that was wrong. Noted{' (' + receipt_id + ')' if receipt_id else ''}. Tell me what you meant, or use “Korrigieren” on the last result."),
            scope=scope, backend="corrections", context_text="[owner correction signalled; handled through the correction memory]",
        )

    # -- the three answering paths --------------------------------------

    def _answer_by_acquisition(self, text: str, scope: str) -> None:
        """"Learn to do X": a capability-acquisition mission, started from the chat.

        Runs the same pipeline the music gap uses (local build, verification,
        escalation only after counted local failure, registration), as a
        durable engine mission with the acquisition's own steps as evidence.
        The conversation stays open; the verdict comes back when it is one.
        """

        from runtime.evidence import from_receipt, inference, owner_statement
        from service.acquisition import AcquisitionMission

        if self.missions.store.active():
            active = [m for m in self.missions.store.active() if m.kind == "capability"]
            if active:
                de = self.language.startswith("de")
                self._deliver((f"Ich lerne gerade schon etwas ({active[0].goal[:60]}). Das nächste nehme ich danach." if de else
                               f"I am already learning something ({active[0].goal[:60]}). I will take the next one after it."),
                              scope=scope, backend="acquisition")
                return
        goal = text.strip()
        mission = self.missions.create(goal, kind="capability", interpretation="acquire a missing primitive: local build, verify, register",
                                       acceptance=["the capability is registered and verified", "a second invocation uses it directly"], scope=scope)
        self.missions.add_evidence(mission, owner_statement(goal))
        de = self.language.startswith("de")
        self._deliver(
            (f"Verstanden, das lerne ich jetzt (Mission {mission.mission_id}): lokaler Build, Verifikation, Registrierung; Experte nur bei "
             f"nachgewiesenem lokalem Scheitern. Ich melde mich, wenn es verifiziert ist.") if de else
            (f"Understood, I will learn that now (mission {mission.mission_id}): local build, verification, registration; the expert only after "
             f"counted local failure. I will report when it is verified."),
            scope=scope, backend="acquisition", final_state=JarvisState.CODING,
        )

        def work() -> None:
            try:
                self.missions.transition(mission, "PLAN", "acquisition pipeline")
                self.missions.transition(mission, "EXECUTE", "local build")
                acq = AcquisitionMission(service=self.capabilities, kernel=self.kernel,
                                         emit=lambda kind, payload: self.emit(kind, payload, scope=scope))
                # "Learn X" names a thing to build, never a thing to look up:
                # the registry's term matcher once answered a word-count goal
                # with the Spotify provider.  A fresh id and the goal's own
                # terms as keywords make the build the only outcome, and let
                # the next request find what was built.
                from capabilities.registry import ADDRESS_TERMS, BOILERPLATE
                from development.experience import terms as goal_terms

                # German function words must never become keywords: a stored
                # "einer" once matched "Öffne Wikipedia" to a word counter.
                words = [w for w in goal_terms(goal)
                         if w not in {"lerne", "lern", "learn", "wie", "man", "how", "to"}
                         and w not in BOILERPLATE and w not in ADDRESS_TERMS]
                cid = "learned." + "_".join(words[:3])[:48] if words else f"learned.{mission.mission_id}"
                result = acq.run(goal, capability_id=cid, keywords=words[:12])
                for step in getattr(result, "steps", [])[-12:]:
                    self.missions.add_evidence(mission, inference(f"{getattr(step, 'phase', '')}: {getattr(step, 'summary', '')}"[:200],
                                                                  tier="BUILD_LOCAL", confidence=0.5))
                if result.escalated:
                    self.missions.transition(mission, "ESCALATE", f"expert {result.expert_used or 'used'}")
                self.missions.transition(mission, "VERIFY", f"acquired={result.acquired} {result.reason[:120]}")
                if result.acquired and result.capability_id:
                    manifest = self.capabilities.registry.get(result.capability_id)
                    receipt = {"id": f"cap_{result.capability_id}", "kind": "capability.acquire", "executor": "acquisition", "ok": True,
                               "verified": bool(manifest is not None), "detail": f"{result.capability_id} registered",
                               "verifications": [{"check": "registry lists the capability", "passed": manifest is not None,
                                                  "observed": str(getattr(manifest, "status", "")) if manifest else "absent"}]}
                    self.missions.add_evidence(mission, from_receipt(receipt))
                    self.missions.transition(mission, "COMPLETE", f"{result.capability_id} in {result.seconds:.0f}s")
                    self._deliver(
                        (f"Gelernt und verifiziert: {result.capability_id} ({result.seconds:.0f}s, {result.local_attempts} lokale Versuche, "
                         f"{'mit' if result.escalated else 'ohne'} Experten). Ab jetzt nutze ich es direkt.") if de else
                        (f"Learned and verified: {result.capability_id} ({result.seconds:.0f}s, {result.local_attempts} local attempts, "
                         f"{'with' if result.escalated else 'without'} an expert). From now on I use it directly."),
                        scope=scope, backend="acquisition", context_text=f"[capability {result.capability_id} acquired; mission {mission.mission_id}]")
                else:
                    self.missions.fail_approach(mission, "acquisition pipeline", result.reason[:300])
                    self.missions.transition(mission, "FAILED", result.reason[:200] or "not acquired")
                    self._deliver((f"Nicht gelernt: {result.reason[:200]}" if de else f"Not learned: {result.reason[:200]}"),
                                  scope=scope, backend="acquisition", final_state=JarvisState.ERROR)
            except Exception as exc:  # noqa: BLE001
                self.missions.settle(mission, exc)
                self._deliver((f"Akquise fehlgeschlagen: {exc}" if de else f"Acquisition failed: {exc}"), scope=scope, backend="acquisition",
                              final_state=JarvisState.ERROR)

        threading.Thread(target=work, daemon=True, name=f"acquire-{mission.mission_id}").start()

    def _answer_by_research(self, text: str, scope: str) -> None:
        """A question about the current state of the world: sources, not memory.

        A durable research mission carries the question, the queries, the
        sources with provenance and freshness, the findings and the
        contradictions; the answer the owner reads names its sources and its
        confidence, and says when it could not reach any.
        """

        from runtime.evidence import external, inference

        mission = self.missions.create(text, kind="research", interpretation="answer from sources with provenance", scope=scope)
        self.state.set(JarvisState.RESEARCHING, detail=text[:120], scope=scope)

        def work() -> None:
            de = self.language.startswith("de")
            try:
                self.missions.transition(mission, "RESEARCH", "querying sources")
                report = self.research(text, max_sources=4)
                if not report.get("ok", True) and report.get("error"):
                    raise RuntimeError(report["error"])
                sources = report.get("sources", [])
                findings = report.get("findings", [])
                if sources:
                    # research results are web context too: "fass den
                    # wichtigsten Artikel davon zusammen" must know "davon"
                    self._web_context = {"query": text, "at": time.time(),
                                         "results": [{"title": str(s.get("title") or ""), "url": str(s.get("url") or ""), "snippet": ""}
                                                     for s in sources if s.get("url")]}
                for s in sources[:8]:
                    self.missions.add_evidence(mission, external(str(s.get("title") or s.get("url", "")), url=str(s.get("url", "")),
                                                                 fetched_at=str(s.get("fetched_at", "")), authority=int(s.get("authority", 0) or 0)))
                for f in findings[:8]:
                    self.missions.add_evidence(mission, inference(str(f.get("claim") or f.get("text", "")), confidence=float(f.get("confidence", 0.4) or 0.4)))
                contradictions = report.get("contradictions", [])
                self.missions.transition(mission, "VERIFY", f"{len(sources)} sources, {len(findings)} findings, {len(contradictions)} contradictions")
                summary = str(report.get("summary", "")).strip()
                lines = [summary or ("Keine belastbare Antwort aus Quellen." if de else "No sourced answer could be established.")]
                if sources:
                    lines.append("")
                    lines.append("Quellen:" if de else "Sources:")
                    for s in sources[:6]:
                        lines.append(f"- {s.get('title') or s.get('url', '')} — {s.get('url', '')}" + (f" ({s.get('fetched_at', '')[:10]})" if s.get("fetched_at") else ""))
                if contradictions:
                    lines.append("")
                    lines.append(("Widersprüche zwischen Quellen: " if de else "Contradictions between sources: ") + str(len(contradictions)))
                if report.get("offline"):
                    lines.append("(offline: no source could be fetched; this is from memory and is unverified)" if not de else "(offline: keine Quelle erreichbar; das ist aus dem Gedächtnis und unverifiziert)")
                if not sources:
                    self.missions.block(mission, "no source reachable", owner_input="")
                else:
                    mission.next_action = "answered; open the mission for sources and contradictions"
                    self.missions.store.save(mission)
                self._deliver("\n".join(lines), scope=scope, backend="research",
                              context_text=f"[research mission {mission.mission_id}: {len(sources)} sources]",
                              final_state=JarvisState.IDLE)
            except Exception as exc:  # noqa: BLE001
                self.missions.settle(mission, exc)
                self._deliver((f"Recherche fehlgeschlagen: {exc}" if de else f"Research failed: {exc}"), scope=scope, backend="research",
                              final_state=JarvisState.ERROR)

        threading.Thread(target=work, daemon=True, name=f"research-{mission.mission_id}").start()

    # ------------------------------------------------------------------
    # The Mission Engine -- one durable record per long job, of every kind
    # ------------------------------------------------------------------

    @property
    def missions(self) -> Any:
        from runtime.mission_engine import MissionEngine, MissionEngineStore

        if getattr(self, "_missions_engine", None) is None:
            self._missions_engine = MissionEngine(
                MissionEngineStore(Path(self.kernel.state_root) / "engine"),
                emit=lambda kind, payload: self.emit(EventType(kind) if kind in {e.value for e in EventType} else EventType.PROGRESS, payload),
            )
        return self._missions_engine

    @staticmethod
    def mission_title(goal: str) -> str:
        """A durable, concise title from the owner's sentence; the prompt itself stays in the record."""

        text = " ".join(str(goal or "").split())
        text = re.sub(r"^(?:hey\s+|ok\s+)?(?:zeus|jarvis)\s*[,:!.-]?\s*", "", text, flags=re.I)
        first = re.split(r"(?<=[.!?])\s|\s(?:und dann|danach|and then)\s", text, maxsplit=1)[0].strip().rstrip(".!?,;:")
        if len(first) > 64:
            cut = first[:64]
            first = cut[: cut.rfind(" ")] if " " in cut[20:] else cut
            first = first.rstrip(",;:") + "…"
        return (first[:1].upper() + first[1:]) if first else (text[:64] or "(no goal recorded)")

    @staticmethod
    def mission_family(goal: str) -> str:
        """Attempts at the same request share a family: the normalised sentence."""

        import hashlib

        normalised = re.sub(r"[^\w\s]", "", str(goal or "").lower())
        normalised = " ".join(normalised.split())
        return hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:12] if normalised else ""

    @staticmethod
    def mission_state(row: dict[str, Any]) -> str:
        """One state word for filters: active | waiting | blocked | paused | failed | cancelled | completed."""

        phase = str(row.get("phase", "")).upper()
        outcome = str(row.get("outcome", "")).lower()
        if outcome in {"cancelled", "canceled"} or phase == "CANCELLED":
            return "cancelled"
        if outcome in {"failed", "rolled_back"} or phase == "FAILED":
            return "failed"
        if outcome in {"complete", "completed", "promoted", "acquired"} or phase in {"DONE", "COMPLETE"}:
            return "completed"
        if phase == "PAUSED":
            return "paused"
        if phase == "BLOCKED":
            return "blocked"
        if row.get("owner_input_required") or phase == "WAITING":
            return "waiting"
        return "active"

    def list_missions(self, *, status: str = "") -> dict[str, Any]:
        """Every long-running job, whichever system runs it, in one shape."""

        rows: list[dict[str, Any]] = []
        for m in self.missions.store.list():
            rows.append({"id": m.mission_id, "kind": m.kind, "goal": m.goal, "phase": m.phase, "outcome": m.outcome or ("running" if not m.finished else ""),
                         "started": m.created_at, "updated": m.updated_at, "finished": m.finished, "next_action": m.next_action, "reason": getattr(m, "reason", ""),
                         "blockers": m.blockers, "owner_input_required": m.owner_input_required, "system": "engine",
                         "tasks": {"done": len(m.completed), "total": len(m.tasks)}, "evidence": len(m.evidence)})
        try:
            for m in self.selfdev_store.list():
                rows.append({"id": m.mission_id, "kind": "selfdev", "goal": m.request, "phase": m.phase, "outcome": m.outcome or ("running" if not m.finished else ""),
                             "started": m.started_at, "updated": m.updated_at, "finished": m.finished, "next_action": "", "reason": getattr(m, "reason", ""),
                             "blockers": [], "owner_input_required": "", "system": "selfdev",
                             "tasks": {"done": 0, "total": 0}, "evidence": len(m.isolation) + len(m.verification.get("checks", []))})
        except Exception:  # noqa: BLE001
            pass
        try:
            from runtime.missions import MissionStore

            for c in MissionStore(Path(self.kernel.state_root) / "missions").all():
                d = c.to_dict() if hasattr(c, "to_dict") else {}
                rows.append({"id": getattr(c, "mission_id", ""), "kind": "capability", "goal": f"{getattr(c, 'capability_id', '')}: {getattr(c, 'defect', '') or 'acquire'}",
                             "phase": str(getattr(c, "phase", "")).upper(), "outcome": "acquired" if getattr(c, "acquired", False) else "checkpoint",
                             "started": getattr(c, "started_at", ""), "updated": getattr(c, "updated_at", ""), "finished": bool(getattr(c, "acquired", False)),
                             "next_action": getattr(c, "next_action", ""), "blockers": [], "owner_input_required": "", "system": "acquisition",
                             "tasks": {"done": 0, "total": 0}, "evidence": len(getattr(c, "attempts", []) or [])})
        except Exception:  # noqa: BLE001
            pass
        for r in rows:
            r["title"] = self.mission_title(r.get("goal", ""))
            r["family"] = self.mission_family(r.get("goal", ""))
            r["state"] = self.mission_state(r)
            r["deployment"] = "promoted" if r.get("outcome") == "promoted" else ("rolled back" if r.get("outcome") == "rolled_back" else "")
        families: dict[str, int] = {}
        for r in rows:
            families[r["family"]] = families.get(r["family"], 0) + 1
        for r in rows:
            r["attempts"] = families.get(r["family"], 1)
        if status == "active":
            rows = [r for r in rows if r["state"] in {"active", "waiting"}]
        elif status == "blocked":
            rows = [r for r in rows if r["state"] == "blocked"]
        elif status in {"completed", "failed", "cancelled", "paused", "waiting"}:
            rows = [r for r in rows if r["state"] == status]
        rows.sort(key=lambda r: str(r.get("updated", "")), reverse=True)
        return {"missions": rows, "count": len(rows)}

    def mission_detail(self, mission_id: str) -> dict[str, Any]:
        m = self.missions.store.load(mission_id)
        if m is not None:
            return {"ok": True, "system": "engine", "mission": m.to_dict(), "brief": self.missions.brief(m)}
        sd = self.selfdev_store.load(mission_id)
        if sd is not None:
            return {"ok": True, "system": "selfdev", "mission": sd.to_dict()}
        return {"ok": False, "error": f"no mission {mission_id}"}

    def mission_control(self, mission_id: str, action: str) -> dict[str, Any]:
        if self.missions.store.load(mission_id) is not None:
            if action == "cancel":
                return self.missions.request_cancel(mission_id)
            if action == "pause":
                return self.missions.request_pause(mission_id)
            if action == "resume":
                return self.missions.resume(mission_id)
            return {"ok": False, "error": f"unknown action {action}"}
        if action == "cancel":
            return self.cancel_selfdev(mission_id)
        if action == "resume":
            return self.resume_selfdev(mission_id)
        return {"ok": False, "error": "self-development missions can be cancelled or resumed"}

    def _answer_owner_config(self, text: str, scope: str, classification: Any) -> None:
        """A change to the owner core never happens from a chat sentence.

        It is prepared as far as it safely can be -- named, pointed at the
        Owner Settings flow -- and stops there.  The owner approves a diff.
        """

        de = self.language.startswith("de")
        terms = ", ".join(classification.route.reading.core_terms[:3]) if classification.route else ""
        self.emit(EventType.NOTIFICATION, {"kind": "owner_config", "text": f"owner-core change requested: {text[:160]}",
                                           "terms": terms}, scope=scope)
        self._deliver(
            (f"Das betrifft meinen Owner-Kern ({terms or 'Identität/Persönlichkeit/Policy'}). "
             f"Den ändere ich nicht aus einem Chatsatz heraus: öffne „Owner“ in den Einstellungen, dort schlage ich "
             f"die Änderung als Diff vor und du bestätigst sie ausdrücklich. Kein Selbst-Update wurde gestartet.")
            if de else
            (f"That concerns my owner core ({terms or 'identity/personality/policy'}). I do not change it from a chat "
             f"sentence: open “Owner” in Settings, where I propose the change as a diff for your explicit approval. "
             f"No self-update was started."),
            scope=scope, backend="owner",
            context_text="[owner-core change requested; deferred to the Owner Settings transaction]",
        )

    def _answer_correction_hint(self, text: str, scope: str) -> None:
        """"No, that was wrong": the correction memory is the place for it."""

        de = self.language.startswith("de")
        last = self._session_receipts[-1] if getattr(self, "_session_receipts", None) else None
        receipt_id = getattr(last, "id", "") if last is not None else ""
        self.emit(EventType.NOTIFICATION, {"kind": "correction", "text": text[:160], "receipt_id": receipt_id}, scope=scope)
        self._deliver(
            (f"Verstanden – das war falsch. Nutze „Korrigieren“ am letzten Ergebnis"
             f"{' (' + receipt_id + ')' if receipt_id else ''}, dann lerne ich es dauerhaft und kann es korrigiert erneut ausführen.")
            if de else
            (f"Understood – that was wrong. Use “Korrigieren” on the last result"
             f"{' (' + receipt_id + ')' if receipt_id else ''}; I then learn it durably and can re-run it corrected."),
            scope=scope, backend="corrections",
            context_text="[owner correction signalled; handled through the correction memory]",
        )

    def _answer_from_records(self, text: str, scope: str, classification: Any) -> None:
        """A question about this system, answered from its registries."""

        from service.intent import Intent
        from service.reads import answer as read_answer

        self.state.set(JarvisState.WORKING, detail=classification.reason[:120], scope=scope)
        try:
            body = read_answer(
                self, text, language=self.language,
                acquisition=classification.intent is Intent.CAPABILITY,
            )
        except Exception as exc:
            self.state.set(JarvisState.ERROR, detail=str(exc)[:200])
            self.emit(EventType.ERROR, {"error": f"{type(exc).__name__}: {exc}"}, scope=scope)
            return
        self.emit(
            EventType.TOOL,
            {"summary": f"read: {classification.reason}", "source": "registry"},
            scope=scope,
        )
        self._deliver(
            body, scope=scope, backend="registry",
            context_text=f"[answered from the registry: {classification.reason}]",
        )

    def _answer_musically(self, text: str, scope: str, classification: Any) -> None:
        """A music request, routed to whichever provider the user prefers.

        No model runs on this path at all.  The request is parsed
        deterministically, the provider comes from a stored preference, and the
        outcome is read back from the operating system -- so "Pause." costs a
        media-session call and nothing else, and a music turn can never wake
        BUILD_LOCAL or an expert.
        """

        from service.music import compose as compose_music
        from service.music import understand

        request = understand(text)
        if request is None:  # pragma: no cover - the classifier only sends music here
            self._answer_conversationally(text, scope)
            return

        self.state.set(JarvisState.WORKING, detail=f"music: {request.action}", scope=scope)
        self.emit(
            EventType.TOOL,
            {"summary": f"music.{request.action}" + (f" {request.query!r}" if request.query else ""),
             "music": request.to_dict()},
            scope=scope,
        )

        outcome = self.music.run(request)

        # A capability gap is not a failure. It is a request that arrived before
        # the thing that serves it existed, so the thing gets built and the
        # original request is then answered -- without the user asking twice.
        if outcome.gap:
            outcome = self._acquire_then_retry(request, scope)
        elif outcome.defect and self._defect_confirmed(outcome):
            # The provider exists and is broken. Same shape, different cause:
            # repair it and answer the question that found the defect. Only
            # once -- a repair that does not fix it must surface as a failure
            # rather than as a loop.
            outcome = self._acquire_then_retry(request, scope, repair=outcome.defect)

        if outcome.receipt.verified and outcome.capability_id:
            self._defects.pop(outcome.capability_id, None)

        self.state.set(JarvisState.VERIFYING, detail=outcome.receipt.kind, scope=scope)
        self.receipts.record(outcome.receipt)
        self._session_receipts.append(outcome.receipt)
        self.emit(
            EventType.TOOL,
            {"summary": outcome.receipt.summary(), "receipt_id": outcome.receipt.id,
             "receipt": outcome.receipt.to_dict()},
            scope=scope,
        )
        self._deliver(
            compose_music(outcome, language=self.language),
            scope=scope,
            backend=outcome.capability_id or "music.resolver",
            context_text=f"[music.{request.action}: "
            f"{'verified' if outcome.receipt.verified else 'not verified'}, "
            f"receipt {outcome.receipt.id}]",
            final_state=JarvisState.IDLE if outcome.receipt.verified else JarvisState.ERROR,
        )

    # ------------------------------------------------------------------
    # Composition: a plan over primitives ZEUS already has
    # ------------------------------------------------------------------

    def _composer(self) -> Any:
        from service.composer import Composer

        try:
            manifests = [m.to_dict() for m in self.capabilities.registry.all()]
        except Exception:  # noqa: BLE001
            manifests = []
        context = self.device_context().get("available", ["screen", "speaker", "microphone"])
        return Composer(capabilities=manifests, context_requirements=context)

    def _answer_by_composition(self, text: str, scope: str, *, guidance: str = "", allow_single: bool = False) -> bool:
        """Plan typed steps over existing primitives and run them as a mission.

        Returns True when the request was handled here (executed, or a gap
        was named), False when composition found nothing to compose and the
        ordinary single-action path should take over.
        """

        from brain.tiers import ModelTier
        from runtime.evidence import from_receipt, owner_statement
        from service.composer import Step

        from service.composer import evaluate_goal, extract_constraints

        composer = self._composer()
        provider = self.kernel.provider(ModelTier.FAST_LOCAL)
        constraints = extract_constraints(text)
        plan = composer.plan(text, provider, guidance=guidance, constraints=constraints)
        self.emit(EventType.TOOL, {"summary": f"composition: {plan.mode}, {len(plan.steps)} step(s)" + (f", missing {plan.missing}" if plan.missing else "")
                                   + (f", forbidden {plan.forbidden}" if plan.forbidden else ""),
                                   "plan": plan.to_dict(), "source": "composer"}, scope=scope)
        real = [s for s in plan.steps if s.status not in {"missing", "forbidden"}]
        uses_capability = any(s.step.startswith("capability:") for s in real)
        if plan.mode == "answering" or (len(real) < 2 and not plan.missing and not plan.forbidden and not constraints.required_outcome
                                        and not (allow_single and uses_capability)):
            return False
        de = self.language.startswith("de")
        mission = self.missions.create(text, kind="complex", interpretation=f"composed from {', '.join(s.step for s in plan.steps)}",
                                       constraints=[f"forbidden: {a}" for a in constraints.forbidden_actions]
                                       + ([f"required outcome: {', '.join(constraints.required_outcome)}"] if constraints.required_outcome else [])
                                       + (["no fallbacks"] if constraints.fallbacks_forbidden else []),
                                       acceptance=[s.purpose or s.step for s in plan.steps if s.status != "forbidden"], scope=scope)
        if plan.forbidden:
            self.emit(EventType.TOOL, {"summary": f"plan refused {len(plan.forbidden)} forbidden step(s): {', '.join(plan.forbidden)}",
                                       "forbidden": plan.forbidden, "evidence": constraints.evidence, "source": "composer"}, scope=scope)
        if not any(s.status == "planned" for s in plan.steps) and not plan.missing:
            self.missions.block(mission, "every planned step is forbidden by the request", owner_input="")
            self._deliver((f"Das ginge nur über {', '.join(plan.forbidden)} — und genau das hast du ausgeschlossen. Ich habe nichts angelegt."
                           if de else f"That would need {', '.join(plan.forbidden)}, which you ruled out. Nothing was created."),
                          scope=scope, backend="composer", final_state=JarvisState.WAITING,
                          context_text=f"[composition mission {mission.mission_id}: all steps forbidden]")
            return True
        self.missions.add_evidence(mission, owner_statement(text))
        for s in plan.steps:
            self.missions.add_task(mission, f"{s.step} {json.dumps(s.arguments, ensure_ascii=False)[:80]}")
        if plan.missing:
            self.missions.block(mission, f"missing primitive(s): {', '.join(plan.missing)}", owner_input="")
            self._deliver(
                (f"Das kann ich zusammensetzen aus: {', '.join(s.step for s in plan.steps if s.status != 'missing')}. "
                 f"Mir fehlt dafür noch: {', '.join(plan.missing)}. Sag „lerne …“, dann baue ich genau das und führe den Rest dann aus.")
                if de else
                (f"I can compose this from: {', '.join(s.step for s in plan.steps if s.status != 'missing')}. "
                 f"What I still lack: {', '.join(plan.missing)}. Say “learn …” and I will acquire exactly that, then run the rest."),
                scope=scope, backend="composer", final_state=JarvisState.WAITING,
                context_text=f"[composition mission {mission.mission_id}: gap {plan.missing}]",
            )
            return True

        self.missions.transition(mission, "PLAN", f"{len(plan.steps)} typed steps")
        self.missions.transition(mission, "EXECUTE", "running the steps in order")
        task_ids = [t["task_id"] for t in mission.tasks]

        def run_step(step: Step) -> Any:
            return self._run_primitive(step, text, scope)

        def on_step(step: Step, receipt: Any) -> None:
            idx = plan.steps.index(step)
            ev = self.missions.add_evidence(mission, from_receipt(receipt))
            self.missions.update_task(mission, task_ids[idx], status="done" if step.status == "done" else "failed",
                                      result=step.detail, evidence_id=ev.evidence_id)
            self.receipts.record(receipt)
            self._session_receipts.append(receipt)
            self.emit(EventType.TOOL, {"summary": receipt.summary(), "receipt_id": receipt.id, "receipt": receipt.to_dict()}, scope=scope)

        def replan(current: Any, failed_step: Step) -> Any:
            self.missions.fail_approach(mission, f"step {failed_step.step}", failed_step.detail)
            fresh = composer.replan(current, failed_step, provider, guidance=guidance)
            if fresh is None:
                self.emit(EventType.TOOL, {"summary": f"no replan for {failed_step.step}; the remainder stops", "source": "composer"}, scope=scope)
                return None
            self.missions.transition(mission, "DIAGNOSE", f"{failed_step.step} failed: {failed_step.detail[:100]}; replanned")
            self.emit(EventType.TOOL, {"summary": f"replan after {failed_step.step}: {', '.join(s.step for s in fresh.steps)}",
                                       "plan": fresh.to_dict(), "source": "composer"}, scope=scope)
            for s in fresh.steps:
                self.missions.add_task(mission, f"{s.step} {json.dumps(s.arguments, ensure_ascii=False)[:80]}")
            task_ids[:] = [t["task_id"] for t in mission.tasks]
            self.missions.transition(mission, "EXECUTE", "running the replanned steps")
            return fresh

        receipts = composer.execute(plan, run_step, on_step=on_step, replan=replan)
        done = [s for s in plan.steps if s.status == "done"]
        failed = [s for s in plan.steps if s.status == "failed"]
        required_failed = [s for s in failed if s.required]
        goal = evaluate_goal(plan, receipts)
        self.missions.transition(mission, "VERIFY", f"{len(done)} done, {len(failed)} failed; "
                                 f"EXECUTION_VERIFIED={goal.execution_verified} GOAL_SATISFIED={goal.goal_satisfied}")
        self.emit(EventType.TOOL, {"summary": f"goal: {'SATISFIED' if goal.goal_satisfied else 'NOT satisfied'}"
                                   + ("" if goal.goal_satisfied else " — " + "; ".join(goal.reasons)[:200]),
                                   "goal": goal.to_dict(), "mission_id": mission.mission_id, "source": "composer"}, scope=scope)
        self.think("mission_finished")
        if goal.goal_satisfied and self.missions.has_proof(mission):
            self.missions.transition(mission, "COMPLETE", "goal satisfied: every required step verified, constraints held")
        elif required_failed:
            self.missions.transition(mission, "FAILED", f"{required_failed[-1].step}: {required_failed[-1].detail[:120]}")
        elif not goal.goal_satisfied:
            self.missions.block(mission, "; ".join(goal.reasons)[:300], owner_input="")
        else:
            self.missions.transition(mission, "DIAGNOSE", "steps ran but the mission holds no proof")
        lines = []
        for s in plan.steps:
            mark = {"done": "✓", "failed": "✗", "replanned": "↻", "forbidden": "⛔", "skipped": "·"}.get(s.status, "·")
            role = "" if s.role == "required" else f" [{s.role}]"
            lines.append(f"{mark} {s.step}{role}" + (f" — {s.detail[:90]}" if s.detail else ""))
        if goal.goal_satisfied:
            head = f"Ziel erreicht ({len(done)} Schritte, verifiziert)" if de else f"Goal satisfied ({len(done)} steps, verified)"
        elif required_failed:
            head = (f"Nicht geschafft — {required_failed[-1].step} ist fehlgeschlagen" if de else f"Not done — {required_failed[-1].step} failed")
        else:
            head = ("Schritte ausgeführt, aber das Ziel ist NICHT erreicht: " if de else "Steps ran, but the goal is NOT met: ") + "; ".join(goal.reasons)[:160]
        self._deliver(head + ":\n" + "\n".join(lines), scope=scope, backend="composer",
                      context_text=f"[composition mission {mission.mission_id}: {mission.phase}; goal_satisfied={goal.goal_satisfied}]",
                      final_state=JarvisState.IDLE if goal.goal_satisfied else JarvisState.ERROR)
        return True

    def _run_primitive(self, step: Any, request: str, scope: str) -> Any:
        """One typed step -> one receipt.  Nothing here runs a shell."""

        from runtime.receipts import Receipt, Verification, failed
        from service.actions import ActionPlan

        name, args = step.step, dict(step.arguments)
        if name in {"file.write", "file.read", "project.create"}:
            return self.actions.execute(ActionPlan(name, arguments=args), request=request)
        if name.startswith("music."):
            from service.music import MusicRequest

            outcome = self.music.run(MusicRequest(name.split(".", 1)[1], query=str(args.get("query", ""))))
            return outcome.receipt
        if name.startswith("capability:"):
            from runtime.paths import PathError, resolve_workspace_path

            cid = name.split(":", 1)[1]
            # A relative file argument means the workspace, as it does for
            # every built-in action; a capability sees an absolute path.  One
            # resolver, idempotent, existence-checked for inputs that are read.
            workspace = Path(self.kernel.state_root) / "workspace"
            try:
                for key, value in list(args.items()):
                    if isinstance(value, str) and ("path" in key.lower() or key.lower() in {"file", "source", "folder", "directory"}) and value:
                        args[key] = str(resolve_workspace_path(workspace, value, must_exist=key.lower() not in {"output", "output_path", "target", "destination"}))
            except PathError as exc:
                return Receipt(kind=f"capability.{cid}", executor=cid, ok=False, detail=str(exc)[:300],
                               verifications=[Verification(check="input path resolves inside the workspace", passed=False, observed=str(exc)[:200])],
                               evidence={"capability_id": cid, "arguments": args})
            execution = self.capabilities.execute(cid, args)
            ok = bool(getattr(execution, "ok", False))
            output = getattr(execution, "output", {}) or {}
            error = str(getattr(execution, "error", "") or (output.get("error", "") if isinstance(output, dict) else ""))
            summary = ", ".join(f"{k}={v}" for k, v in output.items() if k not in {"ok", "error"} and not isinstance(v, (dict, list)))[:240] if isinstance(output, dict) else str(output)[:240]
            return Receipt(kind=f"capability.{cid}", executor=cid, ok=ok, detail=summary if ok else (error or "the capability reported a failure")[:300],
                           verifications=[Verification(check="capability reported ok", passed=ok, observed=summary or error[:200]),
                                          Verification(check="capability duration", passed=True, observed=f"{getattr(execution, 'duration_seconds', 0.0):.2f}s")],
                           evidence={"capability_id": cid, "arguments": args, "output": output if isinstance(output, dict) else {}})
        if name == "note.create":
            title = str(args.get("title", "note")).strip() or "note"
            safe = "".join(ch for ch in title if ch.isalnum() or ch in " -_").strip().replace(" ", "_")[:60] or "note"
            return self.actions.execute(ActionPlan("file.write", arguments={"path": f"notizen/{safe}.md", "content": f"# {title}\n\n{args.get('text', '')}\n"}), request=request)
        if name == "knowledge.search":
            graph = self.knowledge_graph(query=str(args.get("query", "")), limit=12)
            nodes = graph.get("nodes", [])
            return Receipt(kind="knowledge.search", executor="knowledge", ok=True, detail=f"{len(nodes)} node(s): " + ", ".join(str(n.get("title", "")) for n in nodes[:5]),
                           verifications=[Verification(check="graph queried", passed=True, observed=f"{len(nodes)} nodes")], evidence={"query": args.get("query", "")})
        if name == "knowledge.create":
            result = self.knowledge_create(str(args.get("title", "")), str(args.get("text", args.get("content", ""))), type=str(args.get("type", "note")),
                                           tags=args.get("tags") or (), links=args.get("links") or (), provenance="owner request",
                                           metadata={"request": request[:300]})
            ok = bool(result.get("ok"))
            return Receipt(kind="knowledge.create", executor="knowledge", ok=ok,
                           detail=(f"stored {result.get('type')} '{result.get('title')}' with {len(result.get('relations', []))} relation(s)" if ok else result.get("error", "failed"))[:300],
                           verifications=[Verification(check="node read back from the graph", passed=bool(result.get("read_back")), observed=str(result.get("node_id", ""))),
                                          Verification(check="node found by search", passed=bool(result.get("searchable")), observed=str(result.get("title", ""))),
                                          Verification(check="relations exist", passed=(not args.get("links")) or bool(result.get("relations")),
                                                       observed=", ".join(f"{r['relation']}->{r['target']}" for r in result.get("relations", [])) or "none")],
                           evidence={"node_id": result.get("node_id"), "relations": result.get("relations", []), "path": result.get("path")})
        if name == "knowledge.link":
            result = self.knowledge_link(str(args.get("source", "")), str(args.get("target", "")), str(args.get("relation", "relates_to")), provenance="owner request")
            ok = bool(result.get("ok"))
            return Receipt(kind="knowledge.link", executor="knowledge", ok=ok, detail=(f"{result.get('source')} -{result.get('relation')}-> {result.get('target')}" if ok else result.get("error", "failed"))[:300],
                           verifications=[Verification(check="edge exists", passed=ok, observed=str(result.get("edge_id", result.get("error", ""))))], evidence=result)
        if name == "knowledge.read":
            result = self.knowledge_read(str(args.get("title", args.get("id", ""))))
            ok = bool(result.get("ok"))
            return Receipt(kind="knowledge.read", executor="knowledge", ok=ok, detail=(str(result.get("title", result.get("node", {}).get("title", "")))[:100] if ok else result.get("error", ""))[:300],
                           verifications=[Verification(check="node read", passed=ok, observed=str(result.get("id", result.get("error", ""))))], evidence={"reference": args.get("title", "")})
        if name == "timer.start":
            return self.start_timer(float(args.get("minutes", 0) or 0), label=str(args.get("label", "")), scope=scope)
        if name == "file.open":
            return self.open_path(str(args.get("path", "")))
        if name == "window.hide":
            result = self.lifecycle.window("hide", reason="composed step")
            return Receipt(kind="window.hide", executor="desktop", ok=bool(result.get("ok")), detail=result.get("action", ""),
                           verifications=[Verification(check="window hidden", passed=bool(result.get("ok")), observed=str(result.get("action", "")))])
        if name == "say":
            text = str(args.get("text", ""))
            self._deliver(text, scope=scope, backend="composer")
            return Receipt(kind="say", executor="core", ok=True, detail=text[:200], verifications=[Verification(check="delivered", passed=True, observed="message event")])
        return failed(f"step.{name}", "composer", f"no executor for primitive {name}", request=request)

    # -- small built-in primitives ------------------------------------

    def start_timer(self, minutes: float, *, label: str = "", scope: str = "") -> Any:
        from runtime.receipts import Receipt, Verification

        if minutes <= 0 or minutes > 24 * 60:
            return Receipt(kind="timer.start", executor="timer", ok=False, detail=f"{minutes} minutes is not a timer",
                           verifications=[Verification(check="duration sane", passed=False, observed=str(minutes))])
        timers = getattr(self, "_timers", None)
        if timers is None:
            timers = self._timers = {}
        timer_id = f"timer_{int(time.time())}_{len(timers)}"
        de = self.language.startswith("de")

        def fire() -> None:
            timers.pop(timer_id, None)
            text = (f"Timer abgelaufen: {label or f'{minutes:g} Minuten'}." if de else f"Timer finished: {label or f'{minutes:g} minutes'}.")
            self.emit(EventType.NOTIFICATION, {"kind": "timer", "text": text, "timer_id": timer_id}, scope=scope)
            try:
                self.voice.speak_stream([text], scope=scope)
            except Exception:  # noqa: BLE001
                pass

        t = threading.Timer(minutes * 60, fire)
        t.daemon = True
        t.start()
        timers[timer_id] = {"label": label, "minutes": minutes, "started": time.time(), "timer": t}
        return Receipt(kind="timer.start", executor="timer", ok=True, detail=f"{minutes:g} min timer{f' ({label})' if label else ''} started",
                       verifications=[Verification(check="timer registered", passed=timer_id in timers, observed=timer_id)],
                       evidence={"timer_id": timer_id, "minutes": minutes, "label": label})

    def list_timers(self) -> dict[str, Any]:
        timers = getattr(self, "_timers", {}) or {}
        return {"timers": [{"id": k, "label": v["label"], "minutes": v["minutes"], "remaining_seconds": max(0, round(v["started"] + v["minutes"] * 60 - time.time()))}
                           for k, v in timers.items()]}

    def open_path(self, path: str) -> Any:
        from runtime.receipts import Receipt, Verification

        root = Path(self.kernel.state_root) / "workspace"
        target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        try:
            inside = target.is_relative_to(root.resolve()) or target.is_relative_to(Path.home().resolve())
        except (OSError, ValueError):
            inside = False
        if not target.exists() or not inside:
            return Receipt(kind="file.open", executor="desktop", ok=False, detail=f"not found or outside the allowed folders: {path}",
                           verifications=[Verification(check="path exists in workspace/home", passed=False, observed=str(target))])
        try:
            os.startfile(str(target))  # type: ignore[attr-defined]
            ok = True
        except Exception as exc:  # noqa: BLE001
            return Receipt(kind="file.open", executor="desktop", ok=False, detail=str(exc)[:200],
                           verifications=[Verification(check="default program launched", passed=False, observed=str(exc)[:120])])
        return Receipt(kind="file.open", executor="desktop", ok=ok, detail=f"opened {target.name}",
                       verifications=[Verification(check="handed to the default program", passed=True, observed=str(target))], evidence={"path": str(target)})

    @property
    def backups(self) -> Any:
        from runtime.backup import BackupManager

        if getattr(self, "_backups", None) is None:
            self._backups = BackupManager(Path(self.kernel.state_root), repository=self.selfdev_repository())
        return self._backups

    def backup_create(self, label: str = "") -> dict[str, Any]:
        report = self.backups.create(label=label)
        self.emit(EventType.TOOL, {"summary": f"backup: {report['files']} files, {report['bytes']} bytes, verified={report['verified']}", "source": "backup"})
        return report

    def experience_view(self, goal: str = "") -> dict[str, Any]:
        from development.experience import ExperienceStore

        store = ExperienceStore(Path(self.kernel.state_root) / "experience" / "selfdev.jsonl")
        rows = store.list()
        return {"count": len(rows), "entries": [r.to_dict() for r in rows[-30:]],
                "relevant": [r.to_dict() for r in store.relevant(goal)] if goal else [],
                "compare": store.compare(goal) if goal else {}}

    def compose_preview(self, goal: str) -> dict[str, Any]:
        """The plan the composer would make, without running it."""

        from brain.tiers import ModelTier

        plan = self._composer().plan(goal, self.kernel.provider(ModelTier.FAST_LOCAL))
        return {"ok": True, "plan": plan.to_dict(), "executable": plan.executable, "menu": [p.name for p in self._composer().primitives.values()]}

    def device_context(self) -> dict[str, Any]:
        """Where ZEUS is running right now: the foundation for "show it here"."""

        from runtime.device_context import current_context

        return current_context(self).to_dict()

    def _capability_payload(self, manifest: Any, goal: str, text: str) -> tuple[dict[str, Any], list[str]]:
        """The payload a capability's input schema asks for, from the request.

        Not ``{"goal": goal}``: no learned capability declares a ``goal`` key,
        so that payload failed every one of them at the first line.  Path-like
        required inputs come from the file name in the request, resolved by
        the one workspace path model; other required inputs that the request
        cannot supply are reported back by name instead of being guessed.
        """

        from runtime.paths import PathError, resolve_workspace_path
        from service.intent import FILENAME

        schema = dict(getattr(manifest, "input_schema", {}) or {})
        properties = dict(schema.get("properties") or {})
        required = [str(r) for r in (schema.get("required") or [])]
        payload: dict[str, Any] = {}
        unmet: list[str] = []
        workspace = Path(self.kernel.state_root) / "workspace"
        names = [m.group(0) for m in FILENAME.finditer(text)] if hasattr(FILENAME, "finditer") else []
        for key in properties:
            lowered = key.lower()
            if "path" in lowered or lowered in {"file", "source", "folder", "directory", "filename", "file_name"}:
                if names:
                    try:
                        payload[key] = str(resolve_workspace_path(workspace, names[0], must_exist=True))
                    except PathError as exc:
                        payload[key] = str(exc)
                        unmet.append(f"{key} ({exc})")
                elif key in required:
                    unmet.append(key)
            elif lowered in {"goal", "request", "text", "query", "prompt", "input"}:
                payload[key] = goal
            elif key in required:
                unmet.append(key)
        if not properties:
            payload = {"goal": goal}
        return payload, unmet

    def _answer_by_capability(self, text: str, scope: str, plan: Any) -> None:
        """Serve a real-world request from a capability, acquiring one if needed.

        The generic form of the music path.  A request that no built-in action
        covers is not automatically a refusal: it may be something ZEUS already
        learned to do, or something it can go and learn.  What it is never
        allowed to be is a description of the thing happening.
        """

        from runtime.receipts import Receipt, Verification, failed

        goal = str(plan.arguments.get("goal") or plan.reason or text).strip()
        self.emit(EventType.TOOL, {"summary": f"capability requested: {goal[:120]}"}, scope=scope)

        manifest = None
        try:
            manifest = self.capabilities.resolve(goal)
        except Exception as exc:
            self.emit(EventType.ERROR, {"error": f"registry unreadable: {exc}"}, scope=scope)

        if manifest is None:
            receipt = failed(
                "capability.missing", "capability.resolver",
                f"I have no verified capability for that yet: {goal[:160]}",
                request=text, goal=goal,
            )
            self.receipts.record(receipt)
            self._session_receipts.append(receipt)
            self.emit(
                EventType.TOOL,
                {"summary": receipt.summary(), "receipt_id": receipt.id,
                 "receipt": receipt.to_dict()},
                scope=scope,
            )
            self._deliver(
                f"{receipt.detail}\n\nreceipt {receipt.id}",
                scope=scope, backend="capability.resolver",
                context_text=f"[no capability for: {goal[:80]}]",
                final_state=JarvisState.ERROR,
            )
            return

        capability_id = str(manifest.capability_id)
        self.state.set(JarvisState.VERIFYING, detail=capability_id, scope=scope)
        payload, unmet = self._capability_payload(manifest, goal, text)
        if unmet:
            receipt = failed(f"capability.{capability_id}", capability_id,
                             f"{capability_id} needs {', '.join(unmet)} and the request does not say which", request=text, goal=goal)
            self.receipts.record(receipt)
            self._session_receipts.append(receipt)
            self.emit(EventType.TOOL, {"summary": receipt.summary(), "receipt_id": receipt.id, "receipt": receipt.to_dict()}, scope=scope)
            # the raw contract line ("… needs file_path and the request does
            # not say which") belongs in Activity, not in the conversation
            de = self.language.startswith("de")
            slots = ", ".join(unmet)
            self._deliver((f"Dafür fehlt mir noch eine Angabe ({slots}). Sag sie mir, dann führe ich es aus." if de
                           else f"One detail is missing for that ({slots}). Tell me and I will run it."),
                          scope=scope, backend=capability_id,
                          context_text=f"[capability {capability_id}: missing input {unmet}; receipt {receipt.id}]",
                          final_state=JarvisState.WAITING)
            return
        try:
            execution = self.capabilities.execute(capability_id, payload)
        except Exception as exc:
            execution = None
            self.emit(EventType.ERROR, {"error": f"{type(exc).__name__}: {exc}"}, scope=scope)

        ok = bool(getattr(execution, "ok", False))
        output = dict(getattr(execution, "output", {}) or {})
        # A capability's own word is not the verdict. What it produced is
        # checked here, from outside, exactly as the acquisition gates do.
        checks = self._verify_capability_output(output)
        receipt = Receipt(
            kind=f"capability.{capability_id}",
            executor=capability_id,
            ok=ok and all(item.passed for item in checks),
            request=text,
            detail=(str(output.get("detail") or output.get("message") or "ran")
                    if ok else str(getattr(execution, "error", "") or output.get("error", "failed"))),
            evidence={"goal": goal, "capability": capability_id,
                      "output": {k: v for k, v in output.items() if k != "client_secret"}},
            verifications=tuple(checks),
        )
        self.receipts.record(receipt)
        self._session_receipts.append(receipt)
        self.emit(
            EventType.TOOL,
            {"summary": receipt.summary(), "receipt_id": receipt.id, "receipt": receipt.to_dict()},
            scope=scope,
        )
        lines = [receipt.detail]
        if receipt.verifications:
            lines += ["", "Belege:" if self.language.startswith("de") else "Evidence:"]
            lines += [f"  - {line}" for line in receipt.evidence_lines()]
        lines += ["", f"receipt {receipt.id}"]
        self._deliver(
            "\n".join(lines), scope=scope, backend=capability_id,
            context_text=f"[capability {capability_id}: "
            f"{'verified' if receipt.verified else 'not verified'}, receipt {receipt.id}]",
            final_state=JarvisState.IDLE if receipt.verified else JarvisState.ERROR,
        )

    @staticmethod
    def _verify_capability_output(output: dict[str, Any]) -> list[Any]:
        """Check whatever the capability says it produced, from outside.

        Deliberately domain-blind: this knows nothing about screenshots or
        exports. It knows the shape of the claim -- "there is a file at this
        path" -- and goes and looks, because a capability that reports a path
        it did not write is the file-write defect wearing different clothes.
        """

        import time

        from runtime.receipts import Verification

        checks: list[Verification] = []
        raw = output.get("path") or output.get("file") or output.get("artifact")
        if not raw:
            return checks
        target = Path(str(raw))
        exists = target.is_file()
        checks.append(
            Verification(
                check="the reported file exists",
                passed=exists,
                observed=f"{target} ({target.stat().st_size} bytes)" if exists else f"{target} is not a file",
                expected=str(target),
            )
        )
        if not exists:
            return checks
        age = time.time() - target.stat().st_mtime
        checks.append(
            Verification(
                check="it was produced just now, not found",
                passed=age <= 300.0,
                observed=f"{age:.0f}s old",
                expected="under 300s old",
            )
        )
        return checks

    def _defect_confirmed(self, outcome: Any) -> bool:
        """Whether a failure has happened often enough to be the code's fault.

        One failure is an incident; two in a row is a defect.  Retiring a
        capability that has passed every gate, on the strength of a single bad
        call, disables working functionality and spends half an hour rebuilding
        something that may be perfectly correct -- which is exactly what
        happened to a verified Spotify provider when one cold call ran over its
        timeout.

        Counted per capability and reset by any success, so a provider that
        works intermittently is left alone and one that has genuinely broken is
        repaired on its second consecutive failure.
        """

        capability = outcome.capability_id or "unknown"
        self._defects[capability] = self._defects.get(capability, 0) + 1
        if self._defects[capability] < 2:
            self.emit(
                EventType.TOOL,
                {"summary": f"{capability} failed once ({outcome.defect[:120]}); "
                            "waiting for a second failure before rebuilding it"},
            )
            return False
        return True

    def _acquire_then_retry(self, request: Any, scope: str, *, repair: str = "") -> Any:
        """Build (or fix) the provider, then answer the request that needed it."""

        from runtime.receipts import failed
        from service.acquisition import AcquisitionMission
        from service.music import (
            MusicOutcome,
            provider_acceptance,
            provider_constraints,
            provider_extra_checks,
            provider_goal,
            provider_keywords,
        )

        provider = self.music.provider
        if not self._acquiring.acquire(blocking=False):
            return MusicOutcome(
                receipt=failed(
                    f"music.{request.action}", "music.resolver",
                    "a capability for this is already being built; ask again in a moment",
                    request=request.query,
                ),
                gap=True,
            )
        try:
            # A disabled version is still a version: rebuilding it starts from
            # its installed source, so calling that "building one now" would
            # misdescribe both what exists and what is about to happen.
            retired = self.music.retired_capability()
            if not repair and retired is not None:
                repair = str(
                    (getattr(retired, "validation_status", {}) or {}).get("disabled_reason", "")
                ) or "a previously registered version was withdrawn"
            verb = "repairing" if repair else "building"
            self.state.set(JarvisState.CODING, detail=f"{verb} the {provider} provider", scope=scope)
            self.emit(
                EventType.NOTIFICATION,
                {"text": (
                    f"The {provider} capability is broken: {repair[:200]} Repairing it now."
                    if repair else
                    f"I have no verified {provider} capability yet. Building one now."
                )},
                scope=scope,
            )
            mission = AcquisitionMission(
                service=self.capabilities,
                kernel=self.kernel,
                emit=lambda kind, payload: self.emit(kind, payload, scope=scope),
            )
            result = mission.run(
                provider_goal(provider),
                capability_id=f"music.provider.{provider}",
                keywords=provider_keywords(provider),
                # The same playback gate on both paths: a local build and an
                # expert's build are held to one bar, and neither is judged by
                # checks a capability that does nothing could pass.
                extra_checks=provider_extra_checks(),
                expert_constraints=provider_constraints(provider),
                expert_acceptance=provider_acceptance(),
                max_steps=60,
                max_seconds=1800.0,
                repair=repair,
            )
            self.emit(
                EventType.PROGRESS,
                {"summary": f"acquisition finished: {'acquired' if result.acquired else 'failed'}",
                 "acquisition": result.to_dict()},
                scope=scope,
            )
            if not result.acquired:
                return MusicOutcome(
                    receipt=failed(
                        f"music.{request.action}", "capability.acquisition",
                        f"I could not {'repair' if repair else 'build'} a verified {provider} "
                        f"capability: {result.reason}",
                        request=request.query, acquisition=result.to_dict(),
                    ),
                    gap=True,
                )
            # Built and verified: answer the question that started this. The
            # retry's own defect field is discarded -- one repair per turn, so a
            # fix that does not hold surfaces as a failure rather than a loop.
            self.state.set(JarvisState.WORKING, detail=f"music: {request.action}", scope=scope)
            retried = self.music.run(request)
            retried.defect = ""
            return retried
        finally:
            self._acquiring.release()

    def _answer_by_executing(self, text: str, scope: str, classification: Any, *, action_request: bool = False) -> None:
        """A request with a side effect.  Nothing is said until something is done.

        No model output reaches the user on this path.  The model is asked for
        one thing -- a machine-readable plan -- and the sentence the user reads
        is composed from the receipt by :func:`service.actions.compose`.  There
        is no step here at which a model could assert that something worked.
        """

        from brain.tiers import ModelTier
        from service.actions import compose

        if self._hold_for_gpu(text, scope):
            return
        self.state.set(JarvisState.WORKING, detail=classification.reason[:120], scope=scope)
        # Owner corrections are retrieved BEFORE the model interprets the
        # request, and their overrides are applied AFTER it: the owner's word
        # outranks the model's guess within the correction's scope.
        from service.corrections import apply_overrides, guidance_lines

        relevant = self.corrections.relevant(text, intent=classification.intent.value)

        # Composition before development: a goal that is several things at
        # once is planned over the primitives ZEUS already has, and only a
        # primitive it genuinely lacks becomes an acquisition.
        from service.composer import looks_compound

        # Compound goals always; a single-step goal too when a registered
        # capability's own keywords match it -- "zähle die Wörter in
        # plan.txt" is the capability learned an hour ago, not file.read.
        try:
            hits = self.capabilities.registry.find(text, limit=1)
        except Exception:  # noqa: BLE001
            hits = []
        if looks_compound(text) or hits:
            try:
                if self._answer_by_composition(text, scope, guidance="\n".join(guidance_lines(relevant)), allow_single=bool(hits)):
                    return
            except Exception as exc:  # noqa: BLE001 - composition is an attempt; the single-action path remains
                self.emit(EventType.DIAGNOSTIC, {"composition": f"failed: {type(exc).__name__}: {exc}"}, scope=scope)
        # The semantic control plane: FAST_LOCAL reads the goal behind the
        # words and picks ONE tool from a closed set.  This replaces lexical
        # guessing as the primary intelligence — the legacy planner below
        # remains the fallback for file writing and everything "delegate".
        try:
            goal = self._semantic_goal(text, scope, guidance="\n".join(guidance_lines(relevant)))
        except Exception as exc:  # noqa: BLE001 - no semantics, the legacy path remains
            goal = None
            self.emit(EventType.DIAGNOSTIC, {"semantic": f"failed: {type(exc).__name__}: {exc}"}, scope=scope)
        if goal is not None and self._dispatch_semantic_goal(goal, text, scope, classification):
            return
        try:
            provider = self.kernel.provider(ModelTier.FAST_LOCAL)
            plan = self.actions.plan(text, provider, guidance="\n".join(guidance_lines(relevant)))
            if relevant and not plan.declined:
                plan.arguments, applied = apply_overrides(plan.arguments, relevant, action=plan.action)
                if applied:
                    self.corrections.note_applied(relevant)
                    self.emit(EventType.TOOL, {"summary": f"owner corrections applied: {', '.join(applied)}",
                                               "corrections": [c.correction_id for c in relevant]}, scope=scope)
        except Exception as exc:
            self.state.set(JarvisState.ERROR, detail=str(exc)[:200])
            self.emit(EventType.ERROR, {"error": f"{type(exc).__name__}: {exc}"}, scope=scope)
            return

        if plan.declined:
            # The classifier is biased toward ACTION on purpose, so it over-
            # triggers; the planner is the second opinion. "Schreibe mir ein
            # Gedicht" lands here, and the honest thing is to have the
            # conversation, not to refuse it. The claim guard still covers it.
            self.emit(
                EventType.TOOL,
                {"summary": f"no executable action: {plan.reason[:160]}", "declined": True},
                scope=scope,
            )
            from service.intent import ACTION_OBJECTS, FILENAME

            names_object = any(word in f" {text.lower()} " for word in ACTION_OBJECTS) or bool(FILENAME.search(text))
            de = self.language.startswith("de")
            if classification.matched and names_object:
                # The request names a side effect and the planner could not
                # turn it into one for want of a detail. Handing that to the
                # conversation model produced an invented "notes database"
                # with a fake commit id; one concise question is the honest
                # answer, and the receipt path resumes when it is answered.
                self._deliver(
                    (f"Das kann ich ausführen, aber ein Detail fehlt: {plan.reason[:160]}. "
                     f"Sag mir zum Beispiel den Dateinamen, dann mache ich es.") if de else
                    (f"I can do that, but a detail is missing: {plan.reason[:160]}. "
                     f"Give me the file name, for example, and I will do it."),
                    scope=scope, backend="planner", final_state=JarvisState.WAITING,
                )
                return
            from service.intents import is_action_request

            creative = re.search(r"\b(gedicht|geschichte|witz|poem|story|joke|text|zusammenfassung|summary|erklaer|erklär|explain|beschreib|describe|liste\s+mir|nenn)\w*", text.lower())
            if (action_request or is_action_request(text)) and not creative:
                # An action request never degrades into advisory prose: it is
                # executed, becomes a mission, asks for what is missing, or
                # says plainly why it cannot be done.  This is the last branch.
                self._deliver(
                    (f"Das kann ich so nicht ausführen: {plan.reason[:160] or 'keine passende Aktion'}. "
                     f"Sag mir genauer, was entstehen soll (Projekt, Datei, Notiz, Knowledge-Eintrag, Musik …), dann mache ich es.") if de else
                    (f"I cannot execute that as asked: {plan.reason[:160] or 'no matching action'}. "
                     f"Tell me more precisely what should exist afterwards (project, file, note, Knowledge entry, music …) and I will do it."),
                    scope=scope, backend="planner", final_state=JarvisState.WAITING,
                    context_text="[action request: no executable action; asked for the missing detail]",
                )
                return
            self._answer_conversationally(text, scope)
            return

        if plan.action == "project.create":
            # The model extracted a project; the typed executor verifies the contract.
            from service.intents import ActionIntent

            args = dict(plan.arguments)
            title = str(args.get("name") or args.get("title") or args.get("goal") or "").strip()
            intent = ActionIntent("project.create", verb="create", object_type="project", target=title,
                                  arguments={"title": title, "goal": str(args.get("goal") or title), "tasks": list(args.get("tasks") or []),
                                             "parent": "", "importance": "", "deadline": "", "description": text},
                                  success_criteria=["the project persists with this title"], confidence=0.7, reason="planner: project.create")
            if not title:
                from service.intents import clarification_for

                intent.missing = ["title"]
                self._ask_clarification(intent, clarification_for(intent, language=self.language), text, scope)
                return
            self._answer_by_project_operation(intent, text, scope)
            return

        if plan.action == "capability":
            # A real-world action outside the built-in set. Either something
            # registered already serves it, or it becomes a mission -- the same
            # shape as a music gap, without the music.
            self._answer_by_capability(text, scope, plan)
            return

        self.emit(
            EventType.TOOL,
            {"summary": f"executing {plan.action}", "action": plan.to_dict()},
            scope=scope,
        )
        receipt = self.actions.execute(plan, request=text)

        self.state.set(JarvisState.VERIFYING, detail=receipt.kind, scope=scope)
        self.receipts.record(receipt)
        self._session_receipts.append(receipt)
        self.emit(
            EventType.TOOL,
            {"summary": receipt.summary(), "receipt_id": receipt.id, "receipt": receipt.to_dict()},
            scope=scope,
        )
        self._deliver(
            compose(receipt, language=self.language),
            scope=scope,
            backend=receipt.executor,
            # An action that did not verify leaves the eye red rather than
            # returning to idle. Going straight back to idle made a failure
            # look exactly like a success from across the room, which is the
            # same defect as a reassuring sentence -- only in the interface.
            # It clears on the next turn, like every other state.
            final_state=JarvisState.IDLE if receipt.verified else JarvisState.ERROR,
            # The transcript gets the fact, not the evidence. The user needs
            # every check; the next prompt needs one line that cannot be
            # mistaken for the model's own prose.
            context_text=f"[executed {receipt.kind}: "
            f"{'verified' if receipt.verified else 'not verified'}, receipt {receipt.id}]",
        )

    @staticmethod
    def _planner_wants_detail(reason: str) -> bool:
        """Whether a decline says "missing detail" rather than "not an action"."""

        lowered = (reason or "").lower()
        return any(word in lowered for word in (
            "path", "filename", "file name", "name", "not specified", "no file", "missing", "unspecified",
            "did not", "didn't", "doesn't specify", "does not specify", "without", "unclear which", "pfad", "dateiname",
        ))

    def _small_talk(self, text: str) -> str | None:
        """A natural answer to a phatic question, from what Zeus actually knows about its state.

        Ordinary social questions ("Wie geht es dir?") are not requests for a
        self-description.  The answer is Zeus's own: short, warm, and true
        (state from the lifecycle, missions from the store).  Anything that
        asks *literally* about feelings or consciousness, or what Zeus is
        technically, is not small talk and goes to the model with the
        personality prompt.
        """

        from persona.smalltalk import identity_answer, small_talk_answer

        who = identity_answer(text, language=self.language or "de", assistant=self.identity.assistant_name)
        if who:
            return who
        try:
            missions = self.list_missions(status="active")["count"]
        except Exception:  # noqa: BLE001
            missions = 0
        try:
            uptime = float(self.lifecycle.health().get("uptime_seconds", 0) or 0)
        except Exception:  # noqa: BLE001
            uptime = 0.0
        try:
            prefs = __import__("owner.core", fromlist=["current"]).current().read("personality").get("preferences", {})
        except Exception:  # noqa: BLE001
            prefs = {}
        return small_talk_answer(text, language=self.language or "de", active_missions=int(missions), uptime_seconds=uptime,
                                 humour=int(prefs.get("humour", 40) or 0), warmth=int(prefs.get("warmth", 50) or 0))

    def _answer_conversationally(self, text: str, scope: str) -> None:
        from brain.tiers import ModelTier

        if self._hold_for_gpu(text, scope):
            return
        quick = self._small_talk(text)
        if quick:
            self.state.set(JarvisState.THINKING, detail=text[:120], scope=scope)
            # Zeus's own words, spoken like any other answer.
            if self._voice is not None and self._voice.settings.enabled and self._voice.settings.speak_replies:
                try:
                    self.voice.speak_stream([quick], scope=scope)
                except Exception as exc:  # noqa: BLE001 - the text still arrives
                    self.emit(EventType.DIAGNOSTIC, {"speech": f"small talk not spoken: {exc}"})
            # Clients read answers as a token stream; a short one is one token.
            self.emit(EventType.TOKEN, {"text": quick, "backend": "personality"}, scope=scope)
            self._deliver(quick, scope=scope, backend="personality", final_state=JarvisState.IDLE)
            return
        self.state.set(JarvisState.THINKING, detail=text[:120], scope=scope)
        collected: list[str] = []
        #: Populated by the guard inside ``tee`` when the model claims a side
        #: effect that nothing performed.  A list because a closure cannot
        #: rebind a local.
        fabricated: list[Any] = []
        #: Receipts that back a claim the model made in passing, so the answer
        #: can carry its own evidence rather than asking to be believed.
        supported: list[Any] = []
        backend = ""
        try:
            tier = ModelTier.FAST_LOCAL
            provider = self.kernel.provider(tier)
            backend = getattr(self.kernel.catalog.get(tier), "model", "") or tier.value
            # The conversation prompt is the *system* message: the owner's
            # identity and personality documents, in their fixed order, and
            # nothing else -- the provider's default engineering preamble
            # ("Your job is to: 1. Understand the user's goal ...") stays out
            # of ordinary conversation, where it read as a robot's job sheet.
            system, user = self._compose_messages(text)
            stream = self._generate(provider, user, system=system)

            def tee():
                """One pass over the model's output feeds both the screen and the voice.

                Generating twice -- once to display, once to speak -- would
                double the cost and let the two drift apart, which the user
                would hear as speech that does not match the text on screen.
                """

                from runtime.receipts import supporting
                from service.claims import find_claim

                for chunk in stream:
                    if self._stop_requested.is_set():
                        return
                    collected.append(chunk)
                    # Checked as each chunk arrives rather than at the end.  A
                    # false claim that has already been printed and spoken has
                    # been believed; replacing it afterwards corrects the record
                    # but not the impression.  The chunk that completes the
                    # claim is withheld, so the user never sees the finished
                    # sentence -- at most "Datei wurde", never "wurde erstellt".
                    joined = "".join(collected)
                    claim = find_claim(joined)
                    if claim is not None:
                        # A claim about something that really was executed is
                        # not a fabrication. Checked against the session's
                        # receipts in memory, so this costs a scan of a short
                        # list rather than a file read per token.
                        backing = supporting(self._session_receipts, joined)
                        if backing is None:
                            fabricated.append(claim)
                            return
                        if backing not in supported:
                            supported.append(backing)
                    self.emit(EventType.TOKEN, {"text": chunk}, scope=scope)
                    yield chunk

            # Speak only in voice mode. Merely having constructed the speech
            # engine is not consent to talk: a user who dictated once should
            # not find every later typed message read aloud.
            speak = (
                self._voice is not None
                and self._voice.settings.enabled
                and self._voice.settings.speak_replies
            )
            if speak:
                self.voice.speak_stream(tee(), scope=scope)
            else:
                for _ in tee():
                    pass
        except Exception as exc:
            # bounded agentic recovery (§ the live "ProviderError timed out"):
            # one retry with a REDUCED prompt before admitting the failure —
            # and the admission is a human sentence, never a stack line
            self.emit(EventType.ERROR, {"error": f"{type(exc).__name__}: {exc}"}, scope=scope)
            de = self.language.startswith("de")
            try:
                from brain.providers import ProviderError

                recoverable = isinstance(exc, ProviderError)
            except Exception:  # noqa: BLE001
                recoverable = False
            if recoverable and not collected:
                try:
                    self.state.set(JarvisState.THINKING, detail="zweiter Versuch (kürzerer Kontext)", scope=scope)
                    time.sleep(1.5)
                    provider = self.kernel.provider(ModelTier.FAST_LOCAL)
                    short = provider.generate((f"Antworte kurz und sachlich auf Deutsch: {text}" if de
                                               else f"Answer briefly: {text}"), max_tokens=300, temperature=0.4)
                    if str(short).strip():
                        self._deliver(str(short).strip(), scope=scope, backend="fast_local",
                                      context_text="[recovered with a reduced prompt after a provider error]")
                        return
                except Exception as retry_exc:  # noqa: BLE001
                    self.emit(EventType.DIAGNOSTIC, {"conversation_retry": f"{type(retry_exc).__name__}: {retry_exc}"}, scope=scope)
            self._deliver(("Die lokale KI hat gerade nicht rechtzeitig geantwortet. Frag mich bitte gleich nochmal – ich bin dran, sie wiederherzustellen." if de
                           else "The local model did not answer in time. Please ask again in a moment – I am restoring it."),
                          scope=scope, backend="recovery", final_state=JarvisState.ERROR,
                          context_text=f"[conversation failed: {type(exc).__name__}: {str(exc)[:160]}]")
            return

        answer = "".join(collected).strip()
        context_text = ""
        if fabricated:
            answer = self._block_fabrication(fabricated[0], text, scope)
            backend = "claim-guard"
            # Kept out of the transcript entirely. Feeding the correction back
            # in would put the very vocabulary the guard watches for into the
            # model's context, which is how one interception becomes a run of
            # them.
            context_text = "[the previous answer was withheld: unsupported success claim]"
        elif supported:
            # The assistant referred to something it really did. Name the
            # receipt so the user can check rather than take its word.
            answer += "\n\n" + "\n".join(f"receipt {item.id}" for item in supported)
        # The reply carries the request it answered, so a 👍/👎 under it can
        # attach to the right exchange and the right learned scope.
        reply_meta: dict[str, Any] = {}
        try:
            last_user = self._last_user_meta()
            if last_user.get("request_id"):
                reply_meta["request_id"] = last_user["request_id"]
        except Exception:  # noqa: BLE001
            pass
        self._deliver(answer, scope=scope, backend=backend, context_text=context_text, meta=reply_meta)
        self._say_pending_thought(scope)

    def _block_fabrication(self, claim: Any, request: str, scope: str) -> str:
        """Refuse to ship a success the system cannot account for.

        Recorded as a receipt rather than merely logged, because "it tried to
        tell you it had done something" is exactly the kind of durable fact the
        receipt ledger exists to hold -- and because a user who was told a
        falsehood should be able to find the interception afterwards.
        """

        from runtime.receipts import Receipt, Verification
        from service.claims import correction

        self.state.set(JarvisState.VERIFYING, detail="checking a success claim", scope=scope)
        receipt = self.receipts.record(
            Receipt(
                kind="claim.blocked",
                executor="service.claims",
                ok=False,
                request=request,
                detail=f"the model claimed a completed side effect with no receipt: {claim.phrase!r}",
                evidence={"claim": claim.to_dict()},
                verifications=(
                    Verification(
                        check="a verified receipt supports this claim",
                        passed=False,
                        observed=(
                            f"nothing in this conversation's {len(self._session_receipts)} "
                            "receipt(s) matches what the claim names"
                        ),
                        expected="a verified receipt naming the same file or project",
                    ),
                ),
            )
        )
        self.emit(
            EventType.TOOL,
            {"summary": receipt.summary(), "receipt_id": receipt.id, "receipt": receipt.to_dict()},
            scope=scope,
        )
        # Stop the voice too. Correcting the text while the speaker finishes
        # saying the false sentence would fix the transcript and not the lie.
        if self._voice is not None:
            try:
                self._voice.interrupt()
            except Exception:
                pass
        return correction(claim, language=self.language)

    def _deliver(
        self,
        answer: str,
        *,
        scope: str,
        backend: str,
        context_text: str = "",
        final_state: JarvisState = JarvisState.IDLE,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Record the assistant's turn, publish it, and settle into a state."""

        reply = ConversationTurn(
            role="assistant", text=answer, at=_now(), backend=backend, context_text=context_text, meta=dict(meta or {})
        )
        with self._lock:
            self._history.append(reply)
        self.emit(EventType.MESSAGE, reply.to_dict(), scope=scope)
        self.state.set(final_state, detail="" if final_state is JarvisState.IDLE else answer[:120])

    def _generate(self, provider: Any, prompt: str, *, system: str | None = None) -> Iterable[str]:
        """Stream from a provider, falling back to a single block if it cannot.

        ``system`` is passed to providers that accept one; a provider that
        does not gets it folded into the prompt, so the personality reaches
        every backend either way.
        """

        stream = getattr(provider, "generate_stream", None)
        if callable(stream):
            if system:
                try:
                    yield from stream(prompt, system=system)
                    return
                except TypeError:
                    yield from stream(f"{system}\n\n{prompt}")
                    return
            yield from stream(prompt)
            return
        if system:
            try:
                yield provider.generate(prompt, system=system)
                return
            except TypeError:
                yield provider.generate(f"{system}\n\n{prompt}")
                return
        yield provider.generate(prompt)

    def _compose_messages(self, text: str) -> tuple[str, str]:
        """(system, user): the personality/identity block and the transcript + the owner's words."""

        full = self._compose_prompt(text)
        marker = "\n\nRecent conversation:\n"
        if marker in full:
            head, tail = full.split(marker, 1)
            return head, "Recent conversation:\n" + tail
        marker = f"\n\nuser: "
        if marker in full:
            head, tail = full.rsplit(marker, 1)
            return head, "user: " + tail
        return "", full

    def _compose_prompt(self, text: str) -> str:
        """Wrap the user's words in the persona and recent context.

        The persona is stated as identity rather than as a costume instruction
        ("you are roleplaying as..."), because the latter invites a model to
        break character and explain what it really is the moment it is asked.

        The text comes from the persona store rather than being written here, so
        that "add another persistent personality" is a stored record instead of
        an edit to this method -- and so the invariant rules (never claim an
        unverified success, prefer admitting ignorance) are appended after the
        persona's own words, where a verbose character cannot crowd them out.
        """

        # Task style from the active persona: style and extra instructions
        # only.  A persona's character text no longer reaches the
        # conversation -- identity and character are the owner's documents.
        task_style = ""
        try:
            persona = self.personas.active()
            task_style = "; ".join(s for s in (getattr(persona, "style", ""), getattr(persona, "extra_instructions", "")) if s)
        except Exception:
            task_style = ""
        guidance: list[str] = []
        try:
            from service.corrections import guidance_lines

            guidance = guidance_lines(self.corrections.relevant(text, intent="conversation"))
        except Exception:
            guidance = []
        # What the owner's feedback has taught, scoped to this kind of turn:
        # owner-authored rules first, then confident learned nudges.
        try:
            from runtime.adaptation import classify_context

            guidance += self.adaptation.guidance(classify_context(request=text))
        except Exception:  # noqa: BLE001 - adaptation never blocks an answer
            pass
        # Capability awareness (§ the live "I cannot access D:" failure): the
        # conversation model must know what THIS system can actually do, and
        # must not confabulate facts about unfamiliar words.
        guidance.append(
            "Du läufst als ZEUS mit echten Werkzeugen auf diesem Windows-PC: Dateisystem "
            "(Ordner zählen/suchen/öffnen), Apps starten, Websites öffnen, Websuche, Spotify/Musik, "
            "Kalender, Projekte, Wissen, PDF-Zusammenfassungen, Bildgenerierung, Screenshots, SelfDev. "
            "Behaupte NIEMALS, du hättest keinen Zugriff auf Dateien oder dieses System — wenn dafür "
            "eine Aktion nötig ist, sage kurz, dass du sie ausführen kannst. "
            "Bei Faktenfragen zu seltenen oder unbekannten Begriffen: NICHT raten und keine Definition "
            "erfinden — sag ehrlich, dass du unsicher bist, und biete an, im Internet nachzusehen.")
        recent = self.history[-8:]
        transcript = "\n".join(f"{turn.role}: {turn.for_prompt()}" for turn in recent[:-1])
        from config import conversation_prompt

        try:
            return conversation_prompt(language=self.language, guidance=guidance, task_style=task_style, transcript=transcript,
                                       text=text, assistant=self.identity.assistant_name, identity=self.identity)
        except Exception:
            # A broken owner document must not silence Zeus.
            base = self.identity.persona_preamble()
            return base + "\n\n" + (f"Recent conversation:\n{transcript}\n\n" if transcript else "") + f"user: {text}\n{self.identity.assistant_name}:"

    # ------------------------------------------------------------------
    # Warming
    # ------------------------------------------------------------------

    def warm(self, *, speech: bool = True) -> threading.Thread:
        """Load the models the first interaction will need, in the background.

        Measured before this existed: the first spoken exchange after startup
        took 54 seconds, nearly all of it loading a 4B model and a whisper model
        that were always going to be needed. The user experienced that as Jarvis
        being broken, then answering.

        Warming is fire-and-forget and every step is individually guarded: a
        machine with no speech venv, or with Ollama not yet up, must still get a
        working text Jarvis rather than an exception during startup.
        """

        def run() -> None:
            try:
                # Before the first diagnostic, so startup itself is on the record.
                self.activity
            except Exception:
                pass
            self.emit(EventType.DIAGNOSTIC, {"warming": "started"})
            # Bounded retries, because the first attempt races the machine:
            # a freshly started Ollama loads the 4B model on the first
            # request (71 s measured cold), and one failed probe used to
            # mark FAST_LOCAL unavailable for ever -- the supervisor then
            # read a healthy revision as a broken one.
            from brain.tiers import ModelTier

            for attempt in range(4):
                try:
                    provider = self.kernel.provider(ModelTier.FAST_LOCAL)
                    answer = provider.generate("Reply with the single word: OK", max_tokens=4, temperature=0.0)
                    text = answer if isinstance(answer, str) else "".join(str(piece) for piece in answer)
                    if not text.strip():
                        raise RuntimeError("the model returned an empty answer")
                    self.emit(EventType.DIAGNOSTIC, {"warming": "conversation model ready" + (f" (attempt {attempt + 1})" if attempt else "")})
                    # READY is earned here and nowhere else: real text came out of
                    # the model that answers the user, in this process.
                    self.lifecycle.mark("fast_local", True, text.strip()[:40])
                    self._health_ok, self._health_checked_at = True, time.time()
                    break
                except Exception as exc:
                    self.emit(EventType.DIAGNOSTIC, {"warming": f"conversation model unavailable (attempt {attempt + 1}/4): {exc}"})
                    self.lifecycle.mark("fast_local", False, f"{type(exc).__name__}: {exc}"[:300])
                    if attempt < 3:
                        time.sleep(20.0)

            # A restart that follows a promotion must resume whatever was in
            # flight; this is where interrupted missions get their chance.
            try:
                restored = self._restore_missions()
                self.lifecycle.mark("missions", True, f"{restored} resumable")
            except Exception as exc:
                self.lifecycle.mark("missions", False, str(exc)[:200])
            try:
                # A self-development mission that asked for this restart gets
                # its verdict from the supervisor's receipt, into the transcript.
                self._settle_selfdev_after_restart()
            except Exception as exc:
                self.emit(EventType.DIAGNOSTIC, {"warming": f"selfdev settlement failed: {exc}"})

            if speech:
                try:
                    # Synthesising one short phrase loads the voice; discarding
                    # it is the cheapest way to pay that cost early.
                    self.voice.engine.synthesize("Bereit.")
                    self.emit(EventType.DIAGNOSTIC, {"warming": "voice ready"})
                    self.lifecycle.mark("voice", True)
                except Exception as exc:
                    self.emit(EventType.DIAGNOSTIC, {"warming": f"speech unavailable: {exc}"})
                    self.lifecycle.mark("voice", False, str(exc)[:200])

                try:
                    # And the recogniser, which is the larger of the two costs:
                    # loading whisper-base took ~50 s of the 54 s first
                    # exchange. Half a second of silence is enough to force it.
                    from speech.contracts import Audio

                    silence = Audio(samples=bytes(32000), sample_rate=16000)
                    self.voice.engine.transcribe(silence)
                    self.emit(EventType.DIAGNOSTIC, {"warming": "recogniser ready"})
                    self.lifecycle.mark("recogniser", True)
                except Exception as exc:
                    self.emit(EventType.DIAGNOSTIC, {"warming": f"recogniser unavailable: {exc}"})
                    self.lifecycle.mark("recogniser", False, str(exc)[:200])

            self.emit(EventType.DIAGNOSTIC, {"warming": "done"})
            self.lifecycle.mark("warm", True)
            # Proactive thinking: cheap detectors on an idle timer (no model).
            try:
                self._schedule_idle_thinking()
            except Exception:  # noqa: BLE001
                pass

        thread = threading.Thread(target=run, daemon=True, name="jarvis-warm")
        thread.start()
        return thread

    # ------------------------------------------------------------------
    # Voice
    # ------------------------------------------------------------------

    def voice_settings(self, **changes: Any) -> dict[str, Any]:
        """Read or update voice settings; returns the full voice status.

        Every field is validated and persisted (``VoiceSettings.apply`` /
        ``save``); a setting the model does not know is refused by name
        rather than dropped on the floor, which is how ``wake_sensitivity``
        and ``volume`` were being lost before.
        """

        refused = self.voice.update_settings({k: v for k, v in changes.items() if v is not None}) if changes else {}
        status = self.voice.status()
        if refused:
            status["refused"] = refused
            status["ok"] = False
            status["error"] = "; ".join(f"{k}: {v}" for k, v in refused.items())
        else:
            status["ok"] = True
        if "wake_sensitivity" in changes or "volume" in changes:
            self.emit(EventType.DIAGNOSTIC, {"voice": "settings saved", "wake_sensitivity": self.voice.settings.wake_sensitivity,
                                             "volume": self.voice.settings.volume})
        return status

    def _stt_hotwords(self) -> str:
        """A bounded, current entity list for the recogniser: product terms, project titles, capability names, owner vocabulary."""

        from speech.normalize import BUILTIN_ENTITIES, entity_hints

        names: list[str] = [self.identity.assistant_name]
        try:
            names += [str(p.get("title") or "") for p in self.list_projects() if not p.get("hidden")][:12]
        except Exception:  # noqa: BLE001
            pass
        try:
            names += [str(m.capability_id).split(".")[0] for m in self.capabilities.registry.all()][:8]
        except Exception:  # noqa: BLE001
            pass
        try:
            names += self.voice.vocabulary.meant_terms()[-10:]
        except Exception:  # noqa: BLE001
            pass
        try:
            # owner-taught aliases are exactly the words the owner will say
            names += [str(e.get("name") or "") for e in self.aliases.all().values()][:8]
        except Exception:  # noqa: BLE001
            pass
        names += list(BUILTIN_ENTITIES)
        return entity_hints(names, limit=32)

    def _normalizer(self) -> Any:
        from speech.normalize import Normalizer

        entities: list[str] = []
        try:
            entities += [str(p.get("title") or "") for p in self.list_projects() if not p.get("hidden")][:40]
        except Exception:  # noqa: BLE001
            pass
        try:
            entities += [str(m.capability_id).split(".")[0] for m in self.capabilities.registry.all()][:20]
        except Exception:  # noqa: BLE001
            pass
        return Normalizer(entities=entities, vocabulary=self.voice.vocabulary)

    def hear(self, wav: bytes, *, language: str = "", answer: bool = True, wake: Any = None,
             session: str = "", origin: str = "", evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        """Transcribe a posted utterance and, unless told otherwise, reply to it.

        ``wake`` is the detector score that opened this listening session
        (the listener sends it), ``origin`` is ``"ui"`` for the microphone
        button, ``evidence`` is what the device measured while recording.
        Audio that carries neither authority is not a request.

        One utterance is one authoritative event: it gets an identity, its
        audio is measured independently of the recogniser, the recogniser's
        own doubt is read rather than ignored, the wake word is removed only
        inside a wake session, the text is normalised conservatively, and
        the evidence gate rules -- with reasons that go to Activity as one
        ``voice_trace`` -- before anything reaches the conversation.  An
        accepted utterance is entered in the ledger so the same audio can
        never be executed twice.
        """

        import uuid

        from speech.utterance import AudioEvidence, UtteranceEvidence

        evidence = dict(evidence or {})
        authorised = origin == "ui" or self._wake_authorised(wake)
        source = "ui_mic" if origin == "ui" else str(evidence.get("source") or ("microphone" if self._wake_authorised(wake) else "unknown"))
        utterance_id = str(evidence.get("utterance") or "") or f"{session or 'ui'}-{uuid.uuid4().hex[:8]}"
        received = time.monotonic()
        # Speaking to Jarvis is what enters voice mode; it is the least
        # surprising trigger and needs no separate switch.
        self.voice.settings.enabled = True

        # 1. The audio itself, measured before any recogniser has an opinion.
        try:
            from speech.contracts import Audio

            audio = Audio.from_wav(wav)
            audio_evidence = AudioEvidence.from_pcm(audio.samples, audio.sample_rate, audio.width)
        except Exception:  # noqa: BLE001 - unreadable audio is empty evidence; transcribe() reports it
            audio_evidence = AudioEvidence()
        device = {k: v for k, v in evidence.items() if k in {"speech_seconds", "noise_floor", "threshold", "elapsed", "interrupted", "started", "wake_at"}}
        try:
            device_speech = float(device.get("speech_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            device_speech = 0.0
        # Was ZEUS talking while this was recorded?  Either the device says
        # its barge-in interrupted speech, or the core's own playback estimate
        # overlaps the recording window.
        window_start = received - max(audio_evidence.duration_seconds, 0.0) - 0.5
        speaking_overlap = "speech" in str(device.get("interrupted", "")) or self.voice.was_speaking_between(window_start, received)

        utterance = UtteranceEvidence(
            utterance_id=utterance_id, session_id=session, source=source, wake_session_id=session if self._wake_authorised(wake) else "",
            wake_score=float(wake) if self._wake_authorised(wake) else 0.0, started_at=float(device.get("started", 0.0) or 0.0) or time.time(),
            ended_at=time.time(), audio=audio_evidence, device=device, speaking_overlap=speaking_overlap,
        )

        def trace(verdict: Any, transcript: Any = None, segmentation: Any = None, normalized: Any = None) -> dict[str, Any]:
            payload = {"voice_trace": True, "utterance": utterance.to_dict(), "verdict": verdict.to_dict() if verdict is not None else None,
                       "segmentation": segmentation.to_dict() if segmentation is not None else None,
                       "normalization": normalized.to_dict() if normalized is not None else None,
                       "session": session, "wake_score": wake, "text": (transcript.text if transcript is not None else "")[:200]}
            self.emit(EventType.DIAGNOSTIC, payload)
            return payload

        # 2. Silence and noise are refused before the recogniser can invent a sentence for them.
        if authorised and audio_evidence.frames and audio_evidence.duration_seconds >= 0.3:
            silent = audio_evidence.rms < self.voice.gate.settings.min_rms or audio_evidence.peak < self.voice.gate.settings.min_peak
            no_speech = max(audio_evidence.speech_seconds, device_speech) < self.voice.gate.settings.min_speech_seconds
            if silent or no_speech:
                from speech.utterance import Check, Verdict

                why = (f"silence: rms {audio_evidence.rms:.0f}, peak {audio_evidence.peak}" if silent
                       else f"no speech energy: {audio_evidence.speech_seconds:.2f}s above the floor")
                verdict = Verdict(False, why, [Check("audio energy above a silent room", not silent, f"rms {audio_evidence.rms:.0f}"),
                                               Check("speech-like energy present", not no_speech, f"{audio_evidence.speech_seconds:.2f}s")], 0.0, "low")
                self.voice.gate.rejected.append({"text": "", "reason": why, "confidence": 0.0, "at": time.time(), "utterance_id": utterance_id, "session": session})
                del self.voice.gate.rejected[:-50]
                trace(verdict)
                self.state.set(JarvisState.IDLE, detail="nothing heard")
                return {"ok": False, "ignored": True, "reason": why, "text": "", "utterance_id": utterance_id, "wake": wake}

        # 3. Recognise, with the language the conversation is already in and
        #    the current entity names as a bounded decoding bias.
        transcript = self.voice.transcribe(
            wav, language=language or self.voice.settings.language or self.language, hotwords=self._stt_hotwords(),
        )
        utterance.raw_transcript = transcript.raw_text or transcript.text
        utterance.stt = dict(transcript.quality or {})
        utterance.stt.setdefault("language", transcript.language)
        utterance.language = transcript.language
        utterance.confidence = float(transcript.confidence or 0.0)
        if transcript.empty:
            from speech.utterance import Verdict

            trace(Verdict(False, "nothing heard", [], 0.0, "low"), transcript)
            self.state.set(JarvisState.IDLE, detail="nothing heard")
            return {"ok": False, "ignored": True, "text": "", "reason": "no speech detected", "utterance_id": utterance_id}

        # 4. The wake word is session metadata, never command content.
        segmentation = None
        if self._wake_authorised(wake):
            from speech.wake_segment import strip_wake_word

            segmentation = strip_wake_word(transcript.text, wake_word=self._wake_word_name(), words=transcript.words, wake_session=True)
            if segmentation.removed:
                self.emit(EventType.DIAGNOSTIC, {"wake_segment": segmentation.to_dict(), "session": session})
                transcript.text = segmentation.text
                if transcript.empty:
                    from speech.utterance import Verdict

                    trace(Verdict(False, "only the wake word was heard", [], 0.0, "low"), transcript, segmentation)
                    self.state.set(JarvisState.IDLE, detail="only the wake word was heard")
                    return {"ok": False, "ignored": True, "reason": "only the wake word was heard", "text": "", "wake": wake, "utterance_id": utterance_id}

        # 5. Conservative normalisation: known entities and owner corrections only.
        try:
            normalized = self._normalizer().apply(transcript.text)
        except Exception:  # noqa: BLE001 - normalisation must never lose the request
            from speech.normalize import Normalized

            normalized = Normalized(transcript.text, transcript.text)
        utterance.normalized_transcript = normalized.text
        transcript.raw_text = utterance.raw_transcript
        transcript.text = normalized.text

        # 6. The gate rules, with every check named.
        verdict = self.voice.gate.check(utterance, authorised=authorised, recent_spoken=self.voice.spoken_recently(), ledger=self.voice.ledger)
        self.emit(EventType.TRANSCRIPT, {**transcript.to_dict(), "accepted": verdict.accepted, "reason": verdict.reason, "utterance_id": utterance_id})
        trace(verdict, transcript, segmentation, normalized)
        if not verdict.accepted:
            self.state.set(JarvisState.IDLE, detail="utterance ignored")
            self.emit(EventType.DIAGNOSTIC, {"utterance": "ignored", "reason": verdict.reason, "text": transcript.text[:80],
                                             "confidence": round(verdict.confidence, 3), "wake": wake, "session": session, "utterance_id": utterance_id})
            return {"ok": False, "ignored": True, "reason": verdict.reason, "utterance_id": utterance_id, **transcript.to_dict()}
        self.voice.ledger.accept(utterance)

        if transcript.language:
            from persona.language import stable_language

            # Trust the recogniser's own verdict as evidence, but require the
            # same confidence threshold as text: a mis-heard word should not
            # switch the voice.
            self.language = stable_language(transcript.text, current=self.language)

        # 7. Provenance travels with the turn: who said it, when, from which
        #    session, what was heard and what was made of it.
        meta: dict[str, Any] = {
            "source": source, "utterance_id": utterance_id, "session": session,
            "raw_transcript": utterance.raw_transcript, "normalized": normalized.text,
            "replacements": [r.to_dict() for r in normalized.replacements],
            "speech_confidence": round(verdict.confidence, 3), "speech_level": verdict.level,
            "stt": {k: v for k, v in utterance.stt.items() if k != "word_probabilities"},
            "audio": audio_evidence.to_dict(),
        }
        if self._wake_authorised(wake):
            meta.update({"wake_word": self._wake_word_name(), "wake_score": round(float(wake), 3),
                         "wake_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "removed": segmentation.removed if segmentation else ""})
            self.emit(EventType.DIAGNOSTIC, {"wake": self._wake_word_name(), "score": round(float(wake), 3), "session": session,
                                             "command": transcript.text[:200], "utterance_id": utterance_id})
        elif origin == "ui":
            meta["origin"] = "ui"
        if answer:
            self.send_message(transcript.text, meta=meta, request_id=utterance_id)
        else:
            # Transcribed on request without an answer: the turn is over, the
            # eye must not stay on TRANSCRIBING.
            self.state.set(JarvisState.IDLE, detail="transcribed")
        return {"ok": True, "utterance_id": utterance_id, "speech_confidence": round(verdict.confidence, 3), **transcript.to_dict()}

    # ------------------------------------------------------------------
    # Persona
    # ------------------------------------------------------------------

    def list_personas(self) -> dict[str, Any]:
        try:
            store = self.personas
            return {
                "active": store.active().to_dict(),
                "personas": [store.get(name).to_dict() for name in store.names()],
                "language": self.language,
            }
        except Exception as exc:
            return {"error": str(exc), "personas": []}

    def set_persona(self, name: str) -> dict[str, Any]:
        try:
            persona = self.personas.activate(name)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        self.emit(EventType.NOTIFICATION, {"text": f"persona: {persona.name}"})
        return {"ok": True, "active": persona.to_dict()}

    def set_language(self, language: str) -> dict[str, Any]:
        """Pin the conversation language, or pass "" to go back to detecting it."""

        self.language = (language or "").strip().lower()
        if self._voice is not None:
            self._voice.settings.language = self.language
        return {"ok": True, "language": self.language or "auto"}

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------

    def device_pair_request(self, name: str, kind: str = "generic", capabilities: list[str] | None = None) -> dict[str, Any]:
        request = self.gateway.request_pairing(name, kind=kind, capabilities=capabilities)
        # Surfaced as a notification so the code appears wherever the user is
        # already looking, rather than only in a panel they must think to open.
        self.emit(
            EventType.NOTIFICATION,
            {"text": f"{request.name} wants to pair. Code: {request.code}"},
        )
        return {"ok": True, "code": request.code, "expires_in": 300}

    def device_pair_collect(self, code: str) -> dict[str, Any]:
        return self.gateway.collect(str(code))

    def device_approve(self, code: str) -> dict[str, Any]:
        device = self.gateway.approve(str(code))
        if device is None:
            return {"ok": False, "error": "no such pairing request, or it expired"}
        self.emit(EventType.NOTIFICATION, {"text": f"paired: {device.name}"})
        return {"ok": True, "device": device.to_dict()}

    def device_deny(self, code: str) -> dict[str, Any]:
        return {"ok": self.gateway.deny(str(code))}

    def device_list(self) -> dict[str, Any]:
        return self.gateway.status()

    def device_revoke(self, device_id: str) -> dict[str, Any]:
        return {"ok": self.gateway.revoke(str(device_id))}

    def device_heartbeat(self, device_id: str) -> dict[str, Any]:
        device = self.gateway.touch(device_id)
        if device is None:
            return {"ok": False, "error": "unknown device"}
        return {
            "ok": True,
            "state": self.state.snapshot.to_dict(),
            "commands": self.gateway.drain(device_id),
        }

    def device_display(self, device_id: str, command: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"ok": self.gateway.send(str(device_id), command, payload)}

    # ------------------------------------------------------------------
    # Self-development
    # ------------------------------------------------------------------

    @property
    def selfdev_store(self) -> Any:
        from service.selfdev import SelfDevStore

        if getattr(self, "_selfdev_store", None) is None:
            self._selfdev_store = SelfDevStore(Path(self.kernel.state_root) / "selfdev")
        return self._selfdev_store

    def list_selfdev(self) -> dict[str, Any]:
        return {"missions": [m.to_dict() for m in self.selfdev_store.list()][-20:]}

    def _answer_by_self_development(self, text: str, scope: str, classification: Any = None) -> None:
        """"Change something about yourself": a mission, not a conversation.

        Acknowledged at once, run in its own thread so the conversation stays
        open, and reported when it is verified -- which, under the supervisor,
        is after the restart that proves it.
        """

        from service.selfdev import SelfDevMission, SelfDevRunner

        # Only the installed product may develop the installed product.  A
        # core with a state root elsewhere -- a test's temporary kernel, an
        # ad-hoc script -- would otherwise build in a worktree of the live
        # repository, call the expert and promote into it; a unit test did
        # exactly that on 2026-08-27 (commit 546d43f).
        allowed, why = self._selfdev_allowed()
        if not allowed:
            self.emit(EventType.TOOL, {"summary": f"self-development refused: {why}", "source": "selfdev"}, scope=scope)
            de = self.language.startswith("de")
            self._deliver((f"Selbstentwicklung ist hier nicht verfügbar: {why}" if de else f"Self-development is not available here: {why}"),
                          scope=scope, backend="selfdev", final_state=JarvisState.WAITING)
            return

        active = self.selfdev_store.active()
        if active is not None:
            de = self.language.startswith("de")
            self._deliver(
                (f"Ich arbeite bereits an einem Selbst-Update ({active.phase}: „{active.request[:60]}“). "
                 f"Sobald es verifiziert ist, nehme ich das nächste.") if de else
                (f"I am already working on a self-update ({active.phase}: “{active.request[:60]}”). "
                 f"I will take the next one once it is verified."),
                scope=scope, backend="selfdev",
            )
            return

        mission = SelfDevMission(request=text, scope=scope, language=self.language)
        if classification is not None and getattr(classification, "route", None) is not None:
            mission.routing = classification.route.to_dict()
        self.selfdev_store.save(mission)
        self.emit(EventType.NOTIFICATION, {"text": f"self-development mission {mission.mission_id} started",
                                           "kind": "selfdev", "mission_id": mission.mission_id, "request": text[:200]},
                  scope=scope)
        de = self.language.startswith("de")
        self._deliver(
            (f"Verstanden. Ich entwickle das jetzt selbst (Mission {mission.mission_id}): isolierter Arbeitsbaum, "
             f"lokales Coder-Modell, Verifikation, dann Übernahme und Neustart. Ich melde mich, wenn es verifiziert ist.")
            if de else
            (f"Understood. I will develop that myself (mission {mission.mission_id}): isolated worktree, local coder "
             f"model, verification, then promotion and a restart. I will report when it is verified."),
            scope=scope, backend="selfdev", final_state=JarvisState.CODING,
        )

        runner = SelfDevRunner(
            repository=self.selfdev_repository(),
            store=self.selfdev_store, kernel=self.kernel, owner=self.owner, lifecycle=self.lifecycle,
            gateway=self.experts, emit=lambda kind, payload: self.emit(kind, payload, scope=scope),
            set_state=self.state.set,
        )

        def work() -> None:
            from service.selfdev import describe

            try:
                finished = runner.run(mission)
            except Exception as exc:  # noqa: BLE001
                finished = mission
                finished.outcome, finished.reason, finished.phase = "failed", f"{type(exc).__name__}: {exc}", "FAILED"
                self.selfdev_store.save(finished)
            if finished.phase == "RESTARTING":
                return  # the verdict arrives after the restart
            self._deliver(describe(finished, self.language), scope=scope, backend="selfdev",
                          final_state=JarvisState.IDLE if finished.outcome != "failed" else JarvisState.ERROR)

        threading.Thread(target=work, daemon=True, name=f"selfdev-{mission.mission_id}").start()

    def _settle_selfdev_after_restart(self) -> int:
        from service.selfdev import describe, settle_after_restart

        settled = settle_after_restart(self.selfdev_store, self.lifecycle)
        for mission in settled:
            self._deliver(describe(mission, mission.language or self.language), scope=mission.scope, backend="selfdev",
                          final_state=JarvisState.IDLE if mission.outcome == "promoted" else JarvisState.ERROR)
        settled_count = len(settled)
        # A mission that was mid-flight when this process last died did not
        # finish: say so in its record rather than letting it look active for
        # ever.  Its candidate is released (diff kept) below.
        interrupted = 0
        for mission in self.selfdev_store.list():
            if mission.finished or mission.phase == "RESTARTING":
                continue
            mission.outcome, mission.reason, mission.phase = "failed", f"interrupted by a restart during {mission.phase}", "FAILED"
            mission.events.append({"at": mission.updated_at, "phase": "FAILED", "detail": mission.reason})
            self.selfdev_store.save(mission)
            interrupted += 1
        try:
            self._sweep_selfdev_isolation()
        except Exception as exc:  # noqa: BLE001
            self.emit(EventType.DIAGNOSTIC, {"warming": f"selfdev isolation sweep failed: {exc}"})
        try:
            for m in self.missions.mark_interrupted():
                self.emit(EventType.NOTIFICATION, {"kind": "mission", "text": f"mission {m.mission_id} ({m.kind}) was interrupted during {m.phase}; resumable from its record",
                                                   "mission_id": m.mission_id})
        except Exception as exc:  # noqa: BLE001
            self.emit(EventType.DIAGNOSTIC, {"warming": f"mission engine sweep failed: {exc}"})
        if interrupted:
            self.emit(EventType.NOTIFICATION, {"kind": "selfdev", "text": f"{interrupted} self-development mission(s) were interrupted by the restart"})
        return settled_count + interrupted

    def _sweep_selfdev_isolation(self) -> dict[str, Any]:
        """At startup: no candidate outlives its mission, and no promotion stays half-applied."""

        from deployment.promotion import recover_interrupted
        from service.isolation import CandidateWorkspace

        repository = self.selfdev_repository()
        keep = [m.mission_id for m in self.selfdev_store.list()
                if m.phase in {"RESTARTING"} or (m.verification.get("ok") and m.outcome == "failed")]
        removed = CandidateWorkspace.reap(repository, keep=keep)
        recovered = recover_interrupted(repository)
        report = {"worktrees_removed": removed, "promotions_recovered": recovered, "kept": keep}
        if removed or recovered:
            self.emit(EventType.TOOL, {"summary": f"isolation sweep: {len(removed)} stale candidate(s) removed, "
                                                  f"{len(recovered)} interrupted promotion(s) restored",
                                       "source": "isolation", "report": report})
        return report

    # ------------------------------------------------------------------
    # Releases -- the executable as one more thing with a known-good version
    # ------------------------------------------------------------------

    @property
    def releases(self) -> Any:
        from deployment.release import ReleaseManager

        if getattr(self, "_releases", None) is None:
            self._releases = ReleaseManager(self.selfdev_repository(), log=lambda m: self.emit(
                EventType.TOOL, {"summary": m[:200], "source": "release"}))
        return self._releases

    def release_status(self) -> dict[str, Any]:
        return self.releases.status()

    def release_build(self, *, verify: bool = True) -> dict[str, Any]:
        """Build a candidate ZEUS.exe in the background; verify it when asked."""

        if getattr(self, "_release_thread", None) is not None and self._release_thread.is_alive():
            return {"ok": False, "error": "a release build is already running"}

        def work() -> None:
            self.state.set(JarvisState.WORKING, detail="building a candidate release")
            try:
                record = self.releases.build_candidate()
                if record.outcome == "built" and verify:
                    self.state.set(JarvisState.VERIFYING, detail="verifying the candidate release")
                    self.releases.verify_candidate(record.candidate)
                self.emit(EventType.NOTIFICATION, {"kind": "release", "text": f"release build {record.outcome}: {record.reason[:160]}",
                                                   "candidate": record.candidate})
            finally:
                self.state.set(JarvisState.IDLE)

        self._release_thread = threading.Thread(target=work, daemon=True, name="release-build")
        self._release_thread.start()
        return {"ok": True, "started": True}

    def release_verify(self, candidate: str) -> dict[str, Any]:
        return self.releases.verify_candidate(candidate).to_dict()

    def release_promote(self, candidate: str, *, relaunch: bool = True, authorization: str = "") -> dict[str, Any]:
        """Promote a verified candidate and, under the supervisor, relaunch into it.

        With an owner password set, promoting code into the product is a
        Level-2 change behind a SELFDEV_PROMOTE authorization.
        """

        if self.security.configured:
            denied = self.require_auth(authorization, "SELFDEV_PROMOTE")
            if denied is not None:
                return denied
        record = self.releases.promote(candidate)
        out = record.to_dict()
        # "staged": the running exe locks its directory (Windows); the swap is
        # the relaunch watchdog's, which runs after this supervisor has exited.
        if record.outcome in {"promoted", "staged"} and relaunch:
            exe = str(self.releases.known_good / "ZEUS.exe")
            previous = str(self.releases.previous) if (self.releases.previous / "ZEUS.exe").is_file() else ""
            out["relaunch"] = self.lifecycle.request_relaunch(
                f"release {record.release_id} promoted", exe=exe, previous=previous, promotion_id=record.release_id,
                requested_by="release",
            )
        return out

    def release_rollback(self, *, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            return {"ok": False, "error": "a release rollback needs confirm=true"}
        return self.releases.rollback("owner requested").to_dict()

    # ------------------------------------------------------------------
    # Universal search -- one box over everything ZEUS keeps
    # ------------------------------------------------------------------

    def universal_search(self, query: str, *, limit: int = 30, types: Iterable[str] = ()) -> dict[str, Any]:
        """Projects, missions, capabilities, corrections, knowledge, activity, receipts.

        Local indexes only; every hit names its type and where to open it.
        Substring matching over the stored records: cheap, instant, honest.
        """

        q = (query or "").strip().lower()
        wanted = set(types) if types else set()
        results: list[dict[str, Any]] = []
        if not q:
            return {"results": results, "query": query}

        def want(kind: str) -> bool:
            return not wanted or kind in wanted

        def add(kind: str, ident: str, title: str, snippet: str = "", when: str = "", score: int = 1) -> None:
            results.append({"type": kind, "id": ident, "title": title[:160], "snippet": snippet[:200], "when": when, "score": score})

        try:
            if want("project"):
                for p in self.list_projects().get("projects", []):
                    text = f"{p.get('title', '')} {p.get('goal', '')}".lower()
                    if q in text:
                        add("project", str(p.get("id", "")), p.get("title") or p.get("goal", ""), f"{p.get('state', '')} · {p.get('tasks', 0)} tasks", str(p.get("updated_at", "")), 3)
        except Exception:  # noqa: BLE001
            pass
        try:
            if want("mission"):
                for m in self.selfdev_store.list():
                    if q in m.request.lower() or q in m.mission_id:
                        add("mission", m.mission_id, m.request, f"{m.phase} · {m.outcome or 'running'}", m.updated_at, 3)
        except Exception:  # noqa: BLE001
            pass
        try:
            if want("capability"):
                for c in self.list_capabilities():
                    text = f"{c.get('capability_id', '')} {c.get('description', '')}".lower()
                    if q in text:
                        add("capability", str(c.get("capability_id", "")), str(c.get("capability_id", "")), f"{c.get('status', '')} v{c.get('version', '')}", "", 3)
        except Exception:  # noqa: BLE001
            pass
        try:
            if want("correction"):
                for c in self.corrections.list(include_inactive=True):
                    if q in f"{c.what_was_wrong} {c.original_request}".lower():
                        add("correction", c.correction_id, c.what_was_wrong, f"{c.classification} · {c.scope}", c.at, 2)
        except Exception:  # noqa: BLE001
            pass
        try:
            if want("knowledge"):
                graph = self.knowledge_graph(query=query, limit=20)
                for n in graph.get("nodes", [])[:12]:
                    add("knowledge", str(n.get("id", "")), str(n.get("title", "")), str(n.get("type", "")), str(n.get("updated_at", "")), 2)
        except Exception:  # noqa: BLE001
            pass
        try:
            if want("activity"):
                for entry in reversed(self.list_activity(600).get("activity", [])):
                    if q in str(entry.get("summary", "")).lower():
                        add("receipt" if entry.get("receipt_id") else "activity", str(entry.get("receipt_id") or entry.get("seq", "")),
                            str(entry.get("summary", "")), str(entry.get("kind", "")), str(entry.get("at", "")), 1)
                        if sum(1 for r in results if r["type"] in {"activity", "receipt"}) >= 10:
                            break
        except Exception:  # noqa: BLE001
            pass
        results.sort(key=lambda r: (-r["score"], r["when"]), reverse=False)
        results.sort(key=lambda r: -r["score"])
        return {"results": results[:limit], "query": query, "count": len(results)}

    def selfdev_diff(self, mission_id: str) -> dict[str, Any]:
        """The candidate's diff: from the kept evidence patch, or the live worktree."""

        mission = self.selfdev_store.load(mission_id)
        if mission is None:
            return {"ok": False, "error": f"no mission {mission_id}"}
        if mission.evidence_patch and Path(mission.evidence_patch).is_file():
            return {"ok": True, "patch": Path(mission.evidence_patch).read_text(encoding="utf-8", errors="replace")[:200_000], "source": "evidence"}
        if mission.worktree and Path(mission.worktree).is_dir():
            from service.isolation import CandidateWorkspace

            ws = CandidateWorkspace.attach(self.selfdev_repository(), mission_id, mission.worktree)
            return {"ok": True, "patch": ws.diff()[:200_000], "source": "worktree"}
        return {"ok": False, "error": "no diff is kept for this mission"}

    # ------------------------------------------------------------------
    # Wake word -- recordings, training and a test, all from the product
    # ------------------------------------------------------------------

    def _wake_dir(self) -> Path:
        return self.selfdev_repository() / "data" / "wake"

    def _wake_word_name(self) -> str:
        try:
            from core.identity import current

            return str(current().wake_word or "Zeus")
        except Exception:  # noqa: BLE001
            return "Zeus"

    def _wake_authorised(self, wake: Any) -> bool:
        """A listener that heard the wake word says so with the score that fired."""

        try:
            return wake is not None and float(wake) > 0.0
        except (TypeError, ValueError):
            return False

    def _wake_model_path(self) -> Path:
        return self.selfdev_repository() / "data" / "models" / "wake" / "zeus.npz"

    def wake_effective_threshold(self) -> tuple[float, str]:
        """One number for the listener and the Voice Studio test, and where it came from."""

        from speech.wake_zeus import read_manifest, resolve_threshold

        return resolve_threshold(self.voice.settings.wake_sensitivity, read_manifest(self._wake_model_path()).get("threshold"))

    def wake_status(self) -> dict[str, Any]:
        """Everything Voice Studio, the doctor and the listener need to agree on the wake word."""

        from speech.wake_zeus import model_fingerprint, read_manifest

        root = self._wake_dir()
        model = self._wake_model_path()
        manifest = read_manifest(model)
        evaluation: dict[str, Any] = {}
        eval_path = model.with_name("zeus_eval.json")
        try:
            evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else {}
        except (OSError, ValueError):
            evaluation = {}
        positive = len(list((root / "positive").glob("*.wav"))) if (root / "positive").is_dir() else 0
        negative = len(list((root / "negative").glob("*.wav"))) if (root / "negative").is_dir() else 0
        hard_negative = len(list((root / "hard_negative").glob("*.wav"))) if (root / "hard_negative").is_dir() else 0
        trained = model.is_file()
        owner_trained = bool((manifest.get("dataset") or {}).get("owner_positive"))
        threshold, source = self.wake_effective_threshold()
        fingerprint = model_fingerprint(model) if trained else ""
        listener = dict(getattr(self, "_wake_listener", {}) or {})
        listener_fresh = bool(listener) and time.time() - float(listener.get("at", 0)) < 30.0
        match = bool(listener_fresh and listener.get("fingerprint") == fingerprint and abs(float(listener.get("threshold", -1)) - threshold) < 1e-6)
        owner_eval = manifest.get("owner_holdout_evaluation") or manifest.get("owner_evaluation") or {}
        at = evaluation.get("at_effective_threshold") or {}
        return {
            "ok": True, "wake_word": "zeus", "model_trained": trained, "model": str(model) if trained else "",
            "speaking": bool(self._voice is not None and self._voice.speaking),
            "model_kind": "OWNER" if trained and owner_trained else ("SYNTHETIC" if trained else "NONE"),
            "model_fingerprint": fingerprint,
            "positive": positive, "negative": negative, "hard_negative": hard_negative, "owner_samples": owner_trained,
            "threshold": manifest.get("threshold"), "manifest_threshold": manifest.get("threshold"),
            "configured_sensitivity": self.voice.settings.wake_sensitivity,
            "effective_threshold": threshold, "threshold_source": source,
            "trained_at": manifest.get("trained_at"),
            "last_score": getattr(self, "_wake_last_test", None),
            "evaluation": {
                "at": evaluation.get("evaluated_at"), "in_sample": evaluation.get("in_sample", True),
                "stale": bool(evaluation) and evaluation.get("model_fingerprint") != fingerprint,
                "positive_recall": at.get("recall"), "positives_detected": at.get("positives_detected"),
                "negative_rejection": at.get("rejection"), "false_activations": at.get("false_activations"),
                "positive_scores": evaluation.get("positive_scores"), "negative_scores": evaluation.get("negative_scores"),
                "recommended_threshold": evaluation.get("recommended_threshold"), "separates": evaluation.get("separates"),
                "silent_positives": evaluation.get("silent_positives", []), "thresholds": evaluation.get("thresholds", []),
                "hard_negatives_evaluated": evaluation.get("hard_negatives_evaluated", False),
                "counts": evaluation.get("counts"),
            } if evaluation else None,
            "owner_holdout": self._holdout_view(owner_eval, threshold),
            "listener": listener if listener_fresh else None, "listener_match": match,
            "metrics": manifest.get("metrics") or manifest.get("held_out") or None, "manifest": manifest,
        }

    @staticmethod
    def _holdout_view(owner_eval: dict[str, Any], threshold: float) -> dict[str, Any] | None:
        """The hold-out figures at the *current* effective threshold, or None when nothing was held out."""

        if not owner_eval or owner_eval.get("in_sample", True):
            return None
        rows = owner_eval.get("thresholds") or []
        at = next((r for r in rows if abs(float(r.get("threshold", -1)) - float(threshold)) < 1e-9), None) or owner_eval.get("at_effective_threshold")
        return {**owner_eval, "at_effective_threshold": at, "effective_threshold": threshold}

    def pronunciation(self, text: str = "", *, language: str = "") -> dict[str, Any]:
        """The lexicon and, for ``text``, what the provider would be given to say."""

        voice = self.voice
        result: dict[str, Any] = {"ok": True, "provider": voice.pronouncer.provider, "path": str(voice.lexicon.owner_path or ""),
                                  "entries": [e.to_dict() for e in voice.lexicon.all()],
                                  "owner_entries": [e.to_dict() for e in voice.lexicon.owner_entries()],
                                  "recent": list(voice.last_spoken[-10:])}
        if text:
            result["preview"] = voice.pronouncer.render(text, language=language or voice.settings.language or self.language or "de").to_dict()
        return result

    def pronunciation_set(self, surface: str, spoken: str, *, language: str = "", note: str = "", test: bool = True) -> dict[str, Any]:
        """An owner correction of how a word is spoken; synthesis is tried so the entry is known to work."""

        surface, spoken = str(surface or "").strip(), str(spoken or "").strip()
        if not surface or not spoken:
            return {"ok": False, "error": "surface and spoken form are required"}
        language = (language or self.voice.settings.language or self.language or "de")[:2]
        entry = self.voice.lexicon.set(surface, spoken, language=language, provider=self.voice.pronouncer.provider, note=note)
        tested: dict[str, Any] = {"tried": False}
        if test:
            try:
                audio = self.voice.engine.synthesize(spoken, voice=self.voice.settings.voice_id, language=language)
                key = self.voice.store.put(audio)
                tested = {"tried": True, "ok": audio.seconds > 0.05, "seconds": round(audio.seconds, 2), "url": f"/api/voice/audio/{key}.wav"}
            except Exception as exc:  # noqa: BLE001
                tested = {"tried": True, "ok": False, "error": str(exc)[:200]}
        self.emit(EventType.NOTIFICATION, {"kind": "pronunciation", "text": f"Aussprache gelernt: {surface} → {spoken}" if language == "de" else f"pronunciation learned: {surface} → {spoken}"})
        return {"ok": True, "entry": entry.to_dict(), "test": tested}

    def pronunciation_remove(self, surface: str, *, language: str = "") -> dict[str, Any]:
        language = (language or self.voice.settings.language or self.language or "de")[:2]
        return {"ok": self.voice.lexicon.remove(surface, language=language)}

    def wake_listener_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """The listener says what it loaded; Voice Studio shows whether that is what the test uses."""

        self._wake_listener = {"model": str(report.get("model", "")), "fingerprint": str(report.get("fingerprint", "")),
                               "threshold": report.get("threshold"), "pid": report.get("pid"), "at": time.time(),
                               "last_score": report.get("last_score")}
        threshold, source = self.wake_effective_threshold()
        from speech.wake_zeus import model_fingerprint

        return {"ok": True, "effective_threshold": threshold, "threshold_source": source,
                "model_fingerprint": model_fingerprint(self._wake_model_path())}

    def wake_record(self, wav: bytes, *, kind: str) -> dict[str, Any]:
        if kind not in {"positive", "negative", "hard_negative"}:
            return {"ok": False, "error": "kind must be positive, negative or hard_negative"}
        if len(wav) < 2000 or wav[:4] != b"RIFF":
            return {"ok": False, "error": "not a WAV recording"}
        folder = self._wake_dir() / kind
        folder.mkdir(parents=True, exist_ok=True)
        n = len(list(folder.glob("owner_*.wav"))) + 1
        (folder / f"owner_{n:03d}.wav").write_bytes(wav)
        self.emit(EventType.TOOL, {"summary": f"wake-word sample recorded ({kind} #{n})", "source": "voice"})
        status = self.wake_status()
        return {"ok": True, "saved": str(folder / f"owner_{n:03d}.wav"), "positive": status["positive"], "negative": status["negative"],
                "hard_negative": status["hard_negative"]}

    def _speech_python(self) -> str:
        venv = self.selfdev_repository() / ".venv-speech" / "Scripts" / "python.exe"
        return str(venv) if venv.is_file() else ""

    def wake_train(self) -> dict[str, Any]:
        python = self._speech_python()
        if not python:
            return {"ok": False, "error": "no .venv-speech; voice is not set up"}
        self.state.set(JarvisState.WORKING, detail="training the wake word")
        try:
            completed = subprocess.run([python, "-m", "speech.wake_training"], cwd=str(self.selfdev_repository()), capture_output=True,
                                       text=True, timeout=1800, encoding="utf-8", errors="replace",
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
        except subprocess.TimeoutExpired:
            self.state.set(JarvisState.IDLE)
            return {"ok": False, "error": "training did not finish within 30 minutes"}
        finally:
            self.state.set(JarvisState.IDLE)
        output = f"{completed.stdout}\n{completed.stderr}"
        metrics: dict[str, Any] = {}
        for line in output.splitlines():
            lowered = line.lower()
            if "recall" in lowered:
                metrics.setdefault("lines", []).append(line.strip())
        if completed.returncode == 0:
            self.wake_evaluate()
        status = self.wake_status()
        self.emit(EventType.NOTIFICATION, {"kind": "voice", "text": "wake-word model trained; the listener reloads it within seconds"
                                           if completed.returncode == 0 else "wake-word training failed"})
        return {"ok": completed.returncode == 0, "output": output[-3000:], "metrics": {**(status.get("metrics") or {}), **metrics},
                "trained_at": status.get("trained_at"), "status": status, "error": "" if completed.returncode == 0 else output[-800:]}

    @staticmethod
    def wake_test_script(path: str, threshold: float) -> str:
        """The program the speech venv runs for one test: the shared scorer, the effective threshold.

        The audio goes through :func:`speech.wake_eval.score_wav` -- int16
        PCM, resampled if needed, the detector fed frame by frame as the
        listener feeds the microphone.  (The previous version divided the
        samples by 32768 and then cast to int16, which turned every recording
        into near-silence; the scores it reported were noise.)
        """

        return (
            "import json\n"
            "from speech.wake_zeus import ZeusDetector\n"
            "from speech.wake_eval import score_wav\n"
            f"det = ZeusDetector.load(threshold={float(threshold)!r})\n"
            f"r = score_wav(det, {path!r})\n"
            "r['fingerprint'] = det.fingerprint\n"
            "print(json.dumps(r))\n"
        )

    def wake_test(self, wav: bytes) -> dict[str, Any]:
        """Score a recording with the trained detector at the effective threshold, in the speech venv."""

        python = self._speech_python()
        if not python:
            return {"ok": False, "error": "no .venv-speech; voice is not set up"}
        if not self._wake_model_path().is_file():
            return {"ok": False, "error": "no trained wake model; train it first"}
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(wav)
            path = fh.name
        threshold, source = self.wake_effective_threshold()
        try:
            completed = subprocess.run([python, "-c", self.wake_test_script(path, threshold)], cwd=str(self.selfdev_repository()),
                                       capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace",
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "the detector did not answer within 120s"}
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if completed.returncode != 0:
            return {"ok": False, "error": completed.stderr.strip()[-600:] or "detector failed"}
        try:
            result = json.loads(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "error": "unreadable detector output: " + completed.stdout[-300:]}
        result.pop("frames", None)
        self._wake_last_test = {"score": result.get("score"), "detected": result.get("detected"), "threshold": threshold,
                                "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "silent": result.get("silent")}
        self.emit(EventType.DIAGNOSTIC, {"wake_test": self._wake_last_test})
        return {"ok": True, "threshold_source": source, **result}

    def wake_evaluate(self) -> dict[str, Any]:
        """Calibrate: the owner's recordings through the real detector; writes zeus_eval.json."""

        python = self._speech_python()
        if not python:
            return {"ok": False, "error": "no .venv-speech; voice is not set up"}
        if not self._wake_model_path().is_file():
            return {"ok": False, "error": "no trained wake model; train it first"}
        threshold, _source = self.wake_effective_threshold()
        try:
            completed = subprocess.run([python, "-m", "speech.wake_eval", "--threshold", str(threshold), "--json"],
                                       cwd=str(self.selfdev_repository()), capture_output=True, text=True, timeout=600,
                                       encoding="utf-8", errors="replace",
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "the evaluation did not finish within 10 minutes"}
        if completed.returncode != 0:
            return {"ok": False, "error": completed.stderr.strip()[-600:] or "evaluation failed"}
        try:
            report = json.loads(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "error": "unreadable evaluation output: " + completed.stdout[-300:]}
        manifest_eval = (self.wake_status().get("manifest") or {}).get("owner_evaluation") or {}
        report["in_sample"] = True  # the recordings on disk are the ones training used (hold-out figures live in the manifest)
        try:
            eval_path = self._wake_model_path().with_name("zeus_eval.json")
            eval_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError:
            pass
        self.emit(EventType.NOTIFICATION, {"kind": "voice", "text": f"wake-word evaluation: recall {report.get('at_effective_threshold', {}).get('recall')} "
                                                                    f"at {threshold}, recommended {report.get('recommended_threshold')}"})
        return {"ok": True, "report": {k: v for k, v in report.items() if k != "clips"}, "holdout": manifest_eval or None,
                "status": self.wake_status()}

    # ------------------------------------------------------------------
    # Schach Analyse -- a screen-watching chess assistant, its own process
    # ------------------------------------------------------------------

    def _chess_status_path(self) -> Path:
        return Path(self.kernel.state_root) / "tools" / "chess_analysis.json"

    def chess_tool_status(self) -> dict[str, Any]:
        from tools.chess_analysis.engine import default_stockfish_path
        from tools.chess_analysis.recognize import default_model_path

        proc = getattr(self, "_chess_proc", None)
        alive = proc is not None and proc.poll() is None
        status: dict[str, Any] = {}
        path = self._chess_status_path()
        try:
            status = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            status = {}
        return {"ok": True, "running": alive, "pid": proc.pid if alive else None, "stockfish": str(default_stockfish_path() or ""),
                "model": str(default_model_path() or ""), "status": status if alive or status.get("running") is False else status}

    def chess_tool_start(self) -> dict[str, Any]:
        """Start the overlay process (system python: it has torch/ultralytics/cv2/mss; CPU inference only)."""

        from tools.chess_analysis.engine import default_stockfish_path
        from tools.chess_analysis.recognize import default_model_path

        proc = getattr(self, "_chess_proc", None)
        if proc is not None and proc.poll() is None:
            return {"ok": True, "running": True, "pid": proc.pid, "already": True}
        stockfish, model = default_stockfish_path(), default_model_path()
        if stockfish is None:
            return {"ok": False, "error": "no local Stockfish found (expected under D:\\stockfish-windows-x86-64-avx2)"}
        if model is None:
            return {"ok": False, "error": "no trained piece model found (expected under D:\\Chessaru\\runs\\detect\\...\\best.pt)"}
        command = [sys.executable, "-m", "tools.chess_analysis", "--stockfish", str(stockfish), "--model", str(model),
                   "--status", str(self._chess_status_path())]
        try:
            self._chess_proc = subprocess.Popen(command, cwd=str(self.selfdev_repository()),
                                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
        except OSError as exc:
            return {"ok": False, "error": f"could not start: {exc}"}
        self.emit(EventType.TOOL, {"summary": f"Schach Analyse started (pid {self._chess_proc.pid})", "source": "chess"})
        return {"ok": True, "running": True, "pid": self._chess_proc.pid}

    def chess_tool_stop(self) -> dict[str, Any]:
        proc = getattr(self, "_chess_proc", None)
        if proc is None or proc.poll() is not None:
            return {"ok": True, "running": False}
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
        self.emit(EventType.TOOL, {"summary": "Schach Analyse stopped", "source": "chess"})
        return {"ok": True, "running": False}

    def doctor(self) -> dict[str, Any]:
        """Deterministic health: never wakes a model (service.doctor)."""

        from service.doctor import Doctor

        return Doctor(self, repository=self.selfdev_repository()).run()

    def _selfdev_allowed(self) -> tuple[bool, str]:
        """Whether this core is the installed product (state root inside the repository)."""

        try:
            state = Path(self.kernel.state_root).resolve()
            repo = self.selfdev_repository().resolve()
        except OSError as exc:
            return False, f"paths unreadable: {exc}"
        if not state.is_relative_to(repo):
            return False, f"this core's state ({state}) is not the installed product's ({repo}); only the product develops the product"
        if not (repo / "zeus_supervisor").is_dir():
            return False, "the repository does not look like ZEUS"
        return True, ""

    def selfdev_repository(self) -> Path:
        """ZEUS's own directory, from where this code lives -- never from an
        environment variable a mission could have inherited."""

        return Path(__file__).resolve().parents[1]

    def cancel_selfdev(self, mission_id: str) -> dict[str, Any]:
        mission = self.selfdev_store.load(mission_id)
        if mission is None:
            return {"ok": False, "error": f"no mission {mission_id}"}
        if mission.finished:
            return {"ok": False, "error": f"mission {mission_id} is already {mission.phase}"}
        mission.cancel_requested = True
        self.selfdev_store.save(mission)
        self.emit(EventType.PROGRESS, {"summary": f"selfdev cancel requested: {mission_id}", "kind": "selfdev",
                                       "mission_id": mission_id, "phase": mission.phase})
        return {"ok": True, "mission_id": mission_id, "phase": mission.phase}

    def resume_selfdev(self, mission_id: str) -> dict[str, Any]:
        """Continue a failed mission whose verified candidate is still on disk."""

        from service.selfdev import SelfDevRunner, describe

        mission = self.selfdev_store.load(mission_id)
        if mission is None:
            return {"ok": False, "error": f"no mission {mission_id}"}
        allowed, why = self._selfdev_allowed()
        if not allowed:
            return {"ok": False, "error": why}
        if self.selfdev_store.active() is not None:
            return {"ok": False, "error": "another self-development mission is active"}
        runner = SelfDevRunner(
            repository=self.selfdev_repository(), store=self.selfdev_store, kernel=self.kernel, owner=self.owner,
            lifecycle=self.lifecycle, gateway=self.experts,
            emit=lambda kind, payload: self.emit(kind, payload, scope=mission.scope), set_state=self.state.set,
        )

        def work() -> None:
            finished = runner.resume(mission)
            if finished.phase != "RESTARTING":
                self._deliver(describe(finished, self.language), scope=mission.scope, backend="selfdev",
                              final_state=JarvisState.IDLE if finished.outcome != "failed" else JarvisState.ERROR)

        threading.Thread(target=work, daemon=True, name=f"selfdev-resume-{mission_id}").start()
        return {"ok": True, "mission_id": mission_id}

    # ------------------------------------------------------------------
    # Korrigieren -- owner corrections
    # ------------------------------------------------------------------

    @property
    def corrections(self) -> Any:
        from service.corrections import CorrectionStore

        if getattr(self, "_corrections", None) is None:
            self._corrections = CorrectionStore(Path(self.kernel.state_root) / "owner" / "corrections.jsonl")
        return self._corrections

    def correction_context(self, receipt_id: str) -> dict[str, Any]:
        """What the Korrigieren dialog shows: request, reading, action, result."""

        receipt = self.receipts.get(receipt_id) if hasattr(self.receipts, "get") else None
        if receipt is None:
            for item in self._session_receipts:
                if getattr(item, "id", "") == receipt_id:
                    receipt = item
                    break
        if receipt is None:
            return {"ok": False, "error": f"no receipt {receipt_id}"}
        from service.intent import classify

        data = receipt.to_dict() if hasattr(receipt, "to_dict") else dict(receipt)
        request = str(data.get("request", ""))
        classification = classify(request)
        evidence = dict(data.get("evidence") or {})
        return {
            "ok": True,
            "receipt_id": receipt_id,
            "original_request": request,
            "parsed_intent": classification.intent.value,
            "intent_reason": classification.reason,
            "entities": {k: v for k, v in evidence.items() if isinstance(v, (str, int, float)) and k in {"path", "track", "provider", "query", "name", "artist"}},
            "executed_action": str(data.get("kind", "")),
            "observed_result": str(data.get("detail", "")),
            "verified": bool(data.get("verified", False)),
            "ok_flag": bool(data.get("ok", False)),
            "verifications": data.get("verifications", []),
        }

    def correction_classify(self, what_was_wrong: str, *, receipt_id: str = "") -> dict[str, Any]:
        from service.corrections import CLASSES, SCOPES, classify_correction, rule_for

        context = self.correction_context(receipt_id) if receipt_id else {}
        request = str(context.get("original_request", ""))
        classification, scope, reason = classify_correction(
            what_was_wrong, receipt_ok=context.get("ok_flag") if context else None, request=request,
        )
        when, then = rule_for(what_was_wrong, request=request, classification=classification, scope=scope,
                              parsed_intent=str(context.get("parsed_intent", "")), entities=dict(context.get("entities") or {}))
        return {"ok": True, "classification": classification, "scope": scope, "reason": reason, "when": when, "then": then,
                "classes": list(CLASSES), "scopes": list(SCOPES)}

    def correction_save(self, what_was_wrong: str, *, receipt_id: str = "", classification: str = "",
                        scope: str = "", original_request: str = "", rerun: bool = False, category: str = "") -> dict[str, Any]:
        """Store a trusted owner correction.  Reached only from the owner's UI.

        ``category`` is the owner's own word for what went wrong (MISHEARD,
        WRONG_INTENT, WRONG_TARGET, WRONG_RESULT, INCOMPLETE, PRONUNCIATION,
        OTHER); it decides which system learns.  A MISHEARD correction becomes
        a bounded vocabulary rule for the recogniser and never a global
        replacement; the protected personality and policy are never touched.
        """

        from service.corrections import CLASSES, OWNER_CATEGORIES, SCOPES, OwnerCorrection, heard_meant_pair, pronunciation_pair, rule_for

        if not what_was_wrong.strip():
            return {"ok": False, "error": "say what was wrong"}
        category = str(category or "").upper()
        if category in OWNER_CATEGORIES and classification not in CLASSES:
            classification = OWNER_CATEGORIES[category]
        pair = pronunciation_pair(what_was_wrong) if category in {"", "PRONUNCIATION"} else None
        if pair:
            # A pronunciation correction lives in the lexicon (and is tested
            # by synthesis); it never touches the personality documents.
            result = self.pronunciation_set(pair[0], pair[1], note=what_was_wrong[:200])
            return {"ok": result.get("ok", False), "classification": "PRONUNCIATION", "scope": "GLOBAL_OWNER_PREFERENCE", **result}
        context = self.correction_context(receipt_id) if receipt_id else {}
        request = original_request or str(context.get("original_request", ""))
        if category == "MISHEARD" or classification == "STT_CORRECTION":
            heard_text = request
            with self._lock:
                for turn in reversed(self._history):
                    if turn.role == "user" and turn.meta.get("raw_transcript"):
                        heard_text = str(turn.meta.get("normalized") or turn.text)
                        break
            found = heard_meant_pair(what_was_wrong, heard_text=heard_text)
            if not found:
                return {"ok": False, "error": "say what was heard and what you meant, e.g. „Starkfisch → Stockfish“ or „ich meinte Stockfish“"}
            heard, meant = found
            learned = self.voice.vocabulary.learn(heard, meant, note=what_was_wrong[:200])
            if not learned.get("ok"):
                return {"ok": False, "error": learned.get("error", "not learned")}
            correction = OwnerCorrection(
                original_request=request, what_was_wrong=what_was_wrong.strip(), classification="STT_CORRECTION", scope="ENTITY_SPECIFIC",
                parsed_intent=str(context.get("parsed_intent", "")), entities={"heard": heard, "meant": meant}, executed_action=str(context.get("executed_action", "")),
                observed_result=str(context.get("observed_result", ""))[:500], receipt_id=receipt_id, when={"terms": [heard.lower()]},
                then={"note": f"{heard} means {meant}", "overrides": {}}, provenance="owner-ui",
            )
            self.corrections.add(correction)
            self.emit(EventType.NOTIFICATION, {"text": f"STT correction learned: „{heard}“ → „{meant}“", "kind": "owner_correction", "correction": correction.to_dict()})
            out: dict[str, Any] = {"ok": True, "correction": correction.to_dict(), "vocabulary": learned}
            if rerun and heard_text.strip():
                corrected = re.sub(r"(?<![\w'])" + re.escape(heard) + r"(?![\w'])", meant, heard_text)
                out["rerun"] = self.send_message(corrected, scope="", meta={"source": "correction_rerun", "corrected_from": heard, "meant": meant})
            return out
        guess = self.correction_classify(what_was_wrong, receipt_id=receipt_id)
        classification = classification if classification in CLASSES else guess["classification"]
        scope = scope if scope in SCOPES else guess["scope"]
        when, then = rule_for(what_was_wrong, request=request, classification=classification, scope=scope,
                              parsed_intent=str(context.get("parsed_intent", "")))
        correction = OwnerCorrection(
            original_request=request, what_was_wrong=what_was_wrong.strip(), classification=classification, scope=scope,
            parsed_intent=str(context.get("parsed_intent", "")), entities=dict(context.get("entities") or {}),
            executed_action=str(context.get("executed_action", "")), observed_result=str(context.get("observed_result", ""))[:500],
            receipt_id=receipt_id, when=when, then=then, provenance="owner-ui",
        )
        self.corrections.add(correction)
        self.think("correction")
        self.emit(EventType.NOTIFICATION, {"text": f"owner correction learned ({classification}, {scope.lower().replace('_', ' ')})",
                                           "kind": "owner_correction", "correction": correction.to_dict()})
        # A capability defect is not a memory; it is a repair to schedule.
        if classification == "CAPABILITY_DEFECT":
            self._defects[str(context.get("executed_action", "capability"))] = 2
        out: dict[str, Any] = {"ok": True, "correction": correction.to_dict()}
        # "Jetzt korrigiert erneut ausführen": the same request goes through
        # the router again, which now reads the correction first.  Reversible
        # actions only decide themselves through the planner's own guards.
        if rerun and request.strip():
            self.emit(EventType.TOOL, {"summary": f"re-running corrected: {request[:100]}", "source": "corrections",
                                       "correction_id": correction.correction_id})
            out["rerun"] = self.send_message(request, scope="")
        return out

    def list_corrections(self) -> dict[str, Any]:
        return {"corrections": [c.to_dict() for c in self.corrections.list(include_inactive=True)]}

    def update_correction(self, correction_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {k: v for k, v in changes.items() if k in {"what_was_wrong", "classification", "scope", "active", "then", "when"}}
        row = self.corrections.update(correction_id, **allowed)
        return {"ok": row is not None, "correction": row.to_dict() if row else None}

    def delete_correction(self, correction_id: str) -> dict[str, Any]:
        return {"ok": self.corrections.delete(correction_id)}

    # ------------------------------------------------------------------
    # Owner core
    # ------------------------------------------------------------------

    @property
    def owner(self) -> Any:
        from owner.core import current as owner_core

        return owner_core()

    def owner_view(self) -> dict[str, Any]:
        owner = self.owner
        return {
            "documents": owner.read_all(),
            "pending": owner.pending(),
            "history": owner.history(limit=20),
            "protected_paths": list(__import__("owner.protected", fromlist=["PROTECTED_PATHS"]).PROTECTED_PATHS),
        }

    # ------------------------------------------------------------------
    # Proactive thoughts
    # ------------------------------------------------------------------

    @property
    def adaptation(self) -> Any:
        """The Adaptive Owner Model: scoped, bounded, deletable learning from feedback."""

        if getattr(self, "_adaptation", None) is None:
            from runtime.adaptation import AdaptiveOwnerModel

            def insight(kind: str, payload: dict[str, Any]) -> None:
                self.emit(EventType.NOTIFICATION, {"kind": "adaptation_insight", "text": str(payload.get("summary", ""))[:200],
                                                   "insight_kind": kind, "detail": {k: v for k, v in payload.items() if k != "summary"}})

            self._adaptation = AdaptiveOwnerModel(Path(self.kernel.state_root) / "owner" / "adaptive.json", on_insight=insight)
        return self._adaptation

    @property
    def fs(self) -> Any:
        """The filesystem index for the File Galaxy: staged, cached, watched."""

        if getattr(self, "_fs", None) is None:
            from service.filesystem import FilesystemIndex

            self._fs = FilesystemIndex(emit=lambda payload: self.emit(EventType.DIAGNOSTIC, payload),
                                       log=lambda m: self.emit(EventType.DIAGNOSTIC, {"fs_log": m}))
        return self._fs

    @property
    def apps(self) -> Any:
        """Deterministic app launching (Start Menu + Store apps, cached index)."""

        if getattr(self, "_apps", None) is None:
            from service.apps import AppLauncher

            self._apps = AppLauncher(cache_path=Path(self.kernel.state_root) / "apps_index.json")
        return self._apps

    @property
    def jobs(self) -> Any:
        """The live WorkItem board: what runs NOW, its phase, its result."""

        if getattr(self, "_jobs", None) is None:
            from service.jobs import JobBoard

            self._jobs = JobBoard(Path(self.kernel.state_root) / "jobs.jsonl",
                                  emit=lambda payload: self.emit(EventType.JOB, payload))
        return self._jobs

    @property
    def defaults(self) -> Any:
        """Owner creation defaults: output folders and naming templates."""

        if getattr(self, "_defaults", None) is None:
            from service.defaults import CreationDefaults

            self._defaults = CreationDefaults(Path(self.kernel.state_root) / "owner" / "defaults.json")
        return self._defaults

    @property
    def imagegen(self) -> Any:
        """Local image generation (persistent SD-Turbo worker in its own venv)."""

        if getattr(self, "_imagegen", None) is None:
            from service.imagegen import ImageGenerator

            self._imagegen = ImageGenerator()
        return self._imagegen

    # -- GPU arbitration: the conversation must survive image generation --

    def _hold_for_gpu(self, text: str, scope: str) -> bool:
        """True = the request was parked because the image model holds the GPU.

        Deterministic paths (time, calendar, app/web open …) never come here —
        only paths that need FAST_LOCAL.  The parked request re-runs the moment
        the model is restored; the owner is told, not timed out.
        """

        if not getattr(self, "_gpu_hold", None):
            return False
        if not hasattr(self, "_gpu_queue_lock"):
            self._gpu_queue_lock = threading.Lock()
            self._gpu_queue = []
        de = self.language.startswith("de")
        with self._gpu_queue_lock:
            self._gpu_queue.append({"text": text, "scope": scope})
        self.emit(EventType.TOOL, {"summary": f"gpu hold: request queued ({text[:60]})", "source": "jobs"}, scope=scope)
        self._deliver(("Das Bildmodell nutzt gerade die Grafikkarte. Deine Frage ist vorgemerkt – ich beantworte sie direkt danach." if de
                       else "The image model is using the GPU right now. Your question is queued – I will answer it right after."),
                      scope=scope, backend="jobs", final_state=JarvisState.WAITING,
                      context_text="[request queued during image GPU window]")
        return True

    def _release_gpu_hold(self) -> None:
        self._gpu_hold = None
        if not hasattr(self, "_gpu_queue_lock"):
            self._gpu_queue_lock = threading.Lock()
            self._gpu_queue = []
        with self._gpu_queue_lock:
            queued, self._gpu_queue = list(self._gpu_queue), []
        for item in queued:
            try:
                self.send_message(item["text"], scope=item.get("scope", ""),
                                  meta={"source": "correction_rerun", "gpu_requeued": True})
            except Exception:  # noqa: BLE001
                pass

    def _answer_image_generate(self, prompt: str, scope: str) -> None:
        """Generate a real image locally, as a visible Job, with an immediate ack.

        The pipeline that fixes the live "25 s became minutes" failure:
        immediate acknowledgment → job with phases → FAST_LOCAL evicted ONLY
        for the GPU window when VRAM demands it → generation via the
        persistent worker → result surfaced in ZEUS (chat + notification with
        thumbnail) → FAST_LOCAL restored IMMEDIATELY → queued questions run.
        """

        de = self.language.startswith("de")
        ready = self.imagegen.available()
        if not ready.get("ok"):
            self._deliver((f"Bildgenerierung ist noch nicht eingerichtet: {ready.get('error')}" if de
                           else f"Image generation is not set up: {ready.get('error')}"),
                          scope=scope, backend="imagegen", final_state=JarvisState.ERROR)
            return
        if self.imagegen.busy:
            self._deliver("Es läuft gerade schon eine Bildgenerierung — gleich danach." if de
                          else "A generation is already running — right after it.", scope=scope, backend="imagegen")
            return

        from service.imagegen import PHASE_LABELS

        job = self.jobs.create(f"Bild: {prompt[:60]}", kind="image", scope=scope, cancellable=True,
                               phase="in der Warteschlange")
        cold = not self.imagegen.model_loaded
        ack = (("Bin dran — ich erzeuge das Bild." + (" Beim ersten Mal lädt das Modell noch, das dauert etwas länger." if cold else ""))
               if de else "On it — generating the image." + (" First run loads the model, that takes a bit longer." if cold else ""))
        self._deliver(ack, scope=scope, backend="imagegen", final_state=JarvisState.WORKING,
                      context_text=f"[image job {job.job_id} acknowledged]")

        def work() -> None:
            t_request = time.time()
            evicted = False
            restore_started = 0.0

            def on_phase(phase: str, at: float) -> None:
                nonlocal evicted
                self.jobs.phase(job.job_id, PHASE_LABELS.get(phase, phase),
                                progress={"loading_model": 0.15, "to_gpu": 0.35, "generating": 0.55,
                                          "saving": 0.9, "to_cpu": 0.95}.get(phase))
                self.state.set(JarvisState.WORKING, detail=f"Bild: {PHASE_LABELS.get(phase, phase)}", scope=scope)
                if phase == "to_gpu":
                    # the GPU window opens NOW: decide about eviction here,
                    # not minutes earlier during the model's CPU load
                    try:
                        from service.imagegen import NEEDED_VRAM_MIB

                        usage = self.gpu_usage()
                        total = int(usage.get("memory_total_mib") or 0)
                        free = total - int(usage.get("memory_used_mib") or 0) if total else 0
                        if free and free < NEEDED_VRAM_MIB:
                            from brain.tiers import ModelTier

                            provider = self.kernel.provider(ModelTier.FAST_LOCAL)
                            if hasattr(provider, "unload"):
                                provider.unload()
                                evicted = True
                                self._gpu_hold = job.job_id
                                self.emit(EventType.TOOL,
                                          {"summary": f"image: {free} MiB frei < {NEEDED_VRAM_MIB} — FAST_LOCAL für das GPU-Fenster entladen",
                                           "source": "imagegen"}, scope=scope)
                    except Exception:  # noqa: BLE001
                        pass

            self.jobs.phase(job.job_id, "Modell wird vorbereitet", state="WAITING_FOR_RESOURCE")
            result = self.imagegen.generate(
                prompt,
                output_dir=self.defaults.get("image_dir"),
                name_template=self.defaults.get("image_name"),
                on_phase=on_phase,
                cancel_check=lambda: self.jobs.cancelled(job.job_id),
            )
            t_generated = time.time()

            # restore the conversation model the MOMENT the GPU is free — the
            # owner's next question must not pay for our housekeeping
            if evicted:
                def restore() -> None:
                    nonlocal restore_started
                    restore_started = time.time()
                    try:
                        from brain.tiers import ModelTier

                        provider = self.kernel.provider(ModelTier.FAST_LOCAL)
                        provider.generate("OK", max_tokens=2)
                        self.emit(EventType.TOOL, {"summary": f"FAST_LOCAL restored in {time.time() - restore_started:.1f}s",
                                                   "source": "imagegen"}, scope=scope)
                    except Exception as exc:  # noqa: BLE001
                        self.emit(EventType.DIAGNOSTIC, {"imagegen": f"restore failed: {exc}"}, scope=scope)
                    finally:
                        self._release_gpu_hold()

                threading.Thread(target=restore, daemon=True, name="fastlocal-restore").start()
            else:
                self._release_gpu_hold()

            self.emit(EventType.TOOL, {"summary": (f"image.generate ok: {result.get('file')} "
                                                   f"({(result.get('timings') or {}).get('total', '?')}s, {result.get('vram_peak_mib')} MiB peak)")
                                                  if result.get("ok") else f"image.generate failed: {result.get('error', '')[:160]}",
                                       "result": result, "job_id": job.job_id, "source": "imagegen"}, scope=scope)
            if result.get("cancelled") or self.jobs.cancelled(job.job_id):
                # cancelled mid-flight: the file (if any) stays on disk, but
                # nothing is announced as if it had been asked for
                self.jobs.fail(job.job_id, "abgebrochen")
                self.emit(EventType.TOOL, {"summary": f"image job {job.job_id} cancelled; result suppressed",
                                           "source": "imagegen"}, scope=scope)
                return
            if not result.get("ok"):
                self.jobs.fail(job.job_id, str(result.get("error", ""))[:300])
                self._deliver((f"Das Bild ist nicht entstanden: {result.get('error', '')[:200]}" if de
                               else f"The image was not created: {result.get('error', '')[:200]}"),
                              scope=scope, backend="imagegen", final_state=JarvisState.ERROR)
                return
            timings = dict(result.get("timings") or {})
            timings["request_to_file_seconds"] = round(t_generated - t_request, 1)
            # worker timings are phase START offsets; the diffusion itself is
            # the distance from "generating" to "saving"
            if "saving" in timings and "generating" in timings:
                timings["generation_seconds"] = round(timings["saving"] - timings["generating"], 1)
            self.jobs.complete(job.job_id, {"file": result.get("file"), "bytes": result.get("bytes"),
                                            "seed": result.get("seed"), "model": result.get("model"),
                                            "width": result.get("width"), "height": result.get("height"),
                                            "vram_peak_mib": result.get("vram_peak_mib"), "timings": timings})
            # the result surfaces INSIDE ZEUS: a notification the UI renders as
            # a thumbnail card (served through /api/image/file), plus the answer
            gen_s = timings.get("generation_seconds", timings.get("total", "?"))
            self.emit(EventType.NOTIFICATION, {"kind": "image", "file": result.get("file"),
                                               "job_id": job.job_id, "title": f"Bild fertig ({gen_s}s)",
                                               "text": "Bild fertig."}, scope=scope)
            self._deliver((f"Bild fertig: {Path(str(result.get('file'))).name} — Generierung {gen_s}s, gespeichert unter {result.get('file')}." if de
                           else f"Image ready: {Path(str(result.get('file'))).name} — generation {gen_s}s, saved at {result.get('file')}."),
                          scope=scope, backend="imagegen",
                          context_text=f"[image generated: {result.get('file')}; job {job.job_id}]")

        threading.Thread(target=work, daemon=True, name="image-generate").start()

    @property
    def speech_corpus(self) -> Any:
        """The owner speech corpus: verified recordings for STT evaluation."""

        if getattr(self, "_speech_corpus", None) is None:
            from speech.corpus import SpeechCorpus

            self._speech_corpus = SpeechCorpus(Path(self.kernel.state_root) / "speech_corpus")
        return self._speech_corpus

    def corpus_benchmark(self, *, models: str = "small", limit: int = 0, held_out_only: bool = False) -> dict[str, Any]:
        """Run the STT benchmark over the owner corpus in the speech venv, async."""

        from speech.engine import venv_python

        python = venv_python()
        if python is None:
            return {"ok": False, "error": "no speech virtualenv (.venv-speech) — the benchmark needs faster-whisper"}
        if not self.speech_corpus.list():
            return {"ok": False, "error": "Der Korpus ist leer — erst Sätze aufnehmen (Voice Studio → Spracherkennung trainieren)."}
        if getattr(self, "_benchmark_running", False):
            return {"ok": False, "error": "a benchmark is already running"}
        out = Path(self.kernel.state_root) / "speech_corpus" / f"benchmark_{time.strftime('%Y%m%dT%H%M%S')}.json"
        command = [str(python), "-m", "speech.benchmark", "--corpus", str(self.speech_corpus.root),
                   "--models", models, "--out", str(out)]
        if limit:
            command += ["--limit", str(int(limit))]
        if held_out_only:
            command += ["--held-out-only"]
        self._benchmark_running = True

        def work() -> None:
            try:
                root = str(Path(__file__).resolve().parent.parent)
                completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                           cwd=root, timeout=3600,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                ok = completed.returncode == 0 and out.is_file()
                detail = (completed.stdout or completed.stderr or "")[-400:]
                self.emit(EventType.NOTIFICATION,
                          {"kind": "stt_benchmark", "ok": ok, "path": str(out) if ok else "",
                           "text": ("STT-Benchmark fertig: " + out.name) if ok else f"STT-Benchmark fehlgeschlagen: {detail[:200]}"})
            except Exception as exc:  # noqa: BLE001
                self.emit(EventType.NOTIFICATION, {"kind": "stt_benchmark", "ok": False,
                                                   "text": f"STT-Benchmark fehlgeschlagen: {exc}"})
            finally:
                self._benchmark_running = False

        threading.Thread(target=work, daemon=True, name="stt-benchmark").start()
        return {"ok": True, "started": True, "out": str(out), "models": models}

    def corpus_reports(self) -> dict[str, Any]:
        reports = []
        try:
            for path in sorted((Path(self.kernel.state_root) / "speech_corpus").glob("benchmark_*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    reports.append({"path": str(path), "started_at": data.get("started_at"),
                                    "utterances": data.get("utterances"),
                                    "summaries": [r.get("summary") for r in data.get("results", [])]})
                except (OSError, ValueError):
                    continue
        except OSError:
            pass
        return {"ok": True, "reports": reports, "running": bool(getattr(self, "_benchmark_running", False))}

    @property
    def calendar(self) -> Any:
        """The local-first calendar: real persisted events, .ics in and out."""

        if getattr(self, "_calendar", None) is None:
            from service.calendar import CalendarStore

            self._calendar = CalendarStore(Path(self.kernel.state_root) / "calendar" / "events.json")
        return self._calendar

    @property
    def aliases(self) -> Any:
        """Owner-taught names: "Uni-Planer" → a real path, app, url or project."""

        if getattr(self, "_aliases", None) is None:
            from service.aliases import AliasStore

            self._aliases = AliasStore(Path(self.kernel.state_root) / "owner" / "aliases.json")
        return self._aliases

    @property
    def semantic(self) -> Any:
        """The semantic control plane: FAST_LOCAL turns words into one typed goal."""

        if getattr(self, "_semantic", None) is None:
            from service.semantic import SemanticPlanner

            self._semantic = SemanticPlanner()
        return self._semantic

    @property
    def library(self) -> Any:
        """The knowledge library: real folders and files under one owner-visible root."""

        if getattr(self, "_library", None) is None:
            from service.library import Library

            self._library = Library()
        return self._library

    @property
    def observer(self) -> Any:
        """Opt-in desktop observation: foreground app + title, never pixels."""

        if getattr(self, "_observer", None) is None:
            from service.observer import DesktopObserver

            self._observer = DesktopObserver(Path(self.kernel.state_root))
        return self._observer

    @property
    def security(self) -> Any:
        """The Owner Security Gate.  Deterministic code; no model ever touches it."""

        if getattr(self, "_security", None) is None:
            from owner.security_gate import SecurityGate

            root = Path(getattr(self.kernel, "state_root", "") or "data/jarvis")
            self._security = SecurityGate(root / "owner" / "auth.json")
        return self._security

    def auth_status(self) -> dict[str, Any]:
        return {"ok": True, **self.security.status()}

    def auth_setup(self, password: str, *, current: str = "") -> dict[str, Any]:
        # the password strings live only in this call frame; nothing is emitted
        return self.security.setup(password, current=current)

    def auth_unlock(self, password: str, scope: str, *, seconds: float = 0.0) -> dict[str, Any]:
        out = self.security.unlock(password, scope, seconds=seconds or None)
        if out.get("ok"):
            self.emit(EventType.NOTIFICATION, {"kind": "auth", "text": f"Freigegeben: {scope} für {int(out['expires_in'])}s", "scope": scope})
        else:
            self.emit(EventType.NOTIFICATION, {"kind": "auth", "text": f"Freigabe verweigert ({scope})", "scope": scope})
        return out

    def auth_lock(self, scope: str = "") -> dict[str, Any]:
        dropped = self.security.lock(scope)
        return {"ok": True, "locked": dropped}

    def require_auth(self, authorization: str, scope: str) -> dict[str, Any] | None:
        """None when authorized; otherwise the standard needs_auth answer the UI understands."""

        if self.security.authorized(authorization, scope):
            return None
        from owner.security_gate import SCOPE_LEVELS

        return {"ok": False, "needs_auth": scope, "level": SCOPE_LEVELS.get(scope, 2),
                "error": "Das ist eine geschützte Änderung – bitte mit deinem Passwort freigeben.",
                "configured": self.security.configured}

    # ------------------------------------------------------------------
    # Feedback: 👍/👎 and owner verdicts, into the adaptive model
    # ------------------------------------------------------------------

    def feedback(self, kind: str, *, rating: str = "", category: str = "", text: str = "", request_id: str = "",
                 receipt_id: str = "", session: str = "") -> dict[str, Any]:
        from runtime.adaptation import classify_context

        if kind == "response":
            request_text, answer_text, backend = "", "", ""
            with self._lock:
                turns = list(self._history)
            for index, turn in enumerate(turns):
                if turn.role == "user" and (not request_id or turn.meta.get("request_id") == request_id):
                    request_text = turn.text
                    for later in turns[index + 1:]:
                        if later.role == "assistant":
                            answer_text, backend = later.text, later.backend
                            break
            context = classify_context(request=request_text, answer=answer_text, backend=backend)
            out = self.adaptation.record_response_feedback(rating=rating, category=category, text=text, context=context,
                                                           request=request_text, answer=answer_text, request_id=request_id)
            learned = out.get("learned") or []
            self.emit(EventType.NOTIFICATION, {"kind": "feedback", "text": f"Feedback: {'👍' if rating == 'up' else '👎'}"
                                               + (f" · {category}" if category else ""), "rating": rating, "category": category,
                                               "request_id": request_id, "context": context,
                                               "learned": [r.get("text") for r in learned]})
            return {"ok": True, "context": context, "learned": learned}
        if kind == "action":
            receipt = self.receipts.get(receipt_id) if receipt_id else None
            action_kind = getattr(receipt, "kind", "") or category or "action"
            verdict = category or "RESULT_WAS_SUCCESSFUL"
            out = self.adaptation.record_action_feedback(kind=action_kind, verdict=verdict, receipt_id=receipt_id,
                                                         request=getattr(receipt, "request", ""), detail=text)
            self.emit(EventType.NOTIFICATION, {"kind": "feedback", "text": f"Aktions-Feedback: {verdict} für {action_kind}",
                                               "receipt_id": receipt_id, "verdict": verdict, "insight": out.get("insight")})
            return out
        if kind == "wake":
            verdict = category or ("WAKE_CORRECT" if rating == "up" else "WAKE_WRONG")
            out = self.adaptation.record_action_feedback(kind="wake", verdict=verdict, receipt_id=session, detail=text, threshold=5)
            self.emit(EventType.NOTIFICATION, {"kind": "feedback", "text": f"Wake-Feedback: {verdict}", "session": session})
            return out
        return {"ok": False, "error": f"unknown feedback kind {kind!r}"}

    def adaptation_rules(self) -> dict[str, Any]:
        return {"ok": True, "rules": self.adaptation.list_rules(), "stats": self.adaptation.stats(),
                "feedback": self.adaptation.feedback_log[-50:]}

    def adaptation_rule(self, *, rule_id: str = "", action: str = "update", text: str = "", domain: str = "STYLE",
                        scope: dict[str, str] | None = None, changes: dict[str, Any] | None = None) -> dict[str, Any]:
        if action == "add":
            rule = self.adaptation.add_owner_rule(text, domain=domain, scope=scope)
            return {"ok": True, "rule": rule.to_dict()}
        if action == "delete":
            return {"ok": self.adaptation.delete_rule(rule_id)}
        rule = self.adaptation.update_rule(rule_id, changes or {})
        return {"ok": rule is not None, "rule": rule.to_dict() if rule else None}

    @property
    def thoughts(self) -> Any:
        from runtime.thoughts import ThoughtEngine, ThoughtStore

        if getattr(self, "_thoughts", None) is None:
            store = ThoughtStore(Path(self.kernel.state_root) / "thoughts.json")

            def facts() -> dict[str, Any]:
                out: dict[str, Any] = {}
                try:
                    out["missions"] = self.list_missions()["missions"]
                except Exception:  # noqa: BLE001
                    out["missions"] = []
                try:
                    out["projects"] = self.list_projects()
                except Exception:  # noqa: BLE001
                    out["projects"] = []
                try:
                    out["corrections"] = self.list_corrections().get("corrections", [])
                except Exception:  # noqa: BLE001
                    out["corrections"] = []
                try:
                    out["capabilities"] = self.list_capabilities()
                except Exception:  # noqa: BLE001
                    out["capabilities"] = []
                return out

            def proactivity() -> int:
                try:
                    return int(self.owner.read("personality").get("preferences", {}).get("proactivity", 50))
                except Exception:  # noqa: BLE001
                    return 50

            def emit(kind: str, payload: dict[str, Any]) -> None:
                self.emit(EventType.NOTIFICATION if kind == "notification" else EventType.DIAGNOSTIC, payload)

            self._thoughts = ThoughtEngine(store, facts=facts, language=lambda: self.language or "de", proactivity=proactivity, emit=emit)
        return self._thoughts

    def think(self, trigger: str = "manual", *, force: bool = False, background: bool = True) -> dict[str, Any]:
        """Run the detectors now (cheap: no model), in the background unless asked otherwise."""

        if not background:
            return self.thoughts.tick(trigger, force=force)
        threading.Thread(target=lambda: self.thoughts.tick(trigger, force=force), daemon=True, name="zeus-think").start()
        return {"scheduled": True, "trigger": trigger}

    def _schedule_idle_thinking(self, minutes: float = 30.0) -> None:
        def run() -> None:
            try:
                self.thoughts.tick("idle")
            finally:
                self._schedule_idle_thinking(minutes)

        timer = threading.Timer(minutes * 60, run)
        timer.daemon = True
        timer.start()
        self._idle_think_timer = timer

    def _say_pending_thought(self, scope: str) -> None:
        """After an answer: one thought worth saying, once, if the owner's dial allows it."""

        try:
            thought = self.thoughts.next_to_say()
        except Exception:  # noqa: BLE001
            return
        if thought is None:
            return
        de = (self.language or "de").startswith("de")
        text = (f"Eine Sache noch: {thought.title}. {thought.text}" if de else f"One more thing: {thought.title}. {thought.text}")
        self.thoughts.store.mark_delivered(thought.thought_id, "spoken")
        if self._voice is not None and self._voice.settings.enabled and self._voice.settings.speak_replies:
            try:
                self.voice.speak_stream([text], scope=scope)
            except Exception:  # noqa: BLE001
                pass
        # A thought is ZEUS's, never the owner's: the turn says so.
        self._deliver(text, scope=scope, backend="thoughts", context_text=f"[thought {thought.thought_id}]",
                      meta={"source": "zeus_thought", "thought_id": thought.thought_id, "importance": thought.importance})

    def list_thoughts(self, status: str = "") -> dict[str, Any]:
        store = self.thoughts.store
        return {"ok": True, "thoughts": [t.to_dict() for t in store.list(status)], "counts": store.counts(),
                "muted_types": sorted(store.muted_types), "dismissed_by_type": store.dismissed_by_type, "runs": self.thoughts.runs[-5:]}

    def thought_action(self, thought_id: str, action: str) -> dict[str, Any]:
        store = self.thoughts.store
        if action == "mute_type":
            thought = store.get(thought_id)
            if thought is None:
                return {"ok": False, "error": f"no thought {thought_id}"}
            store.mute(thought.type, True)
            return {"ok": True, "muted": thought.type}
        thought = store.get(thought_id)
        if thought is None:
            return {"ok": False, "error": f"no thought {thought_id}"}
        if action == "dismiss":
            store.set_status(thought_id, "DISMISSED")
            return {"ok": True, "status": "DISMISSED"}
        if action == "acted":
            store.set_status(thought_id, "ACTED_ON")
            return {"ok": True, "status": "ACTED_ON"}
        if action == "save_knowledge":
            result = self.knowledge_create(thought.title, f"{thought.text}\n\nWhy it matters: {thought.why_it_matters}",
                                           type="verified_lesson" if thought.type in {"INSIGHT", "OPTIMIZATION"} else "note",
                                           tags=[thought.type.lower(), "thought"], links=[{"target": "ZEUS", "relation": "concerns"}],
                                           provenance="zeus thought", metadata={"thought_id": thought_id, "evidence": thought.evidence})
            if result.get("ok"):
                store.set_status(thought_id, "SAVED")
            return {"ok": bool(result.get("ok")), "knowledge": result}
        if action == "attach_project":
            project_id = str(thought.context.get("project_id") or (thought.context.get("project_ids") or [""])[0])
            if not project_id:
                return {"ok": False, "error": "this thought names no project"}
            try:
                project = self.kernel.projects.load(project_id) if hasattr(self.kernel.projects, "load") else None
                if project is None:
                    return {"ok": False, "error": f"no project {project_id}"}
                notes = list(project.metadata.get("zeus_notes", []))
                notes.append({"thought_id": thought_id, "title": thought.title, "text": thought.text, "at": thought.generated_at})
                project.metadata["zeus_notes"] = notes[-20:]
                self.kernel.projects.save(project)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
            store.set_status(thought_id, "SAVED")
            return {"ok": True, "project_id": project_id}
        if action == "create_mission":
            goal = f"{thought.title}: {thought.suggested_action or thought.text}"
            mission = self.missions.create(goal, kind="complex", interpretation=f"from thought {thought_id}", links={"thought_id": thought_id})
            store.set_status(thought_id, "ACTED_ON")
            return {"ok": True, "mission_id": mission.mission_id}
        if action == "tell_me_more":
            # The owner pressed a button; the words are ZEUS's own.  The turn
            # carries that provenance so the interface never renders it as
            # something the owner said.
            self.send_message(f"Erklär mir deinen Gedanken „{thought.title}“ genauer: {thought.text} Belege: " +
                              "; ".join(e.get("summary", "") for e in thought.evidence[:4]),
                              meta={"source": "thought_inbox", "thought_id": thought_id})
            return {"ok": True, "asked": True}
        return {"ok": False, "error": f"unknown action {action}"}

    def owner_personality(self) -> dict[str, Any]:
        """The effective personality: protected core, owner dials, the prompt blocks in order, and its history."""

        try:
            effective = self.owner.effective_personality()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        from owner.core import DEFAULTS

        history = [h for h in self.owner.history() if "personality" in (h.get("documents") or [])]
        return {"ok": True, **effective, "defaults": DEFAULTS["personality"], "history": history[-20:],
                "prompt": "\n\n".join(text for _n, text in effective["blocks"])}

    def owner_propose(self, changes: dict[str, Any], *, reason: str = "", origin: str = "ui", unlock_core: bool = False,
                      authorization: str = "") -> dict[str, Any]:
        # Touching the protected core is a Level-2 change: once the owner has
        # set a password, it takes a scoped PERSONALITY_EDIT authorization
        # minted by the security gate -- no model output can substitute.
        if unlock_core and self.security.configured:
            denied = self.require_auth(authorization, "PERSONALITY_EDIT")
            if denied is not None:
                return denied
        try:
            transaction = self.owner.propose(changes, reason=reason, origin=origin, unlock_core=bool(unlock_core and origin == "ui"))
        except PermissionError as exc:
            return {"ok": False, "error": str(exc), "protected": True}
        self.emit(EventType.NOTIFICATION, {"text": f"owner change proposed: {reason or transaction.transaction_id}",
                                           "kind": "owner_proposal", "transaction": transaction.to_dict()})
        return {"ok": True, "transaction": transaction.to_dict()}

    def owner_approve(self, transaction_id: str, *, confirm: bool, authorization: str = "") -> dict[str, Any]:
        if not confirm:
            return {"ok": False, "error": "explicit confirmation is required to change the owner core"}
        if self.security.configured:
            denied = self.require_auth(authorization, "PERSONALITY_EDIT")
            if denied is not None:
                return denied
        record = self.owner.approve(transaction_id, approved_by="owner-ui")
        note = self._sync_identity(record)
        self.emit(EventType.NOTIFICATION, {"text": "owner core changed" + (f" — {note}" if note else ""),
                                           "kind": "owner_change", "record": record})
        return {"ok": True, "record": record, **({"identity_note": note} if note else {})}

    def owner_reject(self, transaction_id: str) -> dict[str, Any]:
        return {"ok": self.owner.reject(transaction_id)}

    def owner_rollback(self, audit_id: str, *, confirm: bool) -> dict[str, Any]:
        if not confirm:
            return {"ok": False, "error": "explicit confirmation is required to roll back the owner core"}
        record = self.owner.rollback(audit_id, approved_by="owner-ui")
        self._sync_identity(record)
        self.emit(EventType.NOTIFICATION, {"text": "owner core rolled back", "kind": "owner_rollback", "record": record})
        return {"ok": True, "record": record}

    def _sync_identity(self, record: dict[str, Any]) -> str:
        """An approved owner change to the identity document becomes OPERATIVE.

        ``core.identity`` loads ``config/identity.json`` at boot; without this
        bridge, editing the owner identity document would be a diary entry.
        After an approved transaction that touches ``identity``, the operative
        fields are written to that file and the process-wide identity is
        swapped, so name/tagline follow immediately.  The wake word stays
        honest about hardware: the returned note says when no wake model
        exists for the new word (the old model keeps listening until then).
        """

        docs = ({d.get("document") for d in record.get("diff", [])}
                | set(record.get("documents") or []) | set(record.get("restored") or []))
        if "identity" not in docs:
            return ""
        try:
            from dataclasses import replace as _replace

            from core import identity as identity_mod

            doc = self.owner.read("identity")
            fields = {k: str(doc[k]) for k in ("product_name", "assistant_name", "wake_word", "tagline") if doc.get(k)}
            if not fields:
                return ""
            path = Path(__file__).resolve().parent.parent / "config" / "identity.json"
            current: dict[str, Any] = {}
            if path.is_file():
                try:
                    current = json.loads(path.read_text(encoding="utf-8")) or {}
                except ValueError:
                    current = {}
            current.update(fields)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(current, indent=2), encoding="utf-8")
            updated = _replace(identity_mod.current(), **fields)
            identity_mod.set_current(updated)
            self.identity = updated
            self.persona_name = updated.assistant_name
            return updated.wake_word_note()
        except Exception as exc:  # noqa: BLE001 - the transaction itself succeeded; report, don't fail
            return f"identity sync failed: {exc}"

    def _restore_missions(self) -> int:
        """Count the missions a restart left resumable, and say so.

        Acquisition missions checkpoint after every attempt and resume when
        the same goal is asked again; the count is what the health report and
        the owner see. Kicking them off unprompted would put a 40-minute
        BUILD_LOCAL run on the GPU the instant the conversation model loaded,
        which is the wrong order on one card.
        """

        from runtime.missions import MissionStore

        store = MissionStore(Path(self.kernel.state_root) / "missions")
        resumable = 0
        for checkpoint in store.list():
            if not checkpoint.acquired and not checkpoint.stale:
                resumable += 1
        if resumable:
            self.emit(EventType.NOTIFICATION, {
                "text": f"{resumable} mission(s) can be resumed", "kind": "missions",
            })
        return resumable

    #: States in which an answer is being produced for the owner (a
    #: conversation turn), as opposed to long autonomous work.
    _ANSWERING = (JarvisState.THINKING, JarvisState.SPEAKING, JarvisState.TRANSCRIBING)

    def _running_now(self) -> list[str]:
        """What a stop could actually stop right now: "speech" and/or "answer"."""

        running: list[str] = []
        voice = self._voice
        if voice is not None and getattr(voice, "_speaker", None) is not None:
            running.append("speech")
        current = self.state.snapshot.state
        if current in self._ANSWERING:
            if "speech" not in running and current is JarvisState.SPEAKING:
                running.append("speech")
            running.append("answer")
        return running

    def stop_current(self, *, reason: str = "owner", session: str = "") -> dict[str, Any]:
        """The owner's stop (Esc, the stop button): interrupt speech and the current answer.

        Idempotent and reason-coded.  It says what it stopped; when nothing
        was running it says so as a DIAGNOSTIC and produces no transcript
        entry -- the literal "stopped" that used to be posted on every wake
        word is gone.  Long autonomous work (missions) is not touched here;
        Mission Control cancels those explicitly.
        """

        stopped = self._running_now()
        self._stop_requested.set()
        if self._voice is not None:
            # Barge-in has to reach the speaker, not just the generator: the
            # audio already synthesised would otherwise keep playing over the
            # user who interrupted it.
            self._voice.interrupt()
        if stopped:
            de = self.language.startswith("de")
            text = ("Sprachausgabe gestoppt" if de else "speech stopped") if stopped == ["speech"] else ("Antwort abgebrochen" if de else "answer cancelled")
            self.emit(EventType.NOTIFICATION, {"text": text, "kind": "stop", "stopped": stopped, "reason": reason, "session": session})
            self.state.set(JarvisState.IDLE, detail="interrupted")
        else:
            self.emit(EventType.DIAGNOSTIC, {"stop": "nothing running", "reason": reason, "session": session})
        return {"ok": True, "stopped": stopped}

    def voice_interrupt(self, *, session: str = "", wake: float = 0.0) -> dict[str, Any]:
        """Barge-in from the listener: the wake word fired while ZEUS may be talking.

        Interrupts speech and a conversation answer in progress -- and only
        those.  With nothing running it is a no-op that leaves no trace in
        the transcript; it never touches the listening session that the same
        wake word just opened, which stays LISTENING on the device.
        """

        running = self._running_now()
        # Audio already handed to the client may still be playing after the
        # speaker object is gone: the estimate says so, and the listener
        # needs to know its recording may contain ZEUS's own voice.
        speaking = bool(self._voice is not None and self._voice.speaking)
        if not running:
            if speaking and self._voice is not None:
                self._voice.interrupt()
                self.emit(EventType.DIAGNOSTIC, {"voice_interrupt": "playback stopped", "session": session, "wake": wake})
                return {"ok": True, "interrupted": ["speech"], "speaking": True}
            self.emit(EventType.DIAGNOSTIC, {"voice_interrupt": "nothing to interrupt", "session": session, "wake": wake})
            return {"ok": True, "interrupted": [], "speaking": False}
        self._stop_requested.set()
        if self._voice is not None:
            self._voice.interrupt()
        de = self.language.startswith("de")
        self.emit(EventType.NOTIFICATION, {"text": ("Unterbrochen — ich höre zu." if de else "Interrupted — listening."), "kind": "barge_in",
                                           "stopped": running, "session": session, "wake": wake})
        self.state.set(JarvisState.LISTENING, detail=f"barge-in {session}".strip())
        return {"ok": True, "interrupted": running, "speaking": speaking}

    #: The listener's session states, mirrored into the core's state so the
    #: interface (the eye) shows LISTENING while the device is armed.
    _SESSION_STATES = {"WAKE_DETECTED", "LISTENING", "CAPTURING", "UTTERANCE_CAPTURED", "SENT", "IDLE"}

    def voice_session_event(self, session: str, state: str, reason: str = "", *, wake: float = 0.0) -> dict[str, Any]:
        """One reason-coded transition of a listening session on the device."""

        state = str(state or "").upper()
        if state not in self._SESSION_STATES:
            return {"ok": False, "error": f"unknown session state {state!r}"}
        self.emit(EventType.DIAGNOSTIC, {"voice_session": state, "session": session, "reason": reason, "wake": wake})
        current = self.state.snapshot.state
        if state in {"WAKE_DETECTED", "LISTENING", "CAPTURING"} and current in (JarvisState.IDLE, JarvisState.LISTENING):
            self.state.set(JarvisState.LISTENING, detail=f"{state.lower()} {session}")
        elif state == "IDLE" and current is JarvisState.LISTENING:
            self.state.set(JarvisState.IDLE, detail=reason[:60])
        return {"ok": True, "session": session, "state": state}

    def new_conversation(self) -> dict[str, Any]:
        """Start fresh: clear the transcript and settle the state.

        The *conversation* resets; the *record* does not.  The activity log and
        the receipt ledger are untouched, so a failed action the user just
        cleared off their screen is still there for anyone who goes looking.
        Hiding the transcript is a convenience; hiding the evidence would be
        the thing this whole system exists to prevent.
        """

        archived = None
        try:
            with self._lock:
                turns = [t.to_dict() for t in self._history]
            archived = self.conversations.archive(turns, language=self.language, reason="new_conversation")
        except Exception:  # noqa: BLE001 - archiving must never block a fresh start
            archived = None
        with self._lock:
            self._history.clear()
        self._session_receipts.clear()
        self.language = ""
        self.state.set(JarvisState.IDLE)
        if archived:
            self._summarize_conversation_async(archived["id"])
        return {"ok": True, "cleared": True, "archived": archived["id"] if archived else ""}

    @property
    def coach(self) -> Any:
        """The language coach: sessions, learner model, spaced repetition."""

        if getattr(self, "_coach", None) is None:
            from service.coach import LanguageCoach

            self._coach = LanguageCoach(Path(self.kernel.state_root) / "coach")
        return self._coach

    _COACH_START = re.compile(r"\b(?:lass\s+uns|wir\s+(?:ueben|üben)|(?:ueben|üben)\s+wir|ich\s+will\s+.{0,20})?\s*"
                              r"(?:(?P<min>\d{1,3})\s*minuten\s+)?"
                              r"(?P<lang>franz(?:oe|ö)sisch|englisch|spanisch|italienisch|latein|french|english|spanish)\s+"
                              r"(?:ueben|üben|lernen|trainieren|practice)\b", re.I)
    _COACH_END = re.compile(r"\b((?:uebung|übung)\s+beenden|aufh(?:oe|ö)ren|training\s+beenden|stopp?\s*,?\s*(?:uebung|übung)|genug\s+ge(?:uebt|übt))\b", re.I)

    def _handle_coach(self, text: str, scope: str) -> bool:
        """Session mode: while an exercise runs, every turn goes to the coach."""

        # cheap gate first: never CONSTRUCT the coach (it touches the state
        # root) for ordinary messages — stub cores in tests have neither
        if getattr(self, "_coach", None) is None and not self._COACH_START.search(text):
            return False
        from brain.tiers import ModelTier
        from service.coach import LANGUAGES

        try:
            active = self.coach.session is not None
        except Exception:  # noqa: BLE001 - no state root, no coach
            return False
        if not active:
            m = self._COACH_START.search(text)
            if not m:
                return False
            language = LANGUAGES.get(m.group("lang").lower())
            if not language:
                return False
            minutes = int(m.group("min") or 10)
            started = self.coach.start(language, minutes=minutes)
            self.emit(EventType.TOOL, {"summary": f"coach session started: {language}, {minutes}min, Thema {started.get('topic')}",
                                       "source": "coach"}, scope=scope)
            self._deliver(started["text"], scope=scope, backend="coach",
                          context_text=f"[coach session started: {language}]")
            return True
        # a session is running: end phrases close it, everything else is a turn
        if self._COACH_END.search(text) or re.match(r"^\s*(stopp?|stop|beenden|fertig)\s*[.!]?\s*$", text, re.I):
            self._finish_coach(scope)
            return True
        self.state.set(JarvisState.THINKING, detail="bewerte deine Antwort", scope=scope)
        try:
            provider = self.kernel.provider(ModelTier.FAST_LOCAL)
            result = self.coach.evaluate_turn(provider, text)
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": str(exc)}
        if not result.get("ok"):
            self._deliver(f"Die Bewertung ist gerade nicht durchgekommen ({result.get('error', '')[:80]}) — sag den Satz einfach nochmal.",
                          scope=scope, backend="coach", final_state=JarvisState.WAITING)
            return True
        self._deliver(result["text"], scope=scope, backend="coach",
                      context_text=f"[coach turn score {result.get('score')}]")
        if result.get("done"):
            self._finish_coach(scope)
        return True

    def _finish_coach(self, scope: str) -> None:
        summary = self.coach.finish()
        if not summary.get("ok"):
            return
        try:
            self.library.create_folder("Sprachen")
            self.library.write_note("Sprachen", f"{summary['language']}-Übung {time.strftime('%Y-%m-%d %H%M')}",
                                    summary.get("summary_note", ""))
        except Exception:  # noqa: BLE001 - the summary text still reaches the owner
            pass
        self.emit(EventType.TOOL, {"summary": f"coach session finished: {summary.get('record', {})}", "source": "coach"}, scope=scope)
        self._deliver(summary["text"], scope=scope, backend="coach",
                      context_text="[coach session finished; summary saved to Wissen/Sprachen]")

    @property
    def tv(self) -> Any:
        """The paired LG webOS TV (SSAP over the LAN)."""

        if getattr(self, "_tv", None) is None:
            from service.tv import TVService

            self._tv = TVService(Path(self.kernel.state_root) / "owner" / "tv.json")
        return self._tv

    @property
    def conversations(self) -> Any:
        """The conversation archive: recent chats with summaries, persisted."""

        if getattr(self, "_conversations", None) is None:
            from service.conversations import ConversationArchive

            self._conversations = ConversationArchive(Path(self.kernel.state_root) / "conversations")
        return self._conversations

    def _summarize_conversation_async(self, conv_id: str) -> None:
        """Fill in title/summary with one FAST_LOCAL call, off the request path."""

        def work() -> None:
            try:
                from brain.json_utils import lenient_json_loads
                from brain.tiers import ModelTier
                from service.conversations import SUMMARY_PROMPT, SUMMARY_SCHEMA, transcript_text

                record = self.conversations.get(conv_id)
                if not record:
                    return
                prompt = SUMMARY_PROMPT.format(transcript=transcript_text(record.get("turns", [])))
                provider = self.kernel.provider(ModelTier.FAST_LOCAL)
                if hasattr(provider, "generate_structured"):
                    raw = provider.generate_structured(prompt, SUMMARY_SCHEMA, max_tokens=400, temperature=0.1)
                else:
                    raw = provider.generate(prompt, max_tokens=400)
                payload = lenient_json_loads(str(raw))
                if isinstance(payload, dict):
                    self.conversations.set_summary(conv_id, title=str(payload.get("title", "")),
                                                   summary=str(payload.get("summary", "")),
                                                   open_tasks=list(payload.get("open_tasks") or []),
                                                   facts=list(payload.get("facts") or []))
                    self.emit(EventType.DIAGNOSTIC, {"conversation_summarized": conv_id})
            except Exception:  # noqa: BLE001 - a missing summary is only a missing summary
                pass

        threading.Thread(target=work, daemon=True, name=f"conv-summary-{conv_id[-6:]}").start()

    def conversation_restore(self, conv_id: str) -> dict[str, Any]:
        """Bring an archived conversation back as the live transcript."""

        record = self.conversations.get(conv_id)
        if not record:
            return {"ok": False, "error": f"kein Gespräch {conv_id}"}
        # the current transcript is archived first, so nothing is lost
        try:
            with self._lock:
                current = [t.to_dict() for t in self._history]
            parked = self.conversations.archive(current, language=self.language, reason="replaced_by_restore")
            if parked:
                self._summarize_conversation_async(parked["id"])
        except Exception:  # noqa: BLE001
            pass
        turns = [ConversationTurn(role=str(t.get("role", "")), text=str(t.get("text", "")),
                                  at=str(t.get("at", "")), backend=str(t.get("backend", "")))
                 for t in record.get("turns", []) if isinstance(t, dict)]
        with self._lock:
            self._history.clear()
            self._history.extend(turns)
        self.language = str(record.get("language", "")) or self.language
        self.state.set(JarvisState.IDLE)
        return {"ok": True, "id": conv_id, "title": record.get("title"),
                "turns": [t.to_dict() for t in turns], "late_results": record.get("late_results", [])}

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    @staticmethod
    def project_origin(project: Any) -> str:
        """owner | acquisition | unclassified -- from fields the store already persists.

        A capability-acquisition project (``kind == "capability"`` or a
        ``capability_id`` in its metadata) is ZEUS's internal work: it is
        shown *under* the capability, never as an owner project.  Nothing is
        deleted; a legacy record that fits neither rule is reported as such.
        """

        kind = str(getattr(getattr(project, "kind", ""), "value", getattr(project, "kind", "")) or "").lower()
        metadata = dict(getattr(project, "metadata", {}) or {})
        if kind == "capability" or metadata.get("capability_id"):
            return "acquisition"
        if kind in {"software", "owner", "research", "project", "generic", ""}:
            return "owner"
        return "unclassified"

    def list_projects(self) -> list[dict[str, Any]]:
        try:
            projects = self.kernel.projects.list_projects()
        except Exception:
            return []
        rows = []
        for project in projects:
            metadata = dict(getattr(project, "metadata", {}) or {})
            tasks = list(getattr(project, "tasks", []))
            done = sum(1 for t in tasks if str(getattr(getattr(t, "status", ""), "value", getattr(t, "status", ""))).lower() in {"done", "complete", "completed", "accepted"})
            blocked = sum(1 for t in tasks if str(getattr(getattr(t, "status", ""), "value", getattr(t, "status", ""))).lower() in {"blocked", "failed"})
            rows.append({
                "id": project.id,
                # The title is what a user names a project and what they will
                # look for in the panel; the goal can be a paragraph.
                "title": getattr(project, "title", "") or "",
                "goal": project.goal,
                "state": getattr(project.state, "value", str(project.state)),
                "kind": str(getattr(getattr(project, "kind", ""), "value", getattr(project, "kind", "")) or ""),
                "capability_id": str(metadata.get("capability_id", "") or ""),
                "origin": self.project_origin(project),
                "tasks": len(tasks), "tasks_done": done, "tasks_blocked": blocked,
                "steps": len(getattr(project, "steps", [])),
                "created_at": getattr(project, "created_at", ""),
                "updated_at": getattr(project, "updated_at", ""),
                # owner intent about the universe: importance (PINNED FOCUS ACTIVE NORMAL LOW_PRIORITY DORMANT ARCHIVED),
                # hidden flag, and the owner's own placement of the node
                "importance": str(metadata.get("importance") or self._default_importance(project, tasks, blocked)),
                "hidden": bool(metadata.get("hidden", False)),
                "parent_id": str(metadata.get("parent_id") or ""),
                "deadline": str(metadata.get("deadline") or ""),
                "layout": dict(metadata.get("layout") or {}),
                "health": self._project_health(project, tasks, blocked, done),
                "notes": list(metadata.get("zeus_notes") or [])[-3:],
            })
        return rows

    IMPORTANCE_LEVELS = ("PINNED", "FOCUS", "ACTIVE", "NORMAL", "LOW_PRIORITY", "DORMANT", "TEST", "ARCHIVED")
    #: Hidden from the default galaxy; "show everything" reveals them.  Nothing is deleted.
    HIDDEN_IMPORTANCE = frozenset({"TEST", "ARCHIVED"})
    #: "Test", "Test Alpha", "zeus_test", "Zeus Testprojekt", "Zeus Realtest" -- legacy probes.
    #: Deliberately not every compound with "test" in it: "Sprachtest" is a project the owner
    #: creates on purpose and expects to see.
    _TEST_TITLE = re.compile(r"(^|\b)(test|tests|testing|probe|dummy|demo|sample|acceptance)(\b|_|\d)|testprojekt|zeus_test|realtest", re.I)

    @classmethod
    def _default_importance(cls, project: Any, tasks: list[Any], blocked: int) -> str:
        state = str(getattr(project.state, "value", str(project.state))).lower()
        title = str(getattr(project, "title", "") or "")
        goal = str(getattr(project, "goal", "") or "")
        # Legacy test/probe records ("Test Alpha", "zeus_test", "Zeus Testprojekt")
        # are TEST by default: kept as evidence, out of the owner's galaxy.
        if cls._TEST_TITLE.search(title) or (not title and cls._TEST_TITLE.search(goal[:40])):
            return "TEST"
        if state in {"completed", "abandoned"}:
            return "ARCHIVED" if state == "abandoned" else "DORMANT"
        try:
            updated = datetime.fromisoformat(str(getattr(project, "updated_at", "")).replace("Z", "+00:00"))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            idle_days = (datetime.now(timezone.utc) - updated).days
        except ValueError:
            idle_days = 0
        if state in {"executing", "verifying", "investigating", "planning"}:
            return "ACTIVE"
        if idle_days > 14:
            return "DORMANT"
        return "NORMAL"

    @staticmethod
    def _project_health(project: Any, tasks: list[Any], blocked: int, done: int) -> dict[str, Any]:
        """HEALTHY | AT_RISK | BLOCKED | DORMANT | COMPLETE from the real record, with the reason."""

        state = str(getattr(project.state, "value", str(project.state))).lower()
        if state == "completed" or (tasks and done == len(tasks)):
            return {"state": "COMPLETE", "reason": "all tasks done" if tasks else "completed"}
        if blocked:
            return {"state": "BLOCKED", "reason": f"{blocked} task(s) blocked or failed"}
        if state in {"blocked", "waiting_for_owner"}:
            return {"state": "BLOCKED", "reason": state}
        try:
            updated = datetime.fromisoformat(str(getattr(project, "updated_at", "")).replace("Z", "+00:00"))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            idle_days = (datetime.now(timezone.utc) - updated).days
        except ValueError:
            idle_days = 0
        if idle_days >= 7 and state not in {"draft"}:
            return {"state": "DORMANT", "reason": f"no activity for {idle_days} days"}
        if getattr(project, "last_stop_reason", ""):
            return {"state": "AT_RISK", "reason": str(getattr(project, "last_stop_reason", ""))[:80]}
        return {"state": "HEALTHY", "reason": "moving" if idle_days < 3 else "quiet"}

    def project_update(self, project_id: str, *, importance: str = "", hidden: Any = None, layout: dict[str, Any] | None = None,
                       note: str = "") -> dict[str, Any]:
        """Owner intent about a project in the universe: importance, hidden, node position (persisted in metadata)."""

        project = self.kernel.projects.load(project_id) if hasattr(self.kernel.projects, "load") else None
        if project is None:
            return {"ok": False, "error": f"no project {project_id}"}
        meta = project.metadata
        if importance:
            if importance not in self.IMPORTANCE_LEVELS:
                return {"ok": False, "error": f"importance must be one of {', '.join(self.IMPORTANCE_LEVELS)}"}
            meta["importance"] = importance
        if hidden is not None:
            meta["hidden"] = bool(hidden)
        if layout is not None:
            current = dict(meta.get("layout") or {})
            for key in ("x", "y"):
                if key in layout:
                    current[key] = float(layout[key])
            if "state" in layout and layout["state"] in {"AUTO_POSITIONED", "OWNER_POSITIONED", "LOCKED"}:
                current["state"] = layout["state"]
            if "cluster" in layout:
                current["cluster"] = str(layout["cluster"] or "")
            meta["layout"] = current
        if note:
            notes = list(meta.get("owner_notes") or [])
            notes.append({"text": str(note)[:500], "at": _now()})
            meta["owner_notes"] = notes[-20:]
        self.kernel.projects.save(project)
        self.emit(EventType.TOOL, {"summary": f"project {project.title or project.id}: " + ", ".join(
            k for k, v in (("importance " + importance, importance), ("hidden", hidden is not None), ("position", layout is not None), ("note", bool(note))) if v),
            "source": "projects"})
        return {"ok": True, "id": project.id, "importance": meta.get("importance"), "hidden": meta.get("hidden", False), "layout": meta.get("layout", {})}

    def project_timeline(self, project_id: str = "", limit: int = 200) -> dict[str, Any]:
        """Meaningful events from the activity log and the mission stores, for one project or all."""

        project = self.kernel.projects.load(project_id) if project_id and hasattr(self.kernel.projects, "load") else None
        needle = (project.title or project.goal[:40]).lower() if project is not None else ""
        events: list[dict[str, Any]] = []
        if project is not None:
            events.append({"at": project.created_at, "kind": "created", "summary": f"project created: {project.title or project.goal[:60]}", "ref": project.id})
            for d in getattr(project, "decisions", [])[-20:]:
                events.append({"at": getattr(d, "at", ""), "kind": "decision", "summary": str(getattr(d, "text", getattr(d, "summary", "")))[:160], "ref": project.id})
        for m in self.list_missions()["missions"]:
            if needle and needle not in str(m.get("goal", "")).lower() and needle not in str(m.get("title", "")).lower():
                continue
            events.append({"at": m.get("started", ""), "kind": "mission_started", "summary": m.get("title", ""), "ref": m.get("id", "")})
            if m.get("state") in {"failed", "completed", "cancelled"}:
                events.append({"at": m.get("updated", ""), "kind": f"mission_{m['state']}" if m["state"] != "completed" else ("mission_promoted" if m.get("deployment") == "promoted" else "mission_completed"),
                               "summary": m.get("title", ""), "ref": m.get("id", "")})
        try:
            for entry in self.activity.recent(limit=1500):
                if entry.kind not in {"action.verified", "action.failed", "notification"}:
                    continue
                summary = str(entry.summary or "")
                if needle and needle not in summary.lower():
                    continue
                detail = entry.detail or {}
                if entry.kind == "notification" and "correction" in summary:
                    events.append({"at": entry.at, "kind": "owner_correction", "summary": summary[:160], "ref": entry.receipt_id})
                elif entry.kind == "action.verified" and str(detail.get("kind", "")).startswith("capability"):
                    events.append({"at": entry.at, "kind": "capability_used", "summary": summary[:160], "ref": entry.receipt_id})
        except Exception:  # noqa: BLE001
            pass
        events = [e for e in events if e.get("at")]
        events.sort(key=lambda e: str(e["at"]))
        return {"ok": True, "project_id": project_id, "events": events[-limit:]}

    def project_graph(self, *, everything: bool = False) -> dict[str, Any]:
        """The universe: nodes and typed relations for the galaxy (owner projects, missions, capabilities, knowledge clusters, thoughts)."""

        overview = self.projects_overview()
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        shown = [p for p in overview["projects"] if everything or (not p.get("hidden") and p.get("importance") not in self.HIDDEN_IMPORTANCE)]
        for p in shown:
            nodes.append({"id": p["id"], "kind": "project", "label": p["title"] or p["goal"][:40], "importance": p["importance"], "health": p["health"],
                          "state": p["state"], "tasks": p["tasks"], "tasks_done": p["tasks_done"], "layout": p.get("layout", {}), "updated_at": p["updated_at"], "data": p})
        for fam in overview["internal"]:
            if not everything and fam["latest_state"].lower() in {"completed", "paused", "draft"} and fam["count"] > 1:
                pass  # collapsed by default: one node per family below
            nodes.append({"id": f"cap:{fam['capability_id']}", "kind": "capability", "label": fam["capability_id"], "attempts": fam["count"],
                          "state": fam["latest_state"], "collapsed": True, "data": fam})
        titles = {n["id"]: n["label"].lower() for n in nodes if n["kind"] == "project"}
        # Subprojects orbit their parent: an explicit owner hierarchy, never inferred from words.
        for p in shown:
            if p.get("parent_id") and p["parent_id"] in titles:
                edges.append({"source": p["parent_id"], "target": p["id"], "type": "subproject_of", "active": False})
        for m in overview["missions"]:
            if m.get("system") == "acquisition":
                continue
            parent = next((pid for pid, t in titles.items() if t and t in str(m.get("goal", "")).lower()), None)
            nodes.append({"id": m["id"], "kind": "mission", "label": m["title"], "state": m["state"], "system": m["system"], "updated_at": m.get("updated", ""), "data": m})
            edges.append({"source": parent or "zeus", "target": m["id"], "type": "mission_of", "active": m["state"] in {"active", "waiting"}})
        for n in nodes:
            if n["kind"] == "capability":
                for pid, t in titles.items():
                    if any(w in n["label"] for w in t.split() if len(w) > 4):
                        edges.append({"source": pid, "target": n["id"], "type": "uses", "active": False})
        try:
            for t in self.thoughts.store.list():
                if t.status == "DISMISSED":
                    continue
                ids = t.context.get("project_ids") or ([t.context["project_id"]] if t.context.get("project_id") else [])
                if not ids:
                    continue
                nodes.append({"id": t.thought_id, "kind": "thought", "label": t.title, "type": t.type, "importance": t.importance, "data": t.to_dict()})
                for pid in ids:
                    edges.append({"source": pid, "target": t.thought_id, "type": "thought", "active": t.status in {"NEW", "IMPORTANT"}})
        except Exception:  # noqa: BLE001
            pass
        try:
            stats = self.knowledge_stats()
            if stats.get("ok") and int(stats.get("nodes", 0)) > 0:
                nodes.append({"id": "knowledge", "kind": "knowledge", "label": f"Knowledge · {stats['nodes']} nodes", "data": stats})
                edges.append({"source": "zeus", "target": "knowledge", "type": "part_of", "active": False})
        except Exception:  # noqa: BLE001
            pass
        nodes.append({"id": "zeus", "kind": "self", "label": "ZEUS", "data": {}})
        return {"ok": True, "nodes": nodes, "edges": edges, "everything": everything, "hidden": len(overview["projects"]) - len(shown)}

    def projects_overview(self) -> dict[str, Any]:
        """Owner projects first; internal acquisition attempts grouped per capability (attempt families)."""

        rows = self.list_projects()
        owner = [r for r in rows if r["origin"] == "owner"]
        families: dict[str, dict[str, Any]] = {}
        for r in rows:
            if r["origin"] != "acquisition":
                continue
            key = r["capability_id"] or r["title"] or r["id"]
            fam = families.setdefault(key, {"capability_id": key, "attempts": [], "latest_state": "", "updated_at": ""})
            fam["attempts"].append({k: r[k] for k in ("id", "state", "tasks", "steps", "created_at", "updated_at", "title")})
            if str(r["updated_at"]) >= str(fam["updated_at"]):
                fam["updated_at"], fam["latest_state"] = r["updated_at"], r["state"]
        for fam in families.values():
            fam["attempts"].sort(key=lambda a: str(a.get("created_at", "")))
            fam["count"] = len(fam["attempts"])
        missions = self.list_missions().get("missions", [])
        return {"projects": owner, "internal": sorted(families.values(), key=lambda f: str(f["updated_at"]), reverse=True),
                "unclassified": [r for r in rows if r["origin"] == "unclassified"],
                "missions": missions[:40], "counts": {"owner": len(owner), "internal_attempts": sum(f["count"] for f in families.values()),
                                                      "families": len(families), "unclassified": len([r for r in rows if r["origin"] == "unclassified"])}}

    def project_detail(self, reference: str) -> dict[str, Any]:
        project = self.kernel.resolve_project(reference) if reference else None
        if project is None:
            return {"error": f"no project matching {reference!r}"}
        return {
            "id": project.id,
            "goal": project.goal,
            "state": getattr(project.state, "value", str(project.state)),
            "title": getattr(project, "title", "") or "",
            "created_at": getattr(project, "created_at", ""),
            "updated_at": getattr(project, "updated_at", ""),
            "metadata": {k: v for k, v in dict(getattr(project, "metadata", {}) or {}).items() if k in {"parent_id", "parent_title", "deadline", "importance", "owner_request", "renamed", "hidden"}},
            "artifacts": [{"path": a.path, "kind": a.kind, "description": a.description, "at": a.added_at} for a in getattr(project, "artifacts", [])][-20:],
            "decisions": [{"text": d.text, "rationale": d.rationale, "at": d.added_at} for d in getattr(project, "decisions", [])][-20:],
            "findings": [{"text": f.text, "source": f.source, "at": f.added_at} for f in getattr(project, "findings", [])][-20:],
            "blockers": [{"text": b.text, "needs_user": b.needs_user, "resolved": b.resolved} for b in getattr(project, "blockers", []) if not b.resolved][-20:],
            "tasks": [
                {
                    "title": task.title,
                    "status": getattr(task.status, "value", str(task.status)),
                    "attempts": getattr(task, "attempts", 0),
                }
                for task in getattr(project, "tasks", [])
            ],
            "acceptance": [
                {"text": item.text, "satisfied": item.satisfied}
                for item in getattr(project, "acceptance", [])
            ],
            "steps": [
                {
                    "phase": getattr(step.phase, "value", str(step.phase)),
                    "summary": step.summary,
                    "success": step.success,
                }
                for step in getattr(project, "steps", [])[-40:]
            ],
        }

    # ------------------------------------------------------------------
    # Capabilities and knowledge
    # ------------------------------------------------------------------

    #: The registry is a file, not a directory.  Passing the directory made
    #: ``CapabilityRegistry`` raise on read, the bare ``except`` swallowed it,
    #: and this endpoint returned ``[]`` forever -- indistinguishable from
    #: "there are no capabilities".  Every other call site in the tree already
    #: passes ``registry.json``; this one did not.
    CAPABILITY_REGISTRY = ("capabilities", "registry.json")

    def _capability_registry(self) -> Any:
        from capabilities.registry import CapabilityRegistry

        path = self.kernel.state_root.joinpath(*self.CAPABILITY_REGISTRY)
        return CapabilityRegistry(path)

    def list_capabilities(self) -> list[dict[str, Any]]:
        try:
            return [manifest.to_dict() for manifest in self._capability_registry().all()]
        except Exception:
            return []

    def capability_report(self) -> dict[str, Any]:
        """What the registry actually holds, with the failure visible.

        Separate from :meth:`list_capabilities` because that one has to keep
        returning a list for the existing endpoint, and a list has nowhere to
        put "the registry could not be read".  An answer about capabilities
        that silently degrades to "none" is the same defect as a model
        inventing them, arrived at from the other direction.
        """

        path = self.kernel.state_root.joinpath(*self.CAPABILITY_REGISTRY)
        try:
            manifests = self._capability_registry().all()
        except Exception as exc:
            return {"path": str(path), "error": f"{type(exc).__name__}: {exc}", "active": [], "disabled": []}
        records = [manifest.to_dict() for manifest in manifests]
        return {
            "path": str(path),
            "error": "",
            "active": [item for item in records if item.get("status") == "active"],
            "disabled": [item for item in records if item.get("status") != "active"],
        }

    def project_delete(self, project_id: str, *, authorization: str = "") -> dict[str, Any]:
        """Permanent deletion through the API: PROJECT_DELETE authorization when a password is set."""

        if self.security.configured:
            denied = self.require_auth(authorization, "PROJECT_DELETE")
            if denied is not None:
                return denied
        from service.intents import ActionIntent
        from service.project_ops import ProjectOperations, compose_concise

        ops = ProjectOperations(self)
        project = ops._find(project_id)
        if project is None:
            return {"ok": False, "error": f"no project {project_id!r}"}
        intent = ActionIntent("project.delete", verb="delete", object_type="project", target=project.id,
                              success_criteria=["the project no longer exists"], confidence=1.0, reason="owner-authorized deletion")
        receipt = ops.execute(intent, request=f"delete project {project.title or project.id} (owner-authorized)")
        self.receipts.record(receipt)
        self._session_receipts.append(receipt)
        self.emit(EventType.TOOL, {"summary": receipt.summary(), "receipt_id": receipt.id, "receipt": receipt.to_dict()})
        self._deliver(compose_concise(receipt, language=self.language), scope="", backend=receipt.executor)
        return {"ok": receipt.verified, "receipt": receipt.to_dict()}

    def activity_correct(self, *, request_id: str = "", seq: int = 0, correction_type: str = "TRANSCRIPT",
                         corrected_text: str = "", original_text: str = "", note: str = "", rerun: bool = False) -> dict[str, Any]:
        """Append-only owner corrections on the activity ledger.

        The original evidence is never mutated: the correction is a new
        record referencing it, the interface shows ORIGINAL -> corrected,
        and the learning systems (STT lexicon, corrections memory) are fed
        from it.  ``rerun`` re-sends the corrected text through the normal
        request path -- side effects then pass the same guards as any
        request (duplicates, confirmations, the security gate).
        """

        corrected = (corrected_text or "").strip()
        if not corrected and correction_type in {"TRANSCRIPT", "INTENT"}:
            return {"ok": False, "error": "say what it should have been"}
        entry = {"request_id": request_id, "seq": int(seq or 0), "type": correction_type, "original": original_text[:400],
                 "corrected": corrected[:400], "note": note[:300], "at": _now()}
        path = Path(self.kernel.state_root) / "owner" / "activity_corrections.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        learned: dict[str, Any] = {}
        if correction_type == "TRANSCRIPT" and original_text and corrected:
            # a transcript edit is STT ground truth: learn the heard->meant
            # pair(s) that differ, bounded and context-scoped
            from service.corrections import heard_meant_pair

            pair = heard_meant_pair(f"nicht {original_text} sondern {corrected}", heard_text=original_text)
            import re as _re

            orig_tokens = _re.findall(r"[^\W\d_]+", original_text, _re.UNICODE)
            new_tokens = _re.findall(r"[^\W\d_]+", corrected, _re.UNICODE)
            if pair and pair[0].lower() != pair[1].lower() and len(orig_tokens) == len(new_tokens):
                learned = self.voice.vocabulary.learn(pair[0], pair[1], note=f"activity edit {request_id}"[:120])
            elif len(orig_tokens) == len(new_tokens):
                for heard, meant in zip(orig_tokens, new_tokens):
                    if heard.lower() != meant.lower() and len(heard) >= 3:
                        learned = self.voice.vocabulary.learn(heard, meant, note=f"activity edit {request_id}"[:120])
                        break
            self.adaptation.record_action_feedback(kind="stt", verdict="TRANSCRIPT_CORRECTED", receipt_id=request_id,
                                                   request=original_text, detail=corrected, threshold=999)
        self.emit(EventType.NOTIFICATION, {"kind": "activity_correction", "text": f"Korrigiert ({correction_type}): „{corrected[:80]}“",
                                           "request_id": request_id, "entry": entry, "learned": learned})
        out: dict[str, Any] = {"ok": True, "entry": entry, "learned": learned}
        if rerun and corrected:
            out["rerun"] = self.send_message(corrected, meta={"source": "correction_rerun", "corrected_from": original_text[:120]})
        return out

    def activity_corrections(self, limit: int = 200) -> dict[str, Any]:
        path = Path(self.kernel.state_root) / "owner" / "activity_corrections.jsonl"
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass
        return {"ok": True, "corrections": rows}

    def list_activity(self, limit: int = 200) -> dict[str, Any]:
        """Everything that happened, newest last, from the durable log."""

        try:
            entries = [entry.to_dict() for entry in self.activity.recent(limit)]
        except Exception as exc:
            return {"activity": [], "error": f"{type(exc).__name__}: {exc}"}
        return {"activity": entries, "count": len(entries)}

    def list_receipts(self, limit: int = 50) -> dict[str, Any]:
        """Every side effect this system has performed, newest last."""

        try:
            return {"receipts": [receipt.to_dict() for receipt in self.receipts.recent(limit)]}
        except Exception as exc:
            return {"receipts": [], "error": f"{type(exc).__name__}: {exc}"}

    def receipt(self, receipt_id: str) -> dict[str, Any]:
        found = self.receipts.get(receipt_id)
        return found.to_dict() if found is not None else {"error": f"no receipt {receipt_id!r}"}

    #: One file.  The capability service, the experience memory and every
    #: owner-facing route read and write the same graph.  Before this, the
    #: routes used ``graph.db`` while the autonomous machinery wrote
    #: ``palace.sqlite``: the owner asked for a finding to be stored, the
    #: search honestly reported 0 nodes of an empty file, and a note file was
    #: written instead.
    KNOWLEDGE_FILE = ("knowledge", "palace.sqlite")
    LEGACY_KNOWLEDGE_FILE = ("knowledge", "graph.db")

    @property
    def graph_path(self) -> Path:
        return Path(self.kernel.state_root).joinpath(*self.KNOWLEDGE_FILE)

    @property
    def graph(self) -> Any:
        """The long-lived graph handle (thread-safe connections inside)."""

        if getattr(self, "_graph", None) is None:
            from knowledge.graph import KnowledgeGraph

            path = self.graph_path
            path.parent.mkdir(parents=True, exist_ok=True)
            self._graph = KnowledgeGraph(path)
            self._migrate_legacy_graph(self._graph)
        return self._graph

    def _migrate_legacy_graph(self, graph: Any) -> None:
        """Fold a non-empty legacy graph.db into the one graph, once, keeping provenance."""

        legacy = Path(self.kernel.state_root).joinpath(*self.LEGACY_KNOWLEDGE_FILE)
        if not legacy.is_file():
            return
        try:
            from knowledge.graph import KnowledgeGraph

            with KnowledgeGraph(legacy) as old:
                nodes = old.nodes(limit=100000)
                moved = 0
                for node in nodes:
                    if graph.get(node.id) is None:
                        graph.add_node(node)
                        moved += 1
                for node in nodes:
                    for edge in old.edges_from(node.id):
                        if graph.get(edge.target) is not None:
                            graph.link(edge.source, edge.target, edge.type, weight=edge.weight, provenance=edge.provenance or "legacy graph.db")
            legacy.rename(legacy.with_suffix(".db.migrated"))
            if moved:
                self.emit(EventType.KNOWLEDGE, {"migrated": moved, "from": str(legacy)})
        except Exception as exc:  # noqa: BLE001 - never block startup on a legacy file
            self.emit(EventType.DIAGNOSTIC, {"knowledge": f"legacy graph not migrated: {exc}"})

    def knowledge_graph(self, *, query: str = "", limit: int = 300) -> dict[str, Any]:
        try:
            return self.graph.export(query=query, limit=limit)
        except Exception as exc:
            return {"nodes": [], "edges": [], "error": str(exc)}

    # -- typed primitives ----------------------------------------------

    @staticmethod
    def _node_type(name: str) -> Any:
        from knowledge.graph import NodeType

        text = str(name or "note").strip().lower().replace(" ", "_").replace("-", "_")
        for member in NodeType:
            if member.value == text or member.name.lower() == text:
                return member
        return NodeType.NOTE

    @staticmethod
    def _edge_type(name: str) -> Any:
        from knowledge.graph import EdgeType

        text = str(name or "relates_to").strip().lower().replace(" ", "_").replace("-", "_")
        for member in EdgeType:
            if member.value == text or member.name.lower() == text:
                return member
        return EdgeType.RELATES_TO

    def _resolve_node(self, reference: str, *, create_as: Any = None, provenance: str = "") -> Any:
        """A node by id, exact title (any type), or best search hit; optionally created."""

        from knowledge.graph import NodeType

        ref = str(reference or "").strip()
        if not ref:
            return None
        graph = self.graph
        node = graph.get(ref)
        if node is not None:
            return node
        for node_type in NodeType:
            node = graph.find_by_title(node_type, ref)
            if node is not None:
                return node
        for hit in graph.search_keyword(ref, limit=3):
            if str(getattr(hit.node, "title", "")).strip().lower() == ref.lower():
                return hit.node
        if create_as is not None:
            return graph.remember(create_as, ref, "", provenance=provenance or "owner request", confidence=0.6,
                                  metadata={"created_for": "relation target"})
        return None

    def knowledge_create(self, title: str, text: str = "", *, type: str = "note", tags: Any = (), links: Any = (),
                         provenance: str = "owner", metadata: dict[str, Any] | None = None, confidence: float = 0.9) -> dict[str, Any]:
        """Store one typed node and its typed relations; verified by reading it back."""

        from knowledge.graph import EdgeType, NodeType

        title = str(title or "").strip()
        if not title:
            return {"ok": False, "error": "a title is required"}
        try:
            node = self.graph.remember(self._node_type(type), title, str(text or ""), tags=[str(t) for t in (tags or []) if str(t).strip()],
                                      provenance=provenance, confidence=float(confidence), metadata=dict(metadata or {}))
            relations = []
            for item in self._link_items(links):
                target_ref, relation = item
                target = self._resolve_node(target_ref, create_as=NodeType.CONCEPT, provenance=provenance)
                if target is None or target.id == node.id:
                    continue
                edge = self.graph.link(node.id, target.id, self._edge_type(relation) if relation else EdgeType.RELATES_TO, provenance=provenance)
                relations.append({"edge_id": edge.id, "target": target.title, "target_id": target.id, "relation": edge.type.value if hasattr(edge.type, "value") else str(edge.type)})
            back = self.graph.get(node.id)
            found = any(hit.node.id == node.id for hit in self.graph.search(title, limit=10))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result = {"ok": back is not None, "node_id": node.id, "title": node.title, "type": node.type.value if hasattr(node.type, "value") else str(node.type),
                  "relations": relations, "read_back": back is not None, "searchable": found, "path": str(self.graph_path)}
        self.emit(EventType.KNOWLEDGE, {"created": node.id, "title": title, "relations": len(relations)})
        return result

    @staticmethod
    def _link_items(links: Any) -> list[tuple[str, str]]:
        """"ZEUS, Voice, Wakeword" | ["ZEUS", {"target": "Voice", "relation": "concerns"}] -> [(target, relation)]"""

        items: list[tuple[str, str]] = []
        if isinstance(links, str):
            links = [part for part in re.split(r"[,;/|]", links)]
        for item in links or []:
            if isinstance(item, dict):
                target = str(item.get("target") or item.get("title") or item.get("id") or "").strip()
                items.append((target, str(item.get("relation") or "")))
            else:
                items.append((str(item).strip(), ""))
        return [(t, r) for t, r in items if t]

    def knowledge_link(self, source: str, target: str, relation: str = "relates_to", *, provenance: str = "owner") -> dict[str, Any]:
        a = self._resolve_node(source)
        b = self._resolve_node(target)
        if a is None or b is None:
            return {"ok": False, "error": f"unknown node: {source if a is None else target}"}
        try:
            edge = self.graph.link(a.id, b.id, self._edge_type(relation), provenance=provenance)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self.emit(EventType.KNOWLEDGE, {"linked": [a.id, b.id], "relation": relation})
        return {"ok": True, "edge_id": edge.id, "source": a.title, "target": b.title, "relation": edge.type.value if hasattr(edge.type, "value") else str(edge.type)}

    def knowledge_read(self, reference: str) -> dict[str, Any]:
        node = self._resolve_node(reference)
        if node is None:
            return {"ok": False, "error": f"no node {reference!r}"}
        detail = self.graph.node_detail(node.id)
        return {"ok": True, **detail}

    def knowledge_backlinks(self, reference: str) -> dict[str, Any]:
        node = self._resolve_node(reference)
        if node is None:
            return {"ok": False, "error": f"no node {reference!r}"}
        return {"ok": True, "node_id": node.id, "backlinks": [n.to_dict() if hasattr(n, "to_dict") else {"id": n.id, "title": n.title}
                                                              for n in self.graph.backlinks(node.id)]}

    def knowledge_delete(self, reference: str, *, confirm: bool = False) -> dict[str, Any]:
        node = self._resolve_node(reference)
        if node is None:
            return {"ok": False, "error": f"no node {reference!r}"}
        if not confirm:
            return {"ok": False, "error": "deleting is destructive; confirm it", "node_id": node.id, "title": node.title}
        self.graph.delete_node(node.id)
        self.emit(EventType.KNOWLEDGE, {"deleted": node.id, "title": node.title})
        return {"ok": True, "deleted": node.id}

    def knowledge_stats(self) -> dict[str, Any]:
        try:
            return {"ok": True, "path": str(self.graph_path), **self.graph.stats()}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "path": str(self.graph_path), "error": str(exc)}

    def ingest(self, path: str = "", *, text: str = "", title: str = "", recursive: bool = True, max_files: int = 500) -> dict[str, Any]:
        """Read a file, a folder, or a piece of text into the knowledge graph."""

        try:
            from knowledge.graph import KnowledgeGraph
            from knowledge.ingest import Ingester
        except ImportError as exc:  # pragma: no cover - defensive
            return {"ok": False, "error": str(exc)}

        target = (path or "").strip()
        body = (text or "").strip()
        if not target and not body:
            return {"ok": False, "error": "give a path or some text"}

        try:
            if True:
                graph = self.graph
                ingester = Ingester(graph)
                if body:
                    node = ingester.ingest_text(title or body[:60], body)
                    result = {"ok": True, "node_id": node.id, "title": node.title}
                else:
                    source = Path(target).expanduser()
                    report = (
                        ingester.ingest_folder(source, recursive=recursive, max_files=max_files)
                        if source.is_dir()
                        else ingester.ingest_file(source)
                    )
                    result = {"ok": report.files_ingested > 0, **report.to_dict()}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        self.emit(EventType.KNOWLEDGE, result)
        return result

    # -- the knowledge library and PDFs --------------------------------

    def pdf_extract(self, path: str, *, max_pages: int = 60, max_chars: int = 60_000) -> dict[str, Any]:
        """Real text out of a real PDF, bounded; says when a PDF has no text layer."""

        try:
            from pypdf import PdfReader
        except ImportError:
            return {"ok": False, "error": "pypdf ist nicht installiert"}
        source = Path(str(path or "").strip('" ')).expanduser()
        if not source.is_file() or source.suffix.lower() != ".pdf":
            return {"ok": False, "error": f"keine PDF-Datei: {path}"}
        try:
            reader = PdfReader(str(source))
            pages = len(reader.pages)
            chunks: list[str] = []
            total = 0
            for page in reader.pages[:max_pages]:
                text = page.extract_text() or ""
                chunks.append(text)
                total += len(text)
                if total >= max_chars:
                    break
        except Exception as exc:  # noqa: BLE001 - a broken PDF is data, not a crash
            return {"ok": False, "error": f"PDF nicht lesbar: {type(exc).__name__}: {exc}"}
        text = "\n\n".join(chunks)[:max_chars].strip()
        if not text:
            return {"ok": False, "error": "die PDF enthält keinen extrahierbaren Text (vermutlich nur gescannte Bilder)",
                    "pages": pages, "path": str(source)}
        return {"ok": True, "path": str(source), "pages": pages, "pages_read": min(pages, max_pages),
                "chars": len(text), "text": text}

    def pdf_summarize(self, path: str, *, save: bool = True, to_knowledge: bool = True) -> dict[str, Any]:
        """Extract → summarize with the local model → save the summary as a real
        Markdown file in the library → put it into the knowledge graph."""

        from brain.tiers import ModelTier

        extracted = self.pdf_extract(path)
        if not extracted.get("ok"):
            return extracted
        source = Path(extracted["path"])
        # the PDF itself lands on the shelf so Explorer shows what ZEUS read
        imported = self.library.import_file(str(source)) if save else {"ok": False}
        prompt = ("Fasse das folgende Dokument strukturiert auf Deutsch zusammen: zuerst 2-3 Sätze Kernaussage, "
                  "dann die wichtigsten Punkte als knappe Liste. Erfinde nichts, was nicht im Text steht.\n\n"
                  f"DOKUMENT ({source.name}, {extracted['pages']} Seiten):\n{extracted['text'][:16000]}")
        try:
            provider = self.kernel.provider(ModelTier.FAST_LOCAL)
            try:
                summary = provider.generate(prompt, system="Du bist ein präziser Zusammenfasser. Nur Inhalte aus dem Dokument.")
            except TypeError:
                summary = provider.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Zusammenfassung fehlgeschlagen: {exc}", **{k: v for k, v in extracted.items() if k != "text"}}
        summary = str(summary or "").strip()
        if not summary:
            return {"ok": False, "error": "das Modell hat keine Zusammenfassung geliefert", "path": str(source)}
        out: dict[str, Any] = {"ok": True, "path": str(source), "pages": extracted["pages"], "chars": extracted["chars"],
                               "summary": summary, "imported": imported.get("relative", "")}
        if save:
            body = f"# {source.stem} – Zusammenfassung\n\nQuelle: {source.name} · {extracted['pages']} Seiten · erzeugt {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{summary}\n"
            saved = self.library.save_summary(imported.get("relative", source.stem), body)
            out["summary_file"] = saved.get("relative", "") if saved.get("ok") else ""
            out["summary_saved"] = bool(saved.get("ok"))
        if to_knowledge:
            node = self.knowledge_create(f"{source.stem} (PDF-Zusammenfassung)", summary[:4000], type="document",
                                         provenance=str(source), links=[{"target": "Studium", "relation": "part_of"}])
            out["knowledge_node"] = node.get("node_id", "") if node.get("ok") else ""
        self.emit(EventType.KNOWLEDGE, {"pdf_summarized": source.name, "summary_file": out.get("summary_file", "")})
        return out

    def graph_operation(
        self, request: str, *, selected: str = "", confirm: bool = False
    ) -> dict[str, Any]:
        """Do something to the knowledge graph, described in words."""

        request = (request or "").strip()
        if not request:
            return {"ok": False, "error": "say what to do"}

        from brain.tiers import ModelTier
        from knowledge.operations import GraphOperator

        try:
            if True:
                graph = self.graph
                operator = GraphOperator(graph, brain=self.kernel.provider(ModelTier.FAST_LOCAL))
                result = operator.perform(request, selected=selected, confirm=confirm)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        payload = result.to_dict()
        if result.ok:
            self.emit(EventType.KNOWLEDGE, {"operation": result.operation, "detail": result.detail})
        return payload

    def research(self, question: str, *, max_sources: int = 3) -> dict[str, Any]:
        """Answer a technical question from public documentation, with citations."""

        question = (question or "").strip()
        if not question:
            return {"ok": False, "error": "question is required"}

        from brain.tiers import ModelTier
        from research.agent import ResearchAgent

        self.state.set(JarvisState.RESEARCHING, detail=question[:120])
        try:
            if True:
                graph = self.graph
                agent = ResearchAgent(
                    brain=self.kernel.provider(ModelTier.FAST_LOCAL), graph=graph
                )
                report = agent.research(question, max_sources=max_sources)
        except Exception as exc:
            self.state.set(JarvisState.IDLE)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        self.state.set(JarvisState.IDLE)
        payload = report.to_dict()
        self.emit(EventType.KNOWLEDGE, {"research": question, "sources": len(report.sources)})
        return {"ok": report.grounded, **payload}

    def knowledge_node(self, node_id: str) -> dict[str, Any]:
        try:
            return self.graph.node_detail(node_id)
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """What the ordinary UI shows: one Jarvis, and whether it is reachable.

        Never blocks.  An honest health check means a real generation, which
        costs ~80 s cold on this hardware -- so doing it inline would hang the
        page on load and make the interface feel broken while proving the
        opposite.  The probe runs in the background and this returns the last
        thing it learned, which is why ``health_checked`` is reported: "not
        measured yet" and "measured, offline" are different claims and the UI
        should not conflate them.
        """

        snapshot = self.state.snapshot
        expert = self._cached_expert_status()
        online = self._cached_health()

        if expert.get("quota_exhausted"):
            connection = "EXPERT QUOTA EXHAUSTED"
        elif expert.get("expert_available"):
            connection = "EXPERT AVAILABLE"
        elif online:
            connection = "LOCAL"
        elif self._health_checked_at == 0.0:
            # Not measured yet is not the same claim as measured-and-offline,
            # and reporting OFFLINE during the first probe would tell the user
            # their system is broken while it is merely starting.
            connection = "STARTING"
        else:
            connection = "OFFLINE"

        return {
            "persona": self.persona_name,
            "product": self.identity.product_name,
            "state": snapshot.to_dict(),
            "connection": connection,
            "language": self.language or "auto",
            "health_checked": self._health_checked_at > 0,
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "gpu": self.gpu_usage(),
        }

    def gpu_usage(self) -> dict[str, Any]:
        """How busy the GPU is, as of the last background reading.

        Never blocks and never probes a model: the whole point of showing this
        beside the eye is that the user can see the card working while Jarvis
        thinks, and a readout that slowed the generation down would be
        reporting a cost it created.  A machine with no NVIDIA GPU gets
        ``available: False`` and the interface simply shows nothing.
        """

        try:
            if self._gpu_usage is None:
                from brain.resources import GpuUsageMonitor

                self._gpu_usage = GpuUsageMonitor()
            return self._gpu_usage.snapshot()
        except Exception as exc:
            return {"measured": True, "available": False, "error": f"{type(exc).__name__}: {exc}"}

    # -- background probing ---------------------------------------------

    def _cached_health(self) -> bool:
        """Whether local inference worked, as of the last completed probe."""

        age = time.time() - self._health_checked_at
        # Never while a turn is in flight. Even a cheap probe queues behind or
        # ahead of the user's generation on a single GPU, and a status badge
        # must not be able to slow down the thing it is reporting on.
        if self.state.state.busy:
            return self._health_ok
        if age > self.HEALTH_TTL_SECONDS and not self._probe_running.is_set():
            self._probe_running.set()
            threading.Thread(target=self._probe_health, daemon=True, name="jarvis-probe").start()
        return self._health_ok

    def _probe_health(self) -> None:
        """Is local inference working?  Asked of the model that answers.

        This used to call ``ready_for_autonomous_work()``, which runs a real
        generation on **BUILD_LOCAL** -- the 7B coder.  On a single GPU that
        evicts the 4B conversational model, and the user's next sentence then
        pays to load it back.  Measured on this machine:

            after a FAST_LOCAL generation : qwen3:4b-instruct resident
            after the BUILD_LOCAL probe (47.1s): qwen2.5-coder:7b resident
            next FAST_LOCAL generation costs 28.3s

        It fired on a 120-second timer whenever the UI drew its status badge,
        so conversation paid a 28-second reload every two minutes, for the life
        of the process.  A status light was the most expensive thing running.

        The badge's actual question is "can it answer me", and what answers is
        FAST_LOCAL -- which is already resident, so probing it is nearly free.
        BUILD_LOCAL readiness is a different question, asked by the thing that
        is about to use BUILD_LOCAL, where loading it is the point rather than
        collateral damage.
        """

        from brain.tiers import ModelTier

        try:
            health = self.kernel.probe.probe(ModelTier.FAST_LOCAL)
            self._health_ok = bool(health.online)
            self._health_detail = health.summary()
        except Exception as exc:
            self._health_ok = False
            self._health_detail = f"{type(exc).__name__}: {exc}"
        finally:
            self._health_checked_at = time.time()
            self._probe_running.clear()
            self.emit(
                EventType.DIAGNOSTIC,
                {"health_ok": self._health_ok, "detail": self._health_detail[:300]},
            )

    def _cached_expert_status(self) -> dict[str, Any]:
        """Expert availability, refreshed at most once a minute.

        Cached for a different reason than health: asking costs a subprocess
        launch, not a generation, but the UI polls status every 15 seconds and
        spawning a process that often for a value that rarely changes is waste
        the user would feel as fan noise.
        """

        stale = time.time() - self._expert_checked_at >= self.EXPERT_TTL_SECONDS
        if stale and not self._expert_probe_running.is_set():
            self._expert_probe_running.set()
            threading.Thread(
                target=self._probe_experts, daemon=True, name="jarvis-expert-probe"
            ).start()
        # Return what is known now, even on the very first call. Blocking here
        # would put a subprocess launch on the page-load path, and a UI that
        # takes four seconds to appear reads as broken regardless of what it
        # eventually says.
        return self._expert_status

    def _probe_experts(self) -> None:
        try:
            status = self.experts.status()
        except Exception as exc:
            status = {"expert_available": False, "quota_exhausted": False, "error": str(exc)}
        self._expert_status = status
        self._expert_checked_at = time.time()
        self._expert_probe_running.clear()
        self.emit(EventType.DIAGNOSTIC, {"experts": status})

    def diagnostics(self, *, refresh: bool = False) -> dict[str, Any]:
        """The truth about the machinery, for when the user asks for it.

        Reports measured health rather than measuring it.  Probing a tier is a
        real generation, so drawing this panel used to evict the conversational
        model and leave the user's next sentence 28 seconds slower -- a
        diagnostic that degraded the thing it was diagnosing.  ``refresh=True``
        is the explicit way to pay that cost on purpose.
        """

        payload: dict[str, Any] = {
            "persona": self.persona_name,
            "identity": self.identity.to_dict(),
            "measured": refresh,
        }
        try:
            payload["kernel"] = self.kernel.status(force=refresh, probe=refresh)
        except Exception as exc:
            payload["kernel"] = {"error": str(exc)}
        try:
            payload["experts"] = self.experts.status()
        except Exception as exc:
            payload["experts"] = {"error": str(exc)}
        try:
            from runtime.cost_policy import CostPolicy

            payload["cost_policy"] = CostPolicy.load().to_dict()
        except Exception as exc:
            payload["cost_policy"] = {"error": str(exc)}
        payload["events"] = {"sequence": self.bus.sequence, "subscribers": self.bus.subscriber_count}
        payload["state"] = self.state.snapshot.to_dict()
        return payload


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
