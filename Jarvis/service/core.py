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

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "text": self.text, "at": self.at, "backend": self.backend}


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
        persona_name: str = "Jarvis",
    ) -> None:
        self.bus = bus or EventBus()
        self.state = StateMachine(on_change=self._publish_state)
        self.persona_name = persona_name
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
        self._voice: Any = None

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
    def voice(self) -> Any:
        if self._voice is None:
            from service.voice import VoiceService

            self._voice = VoiceService(self.bus)
        return self._voice

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

    def send_message(self, text: str, *, scope: str = "") -> dict[str, Any]:
        """Accept user input and answer it, streaming tokens as events."""

        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "empty message"}

        turn = ConversationTurn(role="user", text=text, at=_now())
        with self._lock:
            self._history.append(turn)
        self.emit(EventType.USER_MESSAGE, turn.to_dict(), scope=scope)

        # Answering happens off the request thread so the HTTP call returns at
        # once and the client watches the event stream, which is what makes
        # "Jarvis starts speaking before the answer is finished" possible.
        thread = threading.Thread(
            target=self._answer, args=(text, scope), daemon=True, name="jarvis-answer"
        )
        with self._lock:
            self._current_work = thread
        self._stop_requested.clear()
        thread.start()
        return {"ok": True, "accepted": text}

    def _answer(self, text: str, scope: str) -> None:
        from brain.tiers import ModelTier

        self.state.set(JarvisState.THINKING, detail=text[:120], scope=scope)
        collected: list[str] = []
        backend = ""
        try:
            tier = ModelTier.FAST_LOCAL
            provider = self.kernel.provider(tier)
            backend = getattr(self.kernel.catalog.get(tier), "model", "") or tier.value
            prompt = self._compose_prompt(text)
            stream = self._generate(provider, prompt)

            def tee():
                """One pass over the model's output feeds both the screen and the voice.

                Generating twice -- once to display, once to speak -- would
                double the cost and let the two drift apart, which the user
                would hear as speech that does not match the text on screen.
                """

                for chunk in stream:
                    if self._stop_requested.is_set():
                        return
                    collected.append(chunk)
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
            self.state.set(JarvisState.ERROR, detail=str(exc)[:200])
            self.emit(EventType.ERROR, {"error": f"{type(exc).__name__}: {exc}"}, scope=scope)
            return

        answer = "".join(collected).strip()
        reply = ConversationTurn(role="assistant", text=answer, at=_now(), backend=backend)
        with self._lock:
            self._history.append(reply)
        self.emit(EventType.MESSAGE, reply.to_dict(), scope=scope)
        self.state.set(JarvisState.IDLE)

    def _generate(self, provider: Any, prompt: str) -> Iterable[str]:
        """Stream from a provider, falling back to a single block if it cannot."""

        stream = getattr(provider, "generate_stream", None)
        if callable(stream):
            yield from stream(prompt)
            return
        yield provider.generate(prompt)

    def _compose_prompt(self, text: str) -> str:
        """Wrap the user's words in the persona and recent context.

        The persona is stated as identity rather than as a costume instruction
        ("you are roleplaying as..."), because the latter invites a model to
        break character and explain what it really is the moment it is asked.
        """

        recent = self.history[-8:]
        transcript = "\n".join(f"{turn.role}: {turn.text}" for turn in recent[:-1])
        return (
            f"You are {self.persona_name}, this user's personal AI system. You are not a chat "
            "assistant demo and you do not describe yourself as a language model. You are "
            "concise, competent and task-oriented. You answer in the language the user writes in.\n"
            "If you genuinely cannot do something, say what is missing rather than refusing "
            "vaguely. Never claim an action succeeded unless it did.\n\n"
            + (f"Recent conversation:\n{transcript}\n\n" if transcript else "")
            + f"user: {text}\n{self.persona_name}:"
        )

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
            self.emit(EventType.DIAGNOSTIC, {"warming": "started"})
            try:
                from brain.tiers import ModelTier

                provider = self.kernel.provider(ModelTier.FAST_LOCAL)
                provider.generate("OK", max_tokens=4, temperature=0.0)
                self.emit(EventType.DIAGNOSTIC, {"warming": "conversation model ready"})
            except Exception as exc:
                self.emit(EventType.DIAGNOSTIC, {"warming": f"conversation model unavailable: {exc}"})

            if speech:
                try:
                    # Synthesising one short phrase loads the voice; discarding
                    # it is the cheapest way to pay that cost early.
                    self.voice.engine.synthesize("Bereit.")
                    self.emit(EventType.DIAGNOSTIC, {"warming": "voice ready"})
                except Exception as exc:
                    self.emit(EventType.DIAGNOSTIC, {"warming": f"speech unavailable: {exc}"})

                try:
                    # And the recogniser, which is the larger of the two costs:
                    # loading whisper-base took ~50 s of the 54 s first
                    # exchange. Half a second of silence is enough to force it.
                    from speech.contracts import Audio

                    silence = Audio(samples=bytes(32000), sample_rate=16000)
                    self.voice.engine.transcribe(silence)
                    self.emit(EventType.DIAGNOSTIC, {"warming": "recogniser ready"})
                except Exception as exc:
                    self.emit(EventType.DIAGNOSTIC, {"warming": f"recogniser unavailable: {exc}"})

            self.emit(EventType.DIAGNOSTIC, {"warming": "done"})

        thread = threading.Thread(target=run, daemon=True, name="jarvis-warm")
        thread.start()
        return thread

    # ------------------------------------------------------------------
    # Voice
    # ------------------------------------------------------------------

    def voice_settings(self, **changes: Any) -> dict[str, Any]:
        """Read or update voice settings; returns the full voice status."""

        settings = self.voice.settings
        for key, value in changes.items():
            if hasattr(settings, key) and value is not None:
                setattr(settings, key, value)
        return self.voice.status()

    def hear(self, wav: bytes, *, language: str = "", answer: bool = True) -> dict[str, Any]:
        """Transcribe a posted utterance and, unless told otherwise, reply to it."""

        # Speaking to Jarvis is what enters voice mode; it is the least
        # surprising trigger and needs no separate switch.
        self.voice.settings.enabled = True
        transcript = self.voice.transcribe(wav, language=language)
        if transcript.empty:
            self.state.set(JarvisState.IDLE, detail="nothing heard")
            return {"ok": False, "text": "", "reason": "no speech detected"}
        if answer:
            self.send_message(transcript.text)
        return {"ok": True, **transcript.to_dict()}

    def stop_current(self) -> dict[str, Any]:
        """Interrupt whatever is running.  Used by barge-in and the stop button."""

        self._stop_requested.set()
        if self._voice is not None:
            # Barge-in has to reach the speaker, not just the generator: the
            # audio already synthesised would otherwise keep playing over the
            # user who interrupted it.
            self._voice.interrupt()
        self.emit(EventType.NOTIFICATION, {"text": "stopped"})
        self.state.set(JarvisState.IDLE, detail="interrupted")
        return {"ok": True}

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        try:
            projects = self.kernel.projects.list_projects()
        except Exception:
            return []
        return [
            {
                "id": project.id,
                "goal": project.goal,
                "state": getattr(project.state, "value", str(project.state)),
                "tasks": len(getattr(project, "tasks", [])),
                "steps": len(getattr(project, "steps", [])),
                "updated_at": getattr(project, "updated_at", ""),
            }
            for project in projects
        ]

    def project_detail(self, reference: str) -> dict[str, Any]:
        project = self.kernel.resolve_project(reference) if reference else None
        if project is None:
            return {"error": f"no project matching {reference!r}"}
        return {
            "id": project.id,
            "goal": project.goal,
            "state": getattr(project.state, "value", str(project.state)),
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

    def list_capabilities(self) -> list[dict[str, Any]]:
        try:
            from capabilities.registry import CapabilityRegistry

            registry = CapabilityRegistry(self.kernel.state_root / "capabilities")
            return [manifest.to_dict() for manifest in registry.all()]
        except Exception:
            return []

    def knowledge_graph(self, *, query: str = "", limit: int = 300) -> dict[str, Any]:
        try:
            from knowledge.graph import KnowledgeGraph

            graph = KnowledgeGraph(self.kernel.state_root / "knowledge" / "graph.db")
            return graph.export(query=query, limit=limit)
        except Exception as exc:
            return {"nodes": [], "edges": [], "error": str(exc)}

    def knowledge_node(self, node_id: str) -> dict[str, Any]:
        try:
            from knowledge.graph import KnowledgeGraph

            graph = KnowledgeGraph(self.kernel.state_root / "knowledge" / "graph.db")
            return graph.node_detail(node_id)
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
            "state": snapshot.to_dict(),
            "connection": connection,
            "health_checked": self._health_checked_at > 0,
            "uptime_seconds": round(time.time() - self._started_at, 1),
        }

    # -- background probing ---------------------------------------------

    def _cached_health(self) -> bool:
        """Whether local inference worked, as of the last completed probe."""

        age = time.time() - self._health_checked_at
        if age > self.HEALTH_TTL_SECONDS and not self._probe_running.is_set():
            self._probe_running.set()
            threading.Thread(target=self._probe_health, daemon=True, name="jarvis-probe").start()
        return self._health_ok

    def _probe_health(self) -> None:
        try:
            ready, detail = self.kernel.ready_for_autonomous_work()
            self._health_ok = bool(ready)
            self._health_detail = str(detail)
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

    def diagnostics(self) -> dict[str, Any]:
        """The truth about the machinery, for when the user asks for it."""

        payload: dict[str, Any] = {"persona": self.persona_name}
        try:
            payload["kernel"] = self.kernel.status()
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
