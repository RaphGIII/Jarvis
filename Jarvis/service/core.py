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
    #: What this turn contributes to the *next* prompt, when that should differ
    #: from what the user was shown.  An action turn displays its full evidence
    #: block -- paths, byte counts, every check -- and that is right for a
    #: reader and wrong for a transcript: a model shown three receipts saying
    #: "erstellt / geschrieben / ok" starts producing that language itself, and
    #: the next ordinary answer trips the claim guard. Empty means "same text".
    context_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "text": self.text, "at": self.at, "backend": self.backend}

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
        self._voice: Any = None
        self._personas: Any = None
        self._gateway: Any = None
        self._receipts: Any = None
        self._actions: Any = None
        self._activity: Any = None
        #: Receipts produced during this conversation, in memory.  The claim
        #: guard consults them on every streamed chunk, and re-reading the
        #: ledger file that often would cost more than the check saves.
        self._session_receipts: list[Any] = []
        #: The language the conversation is currently in.  Sticky: it changes
        #: only on a confident detection, because flipping mid-conversation
        #: changes the recogniser hint and the voice, which sounds worse than
        #: occasionally answering in the wrong language.
        self.language = ""

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

            self._voice = VoiceService(self.bus)
        return self._voice

    @property
    def gateway(self) -> Any:
        if self._gateway is None:
            from devices.gateway import DeviceGateway

            self._gateway = DeviceGateway(self.kernel.state_root / "devices.json")
        return self._gateway

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

    def send_message(self, text: str, *, scope: str = "") -> dict[str, Any]:
        """Accept user input and answer it, streaming tokens as events."""

        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "empty message"}

        # Touched before the first event is published, so the request that
        # started everything is in the record rather than missing from it.
        # `warm()` normally pays this cost at startup; here it is the guarantee.
        try:
            self.activity
        except Exception:
            pass

        from persona.language import stable_language

        self.language = stable_language(text, current=self.language)

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

        classification = classify(text)
        self.emit(
            EventType.DIAGNOSTIC,
            {"classified": classification.to_dict(), "text": text[:120]},
            scope=scope,
        )

        # CAPABILITY joins READ rather than going to the executor, because
        # "learn to do X" cannot be executed from a chat turn in this system.
        # Answering it conversationally invites "I can do that now" -- a
        # present-tense capability claim, which the claim guard does not catch
        # because nothing was claimed to have been *done*. The registry knows
        # what is actually installed; the model does not.
        if classification.intent in {Intent.READ, Intent.CAPABILITY}:
            self._answer_from_records(text, scope, classification)
            return
        if classification.intent.has_side_effect:
            self._answer_by_executing(text, scope, classification)
            return
        self._answer_conversationally(text, scope)

    # -- the three answering paths --------------------------------------

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

    def _answer_by_executing(self, text: str, scope: str, classification: Any) -> None:
        """A request with a side effect.  Nothing is said until something is done.

        No model output reaches the user on this path.  The model is asked for
        one thing -- a machine-readable plan -- and the sentence the user reads
        is composed from the receipt by :func:`service.actions.compose`.  There
        is no step here at which a model could assert that something worked.
        """

        from brain.tiers import ModelTier
        from service.actions import compose

        self.state.set(JarvisState.WORKING, detail=classification.reason[:120], scope=scope)
        try:
            provider = self.kernel.provider(ModelTier.FAST_LOCAL)
            plan = self.actions.plan(text, provider)
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
            self._answer_conversationally(text, scope)
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

    def _answer_conversationally(self, text: str, scope: str) -> None:
        from brain.tiers import ModelTier

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
            prompt = self._compose_prompt(text)
            stream = self._generate(provider, prompt)

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
            self.state.set(JarvisState.ERROR, detail=str(exc)[:200])
            self.emit(EventType.ERROR, {"error": f"{type(exc).__name__}: {exc}"}, scope=scope)
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
        self._deliver(answer, scope=scope, backend=backend, context_text=context_text)

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
    ) -> None:
        """Record the assistant's turn, publish it, and settle into a state."""

        reply = ConversationTurn(
            role="assistant", text=answer, at=_now(), backend=backend, context_text=context_text
        )
        with self._lock:
            self._history.append(reply)
        self.emit(EventType.MESSAGE, reply.to_dict(), scope=scope)
        self.state.set(final_state, detail="" if final_state is JarvisState.IDLE else answer[:120])

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

        The text comes from the persona store rather than being written here, so
        that "add another persistent personality" is a stored record instead of
        an edit to this method -- and so the invariant rules (never claim an
        unverified success, prefer admitting ignorance) are appended after the
        persona's own words, where a verbose character cannot crowd them out.
        """

        base = self.identity.persona_preamble()
        try:
            system = self.personas.system_prompt(
                base=base, assistant=self.identity.assistant_name
            )
        except Exception:
            # A missing or unreadable persona file must not silence Jarvis.
            system = base

        if self.language:
            from persona.language import language_name

            system += (
                f"\nThe user is speaking {language_name(self.language)}; reply in that language."
            )

        recent = self.history[-8:]
        transcript = "\n".join(f"{turn.role}: {turn.for_prompt()}" for turn in recent[:-1])
        # The speaker label comes from the identity, never from persona_name.
        # Those were the same field, so `--persona Jarvis` -- the default in
        # jarvis/serve.py -- ended every prompt with "Jarvis:", asking the model
        # in the most direct way available to answer as Jarvis. It duly did.
        # persona_name selects a *style*; who is speaking is not a style.
        return (
            system
            + "\n\n"
            + (f"Recent conversation:\n{transcript}\n\n" if transcript else "")
            + f"user: {text}\n{self.identity.assistant_name}:"
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
            try:
                # Before the first diagnostic, so startup itself is on the record.
                self.activity
            except Exception:
                pass
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
        # Hint the recogniser with the language the conversation is already in.
        # Whisper decodes measurably better when told, and the alternative --
        # letting it decide per utterance -- makes it flip on short phrases.
        transcript = self.voice.transcribe(
            wav, language=language or self.voice.settings.language or self.language
        )
        if transcript.empty:
            self.state.set(JarvisState.IDLE, detail="nothing heard")
            return {"ok": False, "text": "", "reason": "no speech detected"}
        if transcript.language:
            from persona.language import stable_language

            # Trust the recogniser's own verdict as evidence, but require the
            # same confidence threshold as text: a mis-heard word should not
            # switch the voice.
            self.language = stable_language(transcript.text, current=self.language)
        if answer:
            self.send_message(transcript.text)
        return {"ok": True, **transcript.to_dict()}

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

    def new_conversation(self) -> dict[str, Any]:
        """Start fresh: clear the transcript and settle the state.

        The *conversation* resets; the *record* does not.  The activity log and
        the receipt ledger are untouched, so a failed action the user just
        cleared off their screen is still there for anyone who goes looking.
        Hiding the transcript is a convenience; hiding the evidence would be
        the thing this whole system exists to prevent.
        """

        with self._lock:
            self._history.clear()
        self._session_receipts.clear()
        self.language = ""
        self.state.set(JarvisState.IDLE)
        return {"ok": True, "cleared": True}

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
                # The title is what a user names a project and what they will
                # look for in the panel; the goal can be a paragraph.
                "title": getattr(project, "title", "") or "",
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

    def knowledge_graph(self, *, query: str = "", limit: int = 300) -> dict[str, Any]:
        try:
            from knowledge.graph import KnowledgeGraph

            graph = KnowledgeGraph(self.kernel.state_root / "knowledge" / "graph.db")
            return graph.export(query=query, limit=limit)
        except Exception as exc:
            return {"nodes": [], "edges": [], "error": str(exc)}

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

        graph_path = self.kernel.state_root / "knowledge" / "graph.db"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with KnowledgeGraph(graph_path) as graph:
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

    def graph_operation(
        self, request: str, *, selected: str = "", confirm: bool = False
    ) -> dict[str, Any]:
        """Do something to the knowledge graph, described in words."""

        request = (request or "").strip()
        if not request:
            return {"ok": False, "error": "say what to do"}

        from brain.tiers import ModelTier
        from knowledge.graph import KnowledgeGraph
        from knowledge.operations import GraphOperator

        graph_path = self.kernel.state_root / "knowledge" / "graph.db"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with KnowledgeGraph(graph_path) as graph:
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
        from knowledge.graph import KnowledgeGraph
        from research.agent import ResearchAgent

        self.state.set(JarvisState.RESEARCHING, detail=question[:120])
        graph_path = self.kernel.state_root / "knowledge" / "graph.db"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with KnowledgeGraph(graph_path) as graph:
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
            "product": self.identity.product_name,
            "state": snapshot.to_dict(),
            "connection": connection,
            "language": self.language or "auto",
            "health_checked": self._health_checked_at > 0,
            "uptime_seconds": round(time.time() - self._started_at, 1),
        }

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
