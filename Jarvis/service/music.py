"""Music as an intent, not as a provider.

The forbidden shape for this feature is::

    if "music" in text:
        open_spotify()

which makes Spotify the only provider there can ever be, hides the user's
preference inside a conditional, and puts provider-specific behaviour in the
conversational handler where nothing can verify it.

The shape here instead::

    "spiel Lose Yourself von Eminem"
        -> MusicRequest(action="play", query="Lose Yourself Eminem")
        -> preference: music.default_provider = spotify
        -> capability registry: which capability provides "spotify"?
        -> that capability executes it
        -> Windows is asked what is actually playing
        -> Receipt

Nothing above the resolver knows the word Spotify.  A second provider is a
registered capability plus one preference change.

Two decisions worth defending.

*No model is involved in a music turn.*  Transport commands are a closed set of
short utterances, and a play request is the sentence minus its verb -- both are
deterministic, both are testable, and neither costs a generation.  "Pause."
therefore takes about as long as the media session call itself, and a music
request never wakes BUILD_LOCAL or an expert.

*The provider does not get to say whether it worked.*  It reports what it did;
:mod:`tools.media_session` then asks Windows what is playing and the receipt is
built from that.  An earlier attempt at a music capability in this project was
recorded as acquired while every branch returned ``{"message": "Dry run: ..."}``
and nothing ever played -- which is what happens when the actor is also the
witness.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime.receipts import Receipt, Verification, failed

#: The generic vocabulary.  Providers implement these; the conversation only
#: ever produces these.
ACTIONS = ("play", "pause", "resume", "next", "previous", "search", "current")

#: A transport command is a short utterance. "Weiter." is a resume; "Wie geht
#: es mit meinem Projekt weiter?" is a question about a project, and the only
#: thing separating them is that the second is a sentence. Requiring brevity is
#: cruder than parsing and considerably more reliable than a keyword list.
MAX_TRANSPORT_WORDS = 4

#: Brevity alone was not enough, and the counter-example was four words long.
#: "Wie geht es weiter?" is inside the limit and is still a question -- observed
#: live, where it resumed playback and then began acquiring a Spotify provider
#: to do it with, in answer to something nobody asked.
#:
#: An interrogative opening is a stronger signal than a word count because it
#: needs no threshold: a sentence that begins by asking is asking. The same
#: reasoning as elsewhere in this project -- a rule about the shape of a
#: sentence rather than a list of the phrases seen so far.
#:
#: This does not silence "Was laeuft gerade?": _CURRENT is matched before any
#: transport verb and returns first, which is why it is written that way round.
_INTERROGATIVE = re.compile(
    r"^(wie|was|warum|wieso|weshalb|wann|wo|wer|welche[srn]?|wieviel|how|what"
    r"|why|when|where|who|which)\b"
)


def _fold(text: str) -> str:
    lowered = (text or "").lower().replace("ß", "ss")
    for source, target in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        lowered = lowered.replace(source, target)
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


#: Words that put a sentence in the domain of music even when the verb does not.
CONTEXT = (
    "musik", "music", "lied", "song", "songs", "track", "titel", "album",
    "playlist", "spotify", "band", "radio",
)

_PLAY = re.compile(
    r"\b(spiel|spiele|spiel[ae]?\s*mir|spielt|abspielen|leg\s+auf|mach\s+.{0,12}\s*an"
    r"|play|put\s+on|start|starte|starten)\b"
)
#: Verbs that only mean "play" next to a music word.  "Wenn ich ZEUS.exe
#: starte" and "mach das Licht an" are not requests for a song; observed live,
#: where the first sent a paragraph about the desktop lifecycle to Spotify.
_PLAY_NEEDS_CONTEXT = re.compile(r"^(start|starte|starten|mach\b.*)$")
#: The longest thing anyone has ever asked to hear by name.  Beyond this a
#: "query" is prose, whatever verb list let it through.
MAX_QUERY_WORDS = 12
_PAUSE = re.compile(r"\b(pause|pausiere|pausier|anhalten|halt\s+an|stopp?e?|stop\s+it|pause\s+it)\b")
_RESUME = re.compile(r"\b(weiter|weiterspielen|fortsetzen|resume|continue|unpause|play\s+on)\b")
_NEXT = re.compile(
    r"\b(naechste[srn]?\s+(lied|song|titel|track|stueck)|naechstes|ueberspringen|skip"
    r"|next(\s+(song|track|one))?)\b"
)
_PREVIOUS = re.compile(
    r"\b(vorherige[srn]?(\s+(lied|song|titel|track))?|vorheriges|zurueck|previous|go\s+back)\b"
)
_CURRENT = re.compile(
    r"\b(was\s+(laeuft|spielt|hoere\s+ich)|welches\s+(lied|song|stueck)"
    r"|what.?s?\s+playing|current\s+(track|song)|now\s+playing|wer\s+singt)\b"
)

#: The play verb, found anywhere -- not anchored at the start, because the
#: sentence usually opens with the assistant's name ("Zeus, spiel ...") and
#: anchoring left that in the search query.
_PLAY_VERB = re.compile(
    r"\b(spiele?t?|abspielen|play|put\s+on|start|leg|mach)\b\s*(auf|on)?\s*", re.I
)
#: Trailing instructions that are not part of the track name.
_PLAY_SUFFIX = re.compile(
    r"\s*(auf|on|ueber|über|via|mit|in)\s+(spotify|deezer|apple\s*music|youtube|itunes)\s*\.?\s*$",
    re.I,
)
#: German separable-verb particles stranded at the end: "spiel mir was ... vor".
_PARTICLE = re.compile(r"\s+\b(vor|an|ab|auf)\b\s*\.?\s*$", re.I)
_FILLER = re.compile(r"\b(das\s+lied|den\s+song|die\s+nummer|the\s+song|the\s+track)\b", re.I)

#: Words that carry no identifying information about a track. A query made only
#: of these is not a query -- "Mach Musik an" asks for music, not for a song
#: called "Mach an".
_STOPWORDS = frozenset(
    """
    mir uns mal doch bitte was etwas irgendwas irgendetwas noch schon jetzt hier
    der die das den dem des ein eine einen einem einer und oder von fuer mit
    me us some something anything please just now the a an and or of for with
    zeus jarvis hey ok okay
    """.split()
)


@dataclass(frozen=True)
class MusicRequest:
    """A provider-independent music instruction."""

    action: str
    query: str = ""
    #: Which provider preference resolved this, filled in by the resolver.
    provider: str = ""
    reason: str = ""
    #: What the query names: track | artist | album | playlist | top_track | any.
    #: "Spiel Rammstein" is an ARTIST request; verifying it by looking for
    #: "Rammstein" in the track *title* failed a request that succeeded
    #: (track "Sonne", artist "Rammstein").  The kind decides what is checked.
    kind: str = "any"
    #: The artist named with "von"/"by", when the query is a track by someone.
    artist: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "query": self.query, "provider": self.provider, "kind": self.kind, "artist": self.artist}


def understand(text: str) -> MusicRequest | None:
    """Turn a sentence into a music instruction, or ``None`` if it is not one.

    Deterministic on purpose: this runs on every message, and a model call here
    would put a generation in front of "Pause."
    """

    folded = _fold(text)
    words = folded.split()
    short = len(words) <= MAX_TRANSPORT_WORDS
    has_context = any(word in folded for word in CONTEXT)

    if _CURRENT.search(folded):
        return MusicRequest("current", reason="asks what is playing")

    # Transport verbs are accepted bare only in a short utterance that is not a
    # question. In a longer sentence, or in any question, they need a music word
    # -- otherwise "wie geht es mit meinem Projekt weiter" becomes a resume, and
    # so does "wie geht es weiter", which is inside the word limit.
    commanding = (short and not _INTERROGATIVE.match(folded)) or has_context
    for pattern, action in ((_NEXT, "next"), (_PREVIOUS, "previous"),
                            (_PAUSE, "pause"), (_RESUME, "resume")):
        if pattern.search(folded) and commanding:
            return MusicRequest(action, reason=f"transport command: {action}")

    play = _PLAY.search(folded)
    if play and (has_context or not _PLAY_NEEDS_CONTEXT.match(play.group(1))):
        query = extract_query(text)
        if query and prose_reason(query):
            # A paragraph is not a track name, whichever verb matched.
            return None
        if query:
            kind, artist = classify_target(text, query)
            return MusicRequest("play", query=query, reason=f"names something to play ({kind})", kind=kind, artist=artist)
        if has_context:
            # "Mach Musik an" -- a play with no particular track.
            return MusicRequest("resume", reason="asks for music with no track named")
    return None


_ALBUM = re.compile(r"\b(das\s+album|the\s+album|album)\b", re.I)
_PLAYLIST = re.compile(r"\b(playlist|die\s+liste|wiedergabeliste)\b", re.I)
_TOP = re.compile(r"\b(top[-\s]?(?:track|song|titel|hits?)|beliebteste\w*|bekannteste\w*|most\s+popular|biggest\s+hit)\b", re.I)
_ARTIST_CUE = re.compile(r"\b(?:was|etwas|irgendwas|irgendetwas|musik|songs?|lieder|tracks?|something|some\s+music|anything)\s+(?:von|by|from)\s+(.+)$", re.I)
_BY = re.compile(r"\b(?:von|by)\s+(.+)$", re.I)


def classify_target(text: str, query: str) -> tuple[str, str]:
    """(kind, artist) for a play request.

    album / playlist / top_track by their nouns; "etwas von X" or a bare
    single name is an ARTIST request; "Titel von X" is a TRACK by artist X;
    a multi-word query without a "von" is a track (or any).
    """

    stripped = _PLAY_SUFFIX.sub("", (text or "").strip())
    if _PLAYLIST.search(stripped):
        return "playlist", ""
    if _ALBUM.search(stripped):
        return "album", ""
    if _TOP.search(stripped):
        m = _BY.search(query) or _ARTIST_CUE.search(stripped)
        return "top_track", (m.group(1).strip(" .!?") if m else query)
    m = _ARTIST_CUE.search(stripped)
    if m:
        return "artist", m.group(1).strip(" .!?")
    m = _BY.search(stripped)
    if m:
        return "track", m.group(1).strip(" .!?")
    if len(query.split()) <= 2:
        # one name: an artist or a track -- verified against both
        return "any", ""
    return "track", ""


def prose_reason(query: str) -> str:
    """Why ``query`` must not go to a provider as a search, or ``""``."""

    from service.routing import looks_like_prose

    return looks_like_prose(query, max_words=MAX_QUERY_WORDS)


def extract_query(text: str) -> str:
    """The search query hiding inside a play request.

    ``"Zeus, spiel Lose Yourself von Eminem auf Spotify."`` -> ``"Lose Yourself
    Eminem"``.  The assistant's name goes because it is address, the provider
    because it is routing, ``von`` because the search treats the whole string
    as terms, and a stranded ``vor`` because German puts half the verb at the
    end of the sentence.

    Returns ``""`` when nothing identifying is left.  That is the difference
    between "play Bohemian Rhapsody" and "put some music on", and the caller
    treats them differently -- one is a search, the other is a resume.
    """

    stripped = _PLAY_SUFFIX.sub("", (text or "").strip())

    # Everything up to and including the verb is instruction, not title.
    match = _PLAY_VERB.search(stripped)
    if match:
        stripped = stripped[match.end():]

    stripped = _FILLER.sub(" ", stripped)
    stripped = _PARTICLE.sub("", stripped)
    stripped = re.sub(r"\b(musik|music)\b", " ", stripped, flags=re.I)
    # "von"/"by" join a title to an artist for a reader; a search engine only
    # sees one more term to match against.
    stripped = re.sub(r"\b(von|by)\b", " ", stripped, flags=re.I)
    stripped = stripped.strip(" .,!?;:’'\"")
    stripped = re.sub(r"\s+", " ", stripped)

    # Drop leading filler words one at a time, so "mir was von den Beatles"
    # becomes "Beatles" rather than being kept whole or thrown away.
    words = stripped.split()
    while words and _fold(words[0]).strip(".,!?") in _STOPWORDS:
        words.pop(0)
    while words and _fold(words[-1]).strip(".,!?") in _STOPWORDS:
        words.pop()
    if not any(_fold(word).strip(".,!?") not in _STOPWORDS for word in words):
        return ""
    return " ".join(words).strip(" .,!?;:’'\"")


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

#: Failures that say something about the machine, not about the code.
#:
#: Treating these as defects retires a working capability and spends half an
#: hour rebuilding something that was never broken.  Measured: a verified
#: Spotify provider was disabled because one cold call -- PowerShell starting,
#: a token being fetched, a search over the network -- did not finish inside a
#: 120-second budget on a machine that had just spent thirty minutes running a
#: 7B model. The code was correct. The clock ran out.
#:
#: The distinction is the familiar one in this project: ``ok=False`` is a proxy
#: covering both "it behaved incorrectly" and "it never got to finish", and
#: only the first is evidence about the implementation.
ENVIRONMENTAL = (
    "did not finish within",
    "timed out",
    "timeout",
    "could not reach",
    "connection",
    "temporarily unavailable",
    "no active media session",
    "is not running",
    "powershell is not on path",
)


def is_environmental(reason: str) -> bool:
    """Whether a failure is about the machine rather than about the code."""

    lowered = (reason or "").lower()
    return any(marker in lowered for marker in ENVIRONMENTAL)


def defect_report(request: Any, reason: str, before: Any, after: Any) -> str:
    """A defect report carrying the state the world was in, not just the error.

    "play returned ok=False" tells a repair loop what broke and nothing about
    when.  The same action against the same code succeeds or fails depending on
    what the player was doing beforehand, and that is precisely the variable a
    repair has to discover -- so the report states it rather than making the
    loop go and find it, or worse, not think to look.

    Deliberately observations only.  It says what was true before, what was
    asked for, and what was true after; it does not say what to change. A
    report that names the remedy stops being evidence and starts being a patch
    written by whoever wrote the report.
    """

    return (
        f"run({{'action': '{request.action}'"
        + (f", 'query': {request.query!r}" if request.query else "")
        + f"}}) returned ok=False.\n"
        f"  provider said : {reason}\n"
        f"  player BEFORE : {before.describe() if before is not None else 'unknown'}\n"
        f"  player AFTER  : {after.describe() if after is not None else 'unknown'}\n"
        "  This is what was observed. It is not a diagnosis: establish for yourself "
        "under which prior player states the action succeeds and under which it does not."
    )


@dataclass
class MusicOutcome:
    """What happened, with the evidence to back it."""

    receipt: Receipt
    #: The provider capability that ran it, when one did.
    capability_id: str = ""
    gap: bool = False
    requirement: Any = None
    #: The provider itself is broken -- it failed, or it claimed something the
    #: operating system contradicts. Distinct from a gap (nothing installed) and
    #: from a requirement (the user has not supplied a credential), because
    #: those are not the capability's fault and rebuilding it would fix neither.
    defect: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


class MusicService:
    """Routes a generic music request to the user's configured provider."""

    #: Fields a music provider capability is expected to accept.  Published so
    #: a provider can be written against it rather than guessed at.
    PROVIDER_CONTRACT = {
        "action": "one of " + ", ".join(ACTIONS),
        "query": "free text to search for, when action is play or search",
        "client_id": "provider credential, when the provider needs one",
        "client_secret": "provider credential, when the provider needs one",
        "dry_run": "report what would happen without doing it",
    }

    def __init__(
        self,
        *,
        preferences: Any,
        capabilities: Any,
        secrets: Any = None,
        session: Any = None,
    ) -> None:
        self.preferences = preferences
        self.capabilities = capabilities
        self.secrets = secrets
        #: Injectable so a test can supply a fake Windows. Defaults to the real
        #: media session, which is the whole point of the verification step.
        if session is None:
            from tools import media_session

            session = media_session
        self.session = session

    # -- provider selection ----------------------------------------------

    @property
    def provider(self) -> str:
        return str(self.preferences.get("music.default_provider", "spotify"))

    @property
    def output(self) -> str:
        return str(self.preferences.get("music.default_output", "this_pc"))

    @property
    def capability_id(self) -> str:
        """The identifier the preferred provider is registered under."""

        return f"music.provider.{self.provider.lower()}"

    def provider_capability(self) -> Any:
        """The registered capability implementing the preferred provider.

        Looked up by a declared marker rather than by keyword similarity: a
        provider is a contract, and "the capability whose id contains spotify"
        is a guess that would eventually pick the wrong one.
        """

        wanted = self.provider.lower()
        try:
            manifests = self.capabilities.list()
        except Exception:
            return None
        for manifest in manifests:
            identifier = str(getattr(manifest, "capability_id", "")).lower()
            if identifier == self.capability_id:
                return manifest
        for manifest in manifests:
            identifier = str(getattr(manifest, "capability_id", "")).lower()
            if identifier.startswith("music.provider.") and wanted in identifier:
                return manifest
        return None

    def retired_capability(self) -> Any:
        """A version of the provider that exists but has been disabled.

        Distinguishes "there has never been one" from "there was one and it
        was withdrawn". Saying "I have no capability yet, building one now"
        while actually repairing a disabled version is a small lie, and it also
        misdescribes what will happen -- a rebuild starts from the installed
        source, not from nothing.
        """

        registry = getattr(self.capabilities, "registry", None)
        if registry is None:
            return None
        try:
            manifest = registry.get(self.capability_id)
        except Exception:
            return None
        if manifest is None or str(getattr(manifest, "status", "")) == "active":
            return None
        return manifest

    # -- credentials ------------------------------------------------------

    def credentials(self) -> tuple[dict[str, str], Any]:
        """The provider's credentials, or the requirement that is unmet."""

        if self.secrets is None or self.provider != "spotify":
            return {}, None
        secret = self.secrets.read("spotify", ("client_id", "client_secret"), env_prefix="SPOTIFY")
        if secret.present:
            return dict(secret.values), None
        return {}, self.secrets.requirement(
            "spotify",
            ("client_id", "client_secret"),
            env_prefix="SPOTIFY",
            how=(
                "Spotify needs one-time app credentials before it can look a song up by name.\n"
                "  1. https://developer.spotify.com/dashboard  ->  log in  ->  Create app\n"
                "  2. Any name; redirect URI http://localhost:8420/callback; tick Web API\n"
                "  3. Settings -> copy the Client ID and reveal the Client secret\n"
                "  4. Put them in {path}, or set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET\n"
                "No Premium and no login: these can search, not play. Playback stays in the "
                "desktop app."
            ),
        )

    # -- running ----------------------------------------------------------

    def run(self, request: MusicRequest) -> MusicOutcome:
        """Execute a music request and verify it against the operating system."""

        started = time.perf_counter()
        manifest = self.provider_capability()
        if manifest is None:
            return MusicOutcome(
                receipt=failed(
                    f"music.{request.action}",
                    "music.resolver",
                    f"no verified capability provides {self.provider!r} yet",
                    request=request.query or request.action,
                    provider=self.provider,
                ),
                gap=True,
            )

        capability_id = str(manifest.capability_id)
        payload: dict[str, Any] = {"action": request.action, "query": request.query,
                                   "output": self.output}

        # Credentials are only needed to turn a name into a track. Transport
        # commands must keep working without them, so this is checked per
        # action rather than up front.
        requirement = None
        if request.action in {"play", "search"}:
            # Provider-level guard, independent of whatever routed the request
            # here: prose is never submitted to a media provider as a search.
            why = prose_reason(request.query)
            if why:
                return MusicOutcome(
                    receipt=failed(
                        f"music.{request.action}", capability_id,
                        f"refused to search {self.provider}: {why}",
                        request=request.query[:200], provider=self.provider, guard="prose",
                    ),
                    capability_id=capability_id,
                )
            values, requirement = self.credentials()
            if requirement is not None:
                return MusicOutcome(
                    receipt=failed(
                        f"music.{request.action}",
                        capability_id,
                        requirement.describe().format(path=requirement.path),
                        request=request.query,
                        provider=self.provider,
                        missing_credential=requirement.name,
                    ),
                    capability_id=capability_id,
                    requirement=requirement,
                )
            payload.update(values)

        before = self.session.read(app="spotify" if self.provider == "spotify" else "")
        try:
            execution = self.capabilities.execute(capability_id, payload)
        except Exception as exc:
            return MusicOutcome(
                receipt=failed(f"music.{request.action}", capability_id,
                               f"{type(exc).__name__}: {exc}", request=request.query),
                capability_id=capability_id,
            )

        output = dict(getattr(execution, "output", {}) or {})
        if not getattr(execution, "ok", False):
            reason = (getattr(execution, "error", "")
                      or str(output.get("error", "the provider failed")))
            return MusicOutcome(
                receipt=failed(
                    f"music.{request.action}", capability_id, reason,
                    request=request.query, provider=self.provider, output=output,
                ),
                capability_id=capability_id,
                defect=(
                    ""
                    if is_environmental(reason)
                    else defect_report(request, reason, before, self.session.read(
                        app="spotify" if self.provider == "spotify" else ""))
                ),
            )

        after = self._settled(request)
        verifications = self.verify(request, before, after, output)
        ok = all(item.passed for item in verifications)
        # A provider that ran cleanly and did not do what it said is broken in
        # the way that matters most, so it is a defect rather than a bad day.
        defect = "" if ok else (
            f"run({{'action': '{request.action}'}}) reported success, but Windows disagrees: "
            + "; ".join(f"{item.check} -> {item.observed}" for item in verifications if not item.passed)
        )
        return MusicOutcome(
            receipt=Receipt(
                kind=f"music.{request.action}",
                executor=capability_id,
                ok=ok,
                request=request.query or request.action,
                detail=self._headline(request, after, ok),
                evidence={
                    "provider": self.provider,
                    "output_device": self.output,
                    "query": request.query,
                    "track": f"{after.title} - {after.artist}".strip(" -"),
                    "app": after.app,
                    "status": after.status,
                    "position_seconds": round(after.position_seconds, 2),
                    "provider_output": {k: v for k, v in output.items() if k not in {"client_secret"}},
                },
                verifications=tuple(verifications),
                duration_seconds=time.perf_counter() - started,
            ),
            capability_id=capability_id,
            defect=defect,
            detail=output,
        )

    def _settled(self, request: MusicRequest) -> Any:
        """Read the session, giving a starting track a moment to appear.

        Spotify reports the new title a beat after it accepts the URI, so
        reading once and immediately would record the previous track and call
        the action a failure. Polling is honest here; sleeping a fixed second
        and hoping would not be.
        """

        app = "spotify" if self.provider == "spotify" else ""
        deadline = time.time() + (6.0 if request.action in {"play", "next", "previous"} else 2.0)
        state = self.session.read(app=app)
        while time.time() < deadline:
            if request.action in {"pause"} and state.paused:
                return state
            if request.action in {"play", "resume", "next", "previous"} and state.playing:
                return state
            if request.action in {"current", "search"}:
                return state
            time.sleep(0.4)
            state = self.session.read(app=app)
        return state

    def verify(self, request: MusicRequest, before: Any, after: Any, output: dict[str, Any]) -> list[Verification]:
        """Ask Windows whether the thing the user asked for actually happened."""

        checks: list[Verification] = []
        checks.append(
            Verification(
                check="a media session exists",
                passed=bool(after.ok),
                observed=after.describe(),
                expected="a player registered with Windows",
            )
        )
        if not after.ok:
            return checks

        if self.provider == "spotify":
            checks.append(
                Verification(
                    check="the player is Spotify",
                    passed=after.is_spotify,
                    observed=after.app or "unknown",
                    expected="Spotify",
                )
            )

        if request.action in {"play", "resume", "next", "previous"}:
            checks.append(
                Verification(
                    check="Windows reports playback running",
                    passed=after.playing,
                    observed=after.status,
                    expected="Playing",
                )
            )
        elif request.action == "pause":
            checks.append(
                Verification(
                    check="Windows reports playback paused",
                    passed=after.paused,
                    observed=after.status,
                    expected="Paused",
                )
            )

        if request.action == "play" and request.query:
            # REQUEST INTENT vs RESOLVED TARGET vs PLAYBACK STATE are three
            # different facts.  "Rammstein ohne mich" resolved by the provider
            # to the track "Ohne dich" (Rammstein) and playing exactly that is
            # a SUCCESS -- the old title-only comparison called it a failure
            # while Windows reported the resolved track playing (2026-09-01).
            resolved_title = str(output.get("title") or (output.get("now_playing") or {}).get("title") or "")
            resolved_artist = str(output.get("artist") or (output.get("now_playing") or {}).get("artist") or "")
            if resolved_title:
                checks.append(Verification(
                    check="the resolved track is playing",
                    passed=self._covers(resolved_title, after.title)
                    and (not resolved_artist or self._covers(resolved_artist, after.artist)),
                    observed=f"resolved to {resolved_title} - {resolved_artist}; playing {after.title} - {after.artist}",
                    expected=f"{resolved_title} - {resolved_artist}",
                ))
            checks.append(self._track_matches(request.query, after, kind=getattr(request, "kind", "any"), artist=getattr(request, "artist", "")))
        if request.action in {"next", "previous"}:
            checks.append(
                Verification(
                    check="the track changed",
                    passed=bool(before.ok and after.title and after.title != before.title),
                    observed=f"{before.title!r} -> {after.title!r}",
                    expected="a different track",
                )
            )
        return checks

    @staticmethod
    def _covers(wanted_text: str, have_text: str) -> bool:
        """Most of the wanted words appear in the observed field."""
        want = {w for w in re.findall(r"\w+", _fold(wanted_text)) if len(w) > 2}
        have = {w for w in re.findall(r"\w+", _fold(have_text)) if len(w) > 2}
        if not want:
            return bool(have) is False or True
        return len(want & have) >= max(1, round(len(want) * 0.6))

    @staticmethod
    def _track_matches(query: str, after: Any, *, kind: str = "any", artist: str = "") -> Verification:
        """Does what is playing satisfy what was asked for -- by the *kind* of request?

        Token overlap rather than equality: Spotify returns "Lose Yourself -
        From '8 Mile' Soundtrack" for "Lose Yourself", and calling that a
        mismatch would fail a request that succeeded.  What the tokens are
        compared against depends on the typed target: an ARTIST request is
        satisfied by the artist field ("Rammstein" playing "Sonne"), a TRACK
        request by the title (and the named artist, when there is one), an
        ALBUM/PLAYLIST/TOP_TRACK request by the artist or the title, ANY by
        either.
        """

        def tokens(text: str) -> set[str]:
            return {word for word in re.findall(r"\w+", _fold(text)) if len(word) > 2}

        wanted = tokens(query)
        if not wanted:
            return Verification("the requested music is playing", False, observed="no query to compare", expected="a name")
        title_tokens, artist_tokens = tokens(after.title or ""), tokens(after.artist or "")
        artist_wanted = tokens(artist) if artist else set()
        observed = f"{after.title} - {after.artist}"

        def most(want: set[str], have: set[str]) -> bool:
            hits = want & have
            return len(hits) >= max(1, round(len(want) * 0.6))

        if kind == "artist" or kind == "top_track":
            name = artist_wanted or wanted
            passed = most(name, artist_tokens)
            return Verification(check="the requested artist is playing", passed=passed,
                                observed=f"{observed} (artist tokens {sorted(artist_tokens)} vs {sorted(name)})", expected=f"artist {artist or query}")
        if kind == "track" and artist_wanted:
            # an explicitly named artist ("Titel von X") is checked strictly
            title_ok = most(wanted - artist_wanted or wanted, title_tokens)
            artist_ok = most(artist_wanted, artist_tokens)
            return Verification(check="the requested track is playing", passed=title_ok and artist_ok,
                                observed=f"{observed} (title {'ok' if title_ok else 'differs'}, artist {'ok' if artist_ok else 'differs'})",
                                expected=f"{query} by {artist}")
        if kind == "track":
            # no explicit artist: the words may name the artist as well as the
            # title ("Rammstein ohne mich"), so both fields may satisfy them
            passed = most(wanted, title_tokens | artist_tokens)
            return Verification(check="the requested track is playing", passed=passed,
                                observed=f"{observed} (matched {sorted(wanted & (title_tokens | artist_tokens))} of {sorted(wanted)} in title or artist)",
                                expected=query)
        if kind in {"album", "playlist"}:
            passed = most(wanted, title_tokens | artist_tokens)
            return Verification(check=f"the requested {kind} is playing", passed=passed, observed=observed, expected=query)
        passed = most(wanted, title_tokens | artist_tokens)
        return Verification(check="the requested track is playing", passed=passed,
                            observed=f"{observed} (matched {sorted(wanted & (title_tokens | artist_tokens))} of {sorted(wanted)} in title or artist)", expected=query)

    def _headline(self, request: MusicRequest, after: Any, ok: bool) -> str:
        track = f"{after.title} - {after.artist}".strip(" -")
        if not ok:
            return f"{request.action} did not take effect (Windows reports: {after.describe()})"
        if request.action == "current":
            return f"now playing: {track} [{after.status}]"
        if request.action == "pause":
            return f"paused: {track}"
        if request.action == "play" and request.query and _fold(request.query) not in _fold(track):
            # say what the request was resolved to, so a resolved title is
            # never mistaken for a wrong one
            return f"playing: {track} (resolved from '{request.query}')"
        return f"playing: {track}"


# --------------------------------------------------------------------------
# What a provider capability has to be
# --------------------------------------------------------------------------
#
# These describe the *contract and the machine*, not the implementation. The
# builder is told what run() must accept, what counts as done, and what is
# actually installed on this computer -- and then has to work out how. Handing
# it the code would make the acquisition a formality; handing it nothing repeats
# the six previous music attempts, four of which failed on facts about the
# environment nobody had told the model (packages that were not installed,
# Jarvis tools that are not importable from generated source).

def environment_facts() -> list[str]:
    """What is true about this machine, discovered rather than assumed."""

    import shutil
    from pathlib import Path

    facts: list[str] = []
    store_app = Path.home() / "AppData/Local/Microsoft/WindowsApps/Spotify.exe"
    facts.append(
        f"The Spotify desktop application is installed at {store_app} "
        if store_app.exists() else "The Spotify desktop application was not found at the usual path "
    )
    facts.append(
        "The 'spotify:' URI protocol handler is registered on this machine, so handing a "
        "'spotify:track:<id>' URI to the shell opens it in the desktop app and starts that track."
    )
    facts.append(
        "Neither the 'winrt' nor the 'winsdk' Python package is installed, and installing global "
        "packages is not permitted. Windows APIs must be reached another way."
    )
    facts.append(
        f"PowerShell is {'available at ' + str(shutil.which('powershell')) if shutil.which('powershell') else 'NOT available'}. "
        "PowerShell can project WinRT types, which is how Windows' own media session "
        "(Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager) can be reached "
        "without any Python package."
    )
    facts.append(
        "Spotify's Web API requires a token for every endpoint. The client-credentials flow "
        "(POST https://accounts.spotify.com/api/token with grant_type=client_credentials and the "
        "app's client id and secret) yields a token that can SEARCH but cannot control playback. "
        "That is sufficient: search resolves a name to a track id, and the desktop app plays it."
    )
    facts.append(
        "The client id and secret arrive in the payload as 'client_id' and 'client_secret'. "
        "They are never in the environment: the capability runner strips all but a short "
        "allow-list of environment variables."
    )
    return facts


def provider_goal(provider: str = "spotify") -> str:
    """The requirement a provider capability has to meet."""

    actions = ", ".join(ACTIONS)
    facts = "\n".join(f"- {fact}" for fact in environment_facts())
    return (
        f"Build a {provider} music provider capability for Windows.\n\n"
        f"run(payload) must accept payload['action'], one of: {actions}.\n\n"
        # The example is deliberately not one of the tracks acceptance uses. An
        # example in the brief is the first thing a struggling implementation
        # reaches for to special-case, and naming the test track here would let
        # it pass by recognising the string rather than by working.
        "  play      payload['query'] is free text naming a song and usually an artist, for\n"
        "            example 'Take Five Dave Brubeck'. Resolve it to a specific track and START\n"
        "            THAT TRACK PLAYING on this computer. Return the track title and artist.\n"
        "  pause     pause whatever is currently playing.\n"
        "  resume    resume playback.\n"
        "  next      skip to the next track.\n"
        "  previous  go back to the previous track.\n"
        "  current   report what is playing right now, read from the system, not remembered.\n"
        "  search    resolve payload['query'] to a track without playing it.\n\n"
        "Return a dict with 'ok' (bool), 'error' (str when not ok), and for play/search/current\n"
        "also 'title', 'artist' and 'track_id' where known.\n\n"
        "It must ACTUALLY PLAY. A function that returns a description of playing music, or a\n"
        "'Dry run:' message, or that reports ok without anything happening, is a failure. It is\n"
        "checked by reading Windows' own media session afterwards and by an audio meter.\n\n"
        "Never substitute another source. If the requested track cannot be resolved or played,\n"
        "return ok=False with the reason. Do not fall back to YouTube, a web search, a browser,\n"
        "a local file, or a generated tone.\n\n"
        f"Facts about this machine:\n{facts}\n"
    )


def provider_keywords(provider: str = "spotify") -> list[str]:
    """The words a user will actually say.

    This is not decoration.  It is what the knowledge graph indexes, and it is
    the only reason a later "spiel was von den Beatles" finds a capability whose
    identifier says *spotify* -- lexical matching cannot bridge music to Spotify
    on its own.
    """

    return [provider, "music", "musik", "song", "lied", "track", "titel", "play",
            "spielen", "abspielen", "playback", "audio", "pause", "skip"]


def provider_constraints(provider: str = "spotify") -> list[str]:
    """Rules the implementation must obey, from six previous failed attempts."""

    return [
        "main.py must expose run(payload: dict) -> dict and a module-level INPUT_SCHEMA.",
        "Only the Python standard library. No pip installs, no third-party imports.",
        "ZEUS's own tools (media_control, launch_application, find_media, ...) are NOT "
        "importable from this file. Use the standard library and PowerShell.",
        "Read every payload value with .get(); never index a payload key directly.",
        "Honour payload.get('dry_run'): perform every check and report what would happen, "
        "without starting playback.",
        "Never raise for an expected failure. Return {'ok': False, 'error': '...'}.",
        f"Do not substitute a different provider for {provider} under any circumstances.",
    ]


def provider_acceptance(python: str | None = None) -> list[tuple[str, list[str]]]:
    """Commands that decide whether an implementation is real.

    Handed to an expert as its acceptance criteria and re-run here afterwards,
    so the expert's own report is never what promotes anything.
    """

    import sys

    interpreter = python or sys.executable
    return [
        (
            "run() exists, declares its schema, and survives a dry run",
            [interpreter, "-c",
             "import main;"
             "assert callable(main.run), 'run is not callable';"
             "assert isinstance(getattr(main, 'INPUT_SCHEMA', None), dict), 'no INPUT_SCHEMA';"
             "r = main.run({'action': 'current', 'dry_run': True});"
             "assert isinstance(r, dict), r;"
             "print('CONTRACT_OK')"],
        ),
        (
            "an unsupported action fails honestly instead of raising",
            [interpreter, "-c",
             "import main;"
             "r = main.run({'action': 'teleport'});"
             "assert isinstance(r, dict) and r.get('ok') is False, r;"
             "assert r.get('error'), 'a failure with no reason';"
             "print('HONEST_FAILURE_OK')"],
        ),
        (
            "reading the current track comes from the system, not from memory",
            [interpreter, "-c",
             "import main;"
             "r = main.run({'action': 'current'});"
             "assert isinstance(r, dict), r;"
             "assert 'ok' in r, r;"
             "print('CURRENT_OK', r.get('title'), r.get('artist'))"],
        ),
        ("it really controls playback: resume then pause, each confirmed by Windows",
         [interpreter, "-c", _PLAYBACK_CHECK]),
    ]


#: The check that exercises the search path against the real provider.
#:
#: Its absence is why a broken search shipped. The playback gate proved the
#: capability could control a player and the acceptance commands proved it
#: handled contracts and unknown actions honestly -- and nothing at all
#: exercised name-to-track resolution, so a Spotify provider that could not
#: search was registered as verified. Real execution then found what no gate
#: had asked: Spotify rejects `limit=20` with "Invalid limit" despite
#: documenting a maximum of 50.
#:
#: It names no track. The query is a common word and the assertion is only that
#: *some* track came back with an id -- enough to prove the request, the auth
#: and the parsing all work, and not enough for an implementation to pass by
#: recognising a string.
#:
#: Credentials are read from the secret store by the check itself rather than
#: baked into the command, so they never reach a project record, a log or a
#: process list. The check skips rather than fails when there are none: a
#: machine without credentials cannot search, and holding that against the
#: capability would be blaming it for its environment.
_SEARCH_CHECK = (
    "import sys;"
    "import main;"
    "sys.path.append(r'" + str(Path(__file__).resolve().parent.parent) + "');"
    "from runtime.secrets import SecretStore;"
    "import pathlib;"
    "root = pathlib.Path(r'" + str(Path(__file__).resolve().parent.parent) + "') / 'data' / 'jarvis' / 'secrets';"
    "s = SecretStore(root).read('spotify', ('client_id','client_secret'), env_prefix='SPOTIFY');"
    "print('SEARCH_SKIPPED no credentials configured') if not s.present else None;"
    "sys.exit(0) if not s.present else None;"
    "r = main.run({'action':'search','query':'love','client_id':s.get('client_id'),"
    "'client_secret':s.get('client_secret')});"
    "assert isinstance(r, dict), r;"
    "assert r.get('ok') is True, 'search failed: ' + str(r.get('error'));"
    "assert r.get('track_id'), 'search returned no track id: ' + str(r);"
    "assert r.get('title'), 'search returned no title: ' + str(r);"
    "print('SEARCH_OK', r.get('title'), '-', r.get('artist'))"
)


#: The gate for the defect that six other gates could not see.
#:
#: `playback` proves resume and pause. `search` proves a name resolves to a
#: track. Neither proves the thing a user actually asks for: start THIS track,
#: now, while something else is playing. A provider passed all six while being
#: unable to do exactly that -- handing `spotify:track:<id>` to a player that is
#: already playing was silently ignored and the existing queue carried on.
#:
#: It names no track. It asks the provider to search for a common word, takes
#: whatever comes back as the expectation, and then requires that to be what
#: Windows reports playing. The provider supplies its own answer key and still
#: cannot pass by recognising a string.
#:
#: It deliberately starts from a PLAYING state, because that is the state the
#: defect lives in and the one every other gate happened to avoid.
_PLAY_WHILE_PLAYING_CHECK = (
    "import sys, time;"
    "import main;"
    "sys.path.append(r'" + str(Path(__file__).resolve().parent.parent) + "');"
    "from runtime.secrets import SecretStore;"
    "from tools import media_session;"
    "import pathlib;"
    "root = pathlib.Path(r'" + str(Path(__file__).resolve().parent.parent) + "') / 'data' / 'jarvis' / 'secrets';"
    "s = SecretStore(root).read('spotify', ('client_id','client_secret'), env_prefix='SPOTIFY');"
    "print('SWITCH_SKIPPED no credentials configured') if not s.present else None;"
    "sys.exit(0) if not s.present else None;"
    "creds = {'client_id': s.get('client_id'), 'client_secret': s.get('client_secret')};"
    "found = main.run(dict(creds, action='search', query='love'));"
    "assert found.get('ok') and found.get('title'), 'search failed: ' + str(found);"
    "wanted = found['title'];"
    "main.run({'action': 'resume'});"
    "time.sleep(2.0);"
    "playing_before = media_session.read(app='spotify');"
    "assert playing_before.playing, 'could not get the player into a playing state to test against';"
    "r = main.run(dict(creds, action='play', query=wanted));"
    "time.sleep(3.0);"
    "after = media_session.read(app='spotify');"
    "assert after.ok, 'no media session after play: ' + after.error;"
    "assert after.playing, 'not playing after play: ' + after.status;"
    "wl = [w for w in wanted.lower().split() if len(w) > 2];"
    "hit = sum(1 for w in wl if w in (after.title or '').lower());"
    "assert hit >= max(1, (len(wl) * 6) // 10), "
    "'asked for ' + repr(wanted) + ' while ' + repr(playing_before.title) + ' was playing, "
    "but Windows reports ' + repr(after.title);"
    "print('SWITCH_OK', playing_before.title, '->', after.title)"
)


def provider_extra_checks(python: str | None = None) -> list[Any]:
    """The playback check, as a gate on the *local* build as well.

    Without this the local acquisition is judged only by the four standard
    capability checks -- tests pass, run() returns a dict, the marker is gone,
    no undefined names -- every one of which a capability that does nothing at
    all can satisfy.  That is precisely how this project once registered a
    music capability whose every branch returned ``{"message": "Dry run: ..."}``.

    Handed to ``ensure()`` as an ``extra_check``, so it is simultaneously an
    acceptance criterion the build loop can watch fail and a verification
    re-run independently afterwards.
    """

    import sys

    from capabilities.service import CapabilityCheck

    interpreter = python or sys.executable
    return [
        CapabilityCheck(
            name="playback",
            text=("run({'action':'resume'}) actually starts playback and "
                  "run({'action':'pause'}) actually stops it, both confirmed by reading "
                  "Windows' media session afterwards"),
            command=(interpreter, "-c", _PLAYBACK_CHECK),
        ),
        CapabilityCheck(
            name="search",
            text=("run({'action':'search', 'query': ...}) resolves free text to a real track "
                  "id and title through the provider's own API"),
            command=(interpreter, "-c", _SEARCH_CHECK),
        ),
        CapabilityCheck(
            name="switch",
            text=("run({'action':'play', 'query': ...}) starts the REQUESTED track even when "
                  "the player is already playing something else, confirmed by reading Windows"),
            command=(interpreter, "-c", _PLAY_WHILE_PLAYING_CHECK),
        ),
    ]


#: The check that separates controlling a player from describing one.
#:
#: It resumes, asks Windows whether playback started, pauses, and asks again.
#: Deliberately not a play-by-name: that needs credentials and would have to
#: name a track, and naming a track in a check is how a capability learns to
#: recognise the test instead of doing the work. Resume/pause needs neither and
#: cannot be faked -- the operating system is the one being asked.
#:
#: `main` is imported before the repository root joins sys.path. An earlier
#: version of this project's audio check did it the other way round, and
#: `import main` found Jarvis' own top-level main.py instead of the capability,
#: failing the check for a reason that had nothing to do with the capability.
_PLAYBACK_CHECK = (
    "import sys, time;"
    "import main;"
    "sys.path.append(r'" + str(Path(__file__).resolve().parent.parent) + "');"
    "from tools import media_session;"
    "before = media_session.read(app='spotify');"
    "r1 = main.run({'action': 'resume'});"
    "time.sleep(1.5);"
    "playing = media_session.read(app='spotify');"
    "r2 = main.run({'action': 'pause'});"
    "time.sleep(1.5);"
    "paused = media_session.read(app='spotify');"
    "assert playing.ok, 'no media session after resume: ' + playing.error;"
    "assert playing.playing, 'resume did not start playback: ' + playing.status;"
    "assert paused.paused, 'pause did not stop playback: ' + paused.status;"
    "print('PLAYBACK_CONTROL_OK', playing.title, '|', paused.status)"
)


def compose(outcome: MusicOutcome, *, language: str = "") -> str:
    """The sentence the user reads, built from the receipt.

    Same rule as every other action: the model does not write this, because
    this is the sentence that asserts something about the world.
    """

    german = language.startswith("de")
    receipt = outcome.receipt
    evidence = receipt.evidence
    track = str(evidence.get("track", "")).strip()

    if receipt.verified:
        kind = receipt.kind.split(".", 1)[-1]
        if kind == "pause":
            head = f"Pausiert: {track}" if german else f"Paused: {track}"
        elif kind == "current":
            head = (f"Es laeuft: {track}" if german else f"Now playing: {track}")
        else:
            head = (f"Laeuft jetzt: {track}" if german else f"Now playing: {track}")
        head += f"  ({evidence.get('app', '')})"
    elif outcome.requirement is not None:
        head = receipt.detail
    elif outcome.gap:
        head = receipt.detail
    else:
        head = (f"Fehlgeschlagen. {receipt.detail}" if german else f"That failed. {receipt.detail}")

    lines = [head]
    if receipt.verifications:
        lines += ["", ("Belege:" if german else "Evidence:")]
        lines += [f"  - {line}" for line in receipt.evidence_lines()]
    lines += ["", f"receipt {receipt.id}"]
    return "\n".join(lines)
