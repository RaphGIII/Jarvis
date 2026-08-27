"""What kind of request this is, decided before anything answers it.

The live product had exactly one path for an incoming message: compose a prompt,
stream the model's reply, ship it.  That is correct for "how does inheritance
work" and catastrophic for "create this file", because the second has a truth
condition and the first does not.  A model asked to do something it cannot do
does not decline; it narrates the thing being done.

So every request is classified first, and the classification decides which
machinery answers it.  Conversation keeps the fast path it has.  Anything with a
truth condition goes somewhere that can actually establish it.

Two decisions worth defending.

*The classifier is deterministic, and cheap.*  Asking a model which path to take
costs a model call on every message, including the ones whose whole value is
being instant.  Keyword matching gets the common cases right for nothing, and
:mod:`service.actions` re-checks with the model anyway when it plans the action
-- so a false positive here costs one short generation, not a wrong answer.

*It is deliberately biased toward ACTION.*  The two error directions are not
symmetric.  Classifying conversation as an action costs a few hundred
milliseconds and produces a correct answer anyway, because the planner is
allowed to decline and fall back.  Classifying an action as conversation
produces a confident lie about the user's filesystem.  When a verb could go
either way -- "schreibe mir", "make me" -- it goes to ACTION.

The residual risk, stated rather than hidden: this cannot catch every phrasing.
That is why it is not the only defence.  :mod:`service.claims` checks the
*answer* for a success claim with no receipt behind it, so a request this module
misclassifies still cannot reach the user as a fabricated success.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class Intent(str, Enum):
    """What the request wants, as far as routing is concerned."""

    #: A question or a remark.  No truth condition about the world.
    CONVERSATION = "conversation"
    #: A question *about this system's real state*: projects, capabilities,
    #: diagnostics.  Must be answered from the registries, not from the model.
    READ = "read"
    #: A side effect on the machine.  Must produce a receipt.
    ACTION = "action"
    #: Durable work that outlives a turn.  Becomes a real project record.
    PROJECT = "project"
    #: "Learn to do X" -- acquiring something the system cannot currently do.
    CAPABILITY = "capability"
    #: Play, pause, skip, what is playing.  A side effect on the world like any
    #: other, routed to whichever provider the user prefers -- this layer never
    #: learns which one that is.
    MUSIC = "music"
    #: "Change something about yourself."  Becomes a persistent self-development
    #: mission: worktree, BUILD_LOCAL, verification, promotion, restart.
    SELF_DEVELOPMENT = "self_development"
    #: A change to the owner core (identity, personality, policy).  Never a
    #: SelfDev mission: a protected owner transaction with explicit approval.
    OWNER_CONFIG = "owner_config"
    #: The owner says a previous reading was wrong.  Handled through the
    #: correction memory, not by acting on the sentence.
    CORRECTION = "correction"

    @property
    def has_side_effect(self) -> bool:
        """Whether answering this may change the world."""

        return self in {Intent.ACTION, Intent.PROJECT, Intent.CAPABILITY, Intent.MUSIC, Intent.SELF_DEVELOPMENT, Intent.OWNER_CONFIG}

    @property
    def needs_receipt(self) -> bool:
        """Whether a claim of success here requires executed evidence."""

        return self.has_side_effect


@dataclass(frozen=True)
class Classification:
    intent: Intent
    reason: str
    #: The phrase that decided it, for diagnostics and for tuning.
    matched: str = ""
    #: The top-level route (:mod:`service.routing`) this was derived from.
    route: Any = None

    @property
    def top_level(self) -> str:
        return self.route.intent.value if self.route is not None else ""

    @property
    def confidence(self) -> str:
        return self.route.confidence if self.route is not None else ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"intent": self.intent.value, "reason": self.reason, "matched": self.matched}
        if self.route is not None:
            out["route"] = self.route.to_dict()
        return out


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

#: Questions about what this system actually is or has.  Checked first, because
#: "was kannst du" is a question about capabilities, not a request to acquire
#: one, and the acquisition hints below would otherwise swallow it.
READ_HINTS = (
    "welche projekte",
    "meine projekte",
    "zeig mir die projekte",
    "zeige die projekte",
    "liste die projekte",
    "gibt es ein projekt",
    "welche faehigkeiten",
    "welche fähigkeiten",
    "welche capabilities",
    "welche funktionen hast du",
    "was kannst du",
    "was kannst du alles",
    "wozu bist du faehig",
    "wozu bist du fähig",
    "welche sind verifiziert",
    "was ist verifiziert",
    "verifizierte faehigkeiten",
    "verifizierte fähigkeiten",
    "wie ist dein status",
    "dein status",
    "diagnose",
    "diagnostik",
    "which projects",
    "my projects",
    "list projects",
    "list my projects",
    "show me my projects",
    "which capabilities",
    "what capabilities",
    "what can you do",
    "which are verified",
    "what is verified",
    "verified capabilities",
    "your status",
    "diagnostics",
)

#: Verbs that change something.  A match here is enough on its own: requiring a
#: matching object noun as well is what lets "leg das in zeus_test.txt ab" slip
#: through as conversation, and a missed action is the expensive error.
ACTION_VERBS = (
    "erstelle",
    "erstell ",
    "erzeuge",
    "leg an",
    "lege an",
    "anlegen",
    "erstellen",
    "schreibe",
    "schreib ",
    "speichere",
    "speicher ",
    "abspeichern",
    "speichern",
    "sichere",
    "loesche",
    "lösche",
    "loeschen",
    "löschen",
    "entferne",
    "benenne",
    "umbenennen",
    "aendere",
    "ändere",
    "bearbeite",
    "fuege",
    "füge",
    "hinzufuegen",
    "hinzufügen",
    "lege ab",
    "leg ab",
    "create",
    "make a file",
    "make the file",
    "write",
    "save",
    "store it",
    "delete",
    "remove",
    "rename",
    "edit",
    "modify",
    "append",
    "put it in",
)

#: Nouns that make an action request unambiguous.  Not required, but a match
#: raises confidence and is reported in the reason.
ACTION_OBJECTS = (
    "datei",
    "dateien",
    "ordner",
    "verzeichnis",
    "projekt",
    "notiz",
    "dokument",
    "file",
    "folder",
    "directory",
    "project",
    "note",
    "document",
)

#: German separable verbs: "Leg eine Notiz an", "Speichere das ab", "Lege den
#: Ordner an". The particle lands at the end of the clause, so a substring
#: list of verbs never sees it, and "Leg eine Notiz an" read as conversation
#: -- where the model then invented a "notes database" it had committed to.
SEPARABLE_ACTION = re.compile(
    r"\b(leg|lege|legst|legen|speicher|speichere|speichern|schreib|schreibe|schreiben|fueg|fuege|füge|fügen|fuegen)\b"
    r"[^.!?\n]{0,60}?\b(an|ab|auf|hinzu|ein|rein)\b",
    re.I,
)

#: A filename is about as clear a side-effect signal as exists.
FILENAME = re.compile(r"\b[\w\-. ]{1,60}\.(txt|md|py|json|csv|yaml|yml|log|ini|cfg|html|js|toml)\b", re.I)

#: "Learn to do X".  Distinct from a READ question about capabilities, which is
#: why READ_HINTS is checked first.
CAPABILITY_HINTS = (
    "lerne",
    "bring dir bei",
    "bringe dir bei",
    "learn how to",
    "teach yourself",
    "acquire the ability",
    "new capability",
    "neue faehigkeit",
    "neue fähigkeit",
)

#: A change to ZEUS itself: its interface, its code, its behaviour.  Checked
#: before PROJECT and ACTION because "ändere dein Auge" contains an action verb
#: and "implementiere das in deiner UI" contains a project verb, and both are
#: about this system rather than about a file or an app.
SELF_DEVELOPMENT_HINTS = (
    "an dir selbst",
    "an dir ",
    "dich selbst",
    "verändere dich",
    "veraendere dich",
    "ändere dich",
    "aendere dich",
    "verbessere dich",
    "aktualisiere dich",
    "entwickle dich",
    "deine oberfläche",
    "deine oberflaeche",
    "deine ui",
    "deiner ui",
    "dein interface",
    "deinem interface",
    "dein auge",
    "deinem auge",
    "deinen code",
    "deinem code",
    "dein eigener code",
    "deinen eigenen code",
    "deinem eigenen code",
    "dein verhalten",
    "in dir ",
    "bei dir ",
    "change yourself",
    "about yourself",
    "improve yourself",
    "update yourself",
    "modify yourself",
    "your own code",
    "your codebase",
    "your interface",
    "your ui",
    "your eye",
    "next to your eye",
    "in your ui",
    "your behaviour",
    "your behavior",
    "self-development",
    "selbstentwicklung",
)

#: Durable engineering work.  Kept aligned with :class:`brain.router.BrainRouter`
#: so the CLI and the web service agree about what a project is.
PROJECT_HINTS = (
    "implementiere",
    "implementieren",
    "programmiere",
    "schreibe code",
    "schreib code",
    "baue eine app",
    "baue mir",
    "entwickle",
    "refactor",
    "refaktor",
    "build an app",
    "build me",
    "write a program",
    "write a script",
    "implement",
    "fix the bug",
    "fixe den bug",
    "behebe den bug",
    "debugge",
    "debug this",
    "run the tests",
    "self-develop",
    "self develop",
)


def classify(text: str, *, corrections: Iterable[Any] = (), capability_names: Iterable[str] = ()) -> Classification:
    """Decide what kind of request this is.

    Two stages, in a fixed order.  First :func:`service.routing.route` decides
    the *top-level* intent from the request's operation and object -- whether
    it acts on the world, changes ZEUS itself, acquires an ability, and so on
    -- with owner corrections consulted before that choice.  Only when the top
    level says the request is a real-world action or conversation is any
    domain parser (music first) allowed to claim it.  Spotify's verb list never
    gets to decide whether a paragraph is self-development.

    Order matters and is not arbitrary:

    1. **READ** before everything, because questions about the system's own
       state read as capability or project talk to every later rule.
    2. **CAPABILITY** and **PROJECT** before ACTION, because "implementiere X"
       contains no action verb but is unmistakably durable work, while
       "erstelle eine App" contains one and is not a single side effect.
    3. **ACTION** last among the side-effecting kinds, as the broad catch.
    """

    normalized = f" {(text or '').strip().lower()} "
    if not normalized.strip():
        return Classification(Intent.CONVERSATION, "empty message")

    from service.routing import TopLevelIntent, route

    top = route(text, corrections=corrections, capability_names=capability_names)

    # Questions about the system's own state are answered from the registries
    # whatever the top level made of them ("was kannst du" is not acquisition).
    for hint in READ_HINTS:
        if hint in normalized:
            return Classification(
                Intent.READ, f"asks about this system's own state: {hint!r}", matched=hint, route=top
            )

    # The top level has decided what this is.  Self-modification, acquisition,
    # owner-core changes and corrections are settled here and never reach a
    # domain parser.
    if top.intent in {TopLevelIntent.SELF_DEVELOPMENT, TopLevelIntent.CAPABILITY_REPAIR}:
        return Classification(Intent.SELF_DEVELOPMENT, top.reason, matched=top.intent.value, route=top)
    if top.intent is TopLevelIntent.OWNER_CONFIG_CHANGE:
        return Classification(Intent.OWNER_CONFIG, top.reason, matched=top.intent.value, route=top)
    if top.intent is TopLevelIntent.CAPABILITY_ACQUISITION:
        return Classification(Intent.CAPABILITY, top.reason, matched=top.intent.value, route=top)
    if top.intent is TopLevelIntent.OWNER_CORRECTION:
        return Classification(Intent.CORRECTION, top.reason, matched=top.intent.value, route=top)
    if top.intent is TopLevelIntent.RESEARCH:
        return Classification(Intent.CONVERSATION, top.reason, matched=top.intent.value, route=top)

    # Only now may a domain parser look.  Music is asked of the music module
    # rather than answered with a keyword list here: "Pause." and "Weiter." are
    # transport commands every later rule would miss or mistake for something
    # else, and the test for whether a sentence is about music belongs next to
    # the code that then has to parse it.
    if top.intent.domain_eligible:
        from service.music import understand

        heard = understand(text)
        if heard is not None:
            return Classification(Intent.MUSIC, heard.reason, matched=heard.action, route=top)

    # Legacy phrase hints, kept as a second opinion for self-development only
    # when the top level found no world object to compete with.
    if not normalized.rstrip().endswith("?") and top.intent is not TopLevelIntent.REAL_WORLD_ACTION:
        for hint in SELF_DEVELOPMENT_HINTS:
            if hint in normalized:
                return Classification(
                    Intent.SELF_DEVELOPMENT, f"asks to change this system itself: {hint!r}", matched=hint, route=top
                )

    for hint in CAPABILITY_HINTS:
        if hint in normalized:
            return Classification(
                Intent.CAPABILITY, f"asks to acquire an ability: {hint!r}", matched=hint
            , route=top)

    for hint in PROJECT_HINTS:
        if hint in normalized:
            return Classification(Intent.PROJECT, f"describes durable work: {hint!r}", matched=hint, route=top)

    filename = FILENAME.search(normalized)
    separable = SEPARABLE_ACTION.search(normalized)
    if separable:
        noun = next((word for word in ACTION_OBJECTS if word in normalized), "")
        return Classification(
            Intent.ACTION, f"separable side-effect verb {separable.group(0).strip()!r}" + (f" on a {noun}" if noun else ""),
            matched=separable.group(0).strip(), route=top,
        )
    for verb in ACTION_VERBS:
        if verb in normalized:
            noun = next((word for word in ACTION_OBJECTS if word in normalized), "")
            detail = f" on a {noun}" if noun else (f" naming {filename.group(0).strip()}" if filename else "")
            return Classification(
                Intent.ACTION, f"side-effect verb {verb.strip()!r}{detail}", matched=verb.strip()
            , route=top)

    if top.intent is TopLevelIntent.REAL_WORLD_ACTION and top.reading.action_verbs:
        # The top level saw an imperative on the world that no legacy phrase
        # list covers ("take a screenshot", "open Activity").  The planner can
        # still decline and fall back to conversation; a missed action cannot.
        return Classification(Intent.ACTION, top.reason, matched=top.reading.action_verbs[0], route=top)

    if filename:
        # A bare filename with no verb ("zeus_test.txt bitte mit Inhalt X") is
        # still far more likely to want a file than to want a conversation.
        return Classification(
            Intent.ACTION, f"names a file: {filename.group(0).strip()!r}", matched=filename.group(0).strip()
        , route=top)

    return Classification(Intent.CONVERSATION, "no side effect and no system-state question", route=top)
