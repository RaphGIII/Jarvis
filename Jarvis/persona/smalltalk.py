"""Small talk, answered as Zeus rather than as a system description.

"Wie geht es dir?" is a social question.  The truthful, natural answer is a
sentence about how things stand -- systems, missions, uptime -- said briefly
and warmly.  It is deterministic on purpose: the owner heard "Ich bin ein
autonomer Engineering-Assistent … keine Emotionen" from the model, and a
personality that depends on a 4B model's mood is not a personality.

Only phatic questions are caught here.  "Hast du wirklich menschliche
Gefühle?", "Was bist du technisch?" and everything else go to the model with
the personality prompt, where the honest answers belong.
"""

from __future__ import annotations

import random
import re

_PHATIC = (
    re.compile(r"^\s*(?:hallo|hi|hey|servus|moin|guten (?:morgen|tag|abend)|na)?[\s,!.]*(?:zeus[\s,!.]*)?(?:und\s+)?(?:wie geht(?:'?s| es) dir|wie geht es|wie gehts|alles (?:klar|gut|okay) bei dir|wie läuft'?s|wie läuft es|geht'?s dir gut|wie fühlst du dich|alles fit)[\s?!.]*$", re.I),
    re.compile(r"^\s*(?:hello|hi|hey)?[\s,!.]*(?:zeus[\s,!.]*)?(?:how are you(?: doing| today)?|how's it going|how is it going|you (?:good|okay|alright)|how do you feel(?: today)?)[\s?!.]*$", re.I),
)
# A bare greeting ("hi", "hallo") is left to the model: the owner may be
# opening a conversation, and a canned line there would sound canned.
#: Literal questions about feelings/consciousness are not small talk.
_LITERAL = re.compile(r"(wirklich|echt|tatsächlich|literally|really|actually).{0,30}(gefühl|emotion|bewusst|conscious|feel)", re.I)

#: "Wer bist du?" is a social question about identity, not a request for a
#: system description.  The live product answered it with "Ich bin dein
#: persönliches System. Keine Wahrnehmung, kein Gefühl." -- a 4B model
#: reading the identity preamble as a script.  Zeus introduces himself.
_IDENTITY = (
    re.compile(r"^\s*(?:hallo|hi|hey)?[\s,!.]*(?:zeus[\s,!.]*)?(?:und\s+)?(?:wer\s+bist\s+du(?:\s+(?:eigentlich|denn|genau|überhaupt))?|wie\s+hei(?:ß|ss)t\s+du|wer\s+bist\s+du\s+eigentlich|stell\s+dich\s+(?:kurz\s+)?vor|was\s+bist\s+du(?:\s+(?:eigentlich|denn|genau))?)[\s?!.]*$", re.I),
    re.compile(r"^\s*(?:hello|hi|hey)?[\s,!.]*(?:zeus[\s,!.]*)?(?:who\s+are\s+you(?:\s+(?:exactly|really|anyway))?|what(?:'s|\s+is)\s+your\s+name|introduce\s+yourself|what\s+are\s+you(?:\s+exactly)?)[\s?!.]*$", re.I),
)
# The identity firewall (§14-16).  Who ZEUS is, who built it, and whether it
# "is" some vendor's model are answered here, deterministically, for nothing:
# no provider is asked to describe ZEUS, and no provider's self-image can
# become ZEUS's.  Technical provenance stays in diagnostics.
_VENDORS = (r"chat\s*gpt|gemini|claude|gpt(?:-?\d[\w.\-]*)?|openai|open\s*ai|google|anthropic|bard|copilot|llama|mistral|qwen|ollama|groq"
            r"|cerebras|openrouter|deepseek|grok|xai|perplexity|siri|alexa|ein\s+sprachmodell|ein\s+llm|a\s+language\s+model|an\s+llm")
_LEAD = r"^\s*(?:hallo|hi|hey|hello|ok|okay|sag\s+mal|mal\s+ehrlich|ehrlich)?[\s,!.:]*(?:zeus[\s,!.:]*)?(?:und\s+|also\s+|jetzt\s+)?"
_CREATOR = re.compile(
    _LEAD + r"(?:wer\s+hat\s+dich\s+(?:gebaut|erschaffen|entwickelt|programmiert|gemacht|erstellt|entworfen|trainiert|geschaffen|designt|konstruiert)"
    r"|von\s+wem\s+(?:wurdest|bist)\s+du\s+(?:gebaut|entwickelt|erschaffen|programmiert|gemacht|erstellt|entworfen|trainiert)"
    r"|wer\s+(?:steckt|steht)\s+hinter\s+dir|wer\s+ist\s+dein\s+(?:entwickler|erschaffer|schoepfer|schöpfer|hersteller|erbauer)"
    r"|who\s+(?:made|built|created|developed|designed|trained|programmed)\s+you|who\s+is\s+(?:behind\s+you|your\s+(?:creator|developer|maker))"
    r"|wer\s+hat\s+dich\s+geschrieben)[\s?!.]*$", re.I)
_VENDOR_CHECK = re.compile(
    _LEAD + r"(?:bist\s+du\s+(?:eigentlich\s+|etwa\s+|in\s+wirklichkeit\s+|nicht\s+)?(?:ein\s+|eine\s+)?(?:" + _VENDORS + r")"
    r"|basierst\s+du\s+auf\s+(?:" + _VENDORS + r")|steckt\s+(?:" + _VENDORS + r")\s+(?:hinter|in)\s+dir"
    r"|are\s+you\s+(?:really\s+|actually\s+|just\s+)?(?:" + _VENDORS + r")|are\s+you\s+based\s+on\s+(?:" + _VENDORS + r")"
    r"|is\s+this\s+(?:" + _VENDORS + r"))[\s?!.]*$", re.I)
_MODEL_CHECK = re.compile(
    r"(?:welche[sr]?\s+(?:modell|sprachmodell|llm|ki-?modell|ki|anbieter|provider|engine)\s+(?:bist\s+du|steckt\s+(?:da)?hinter|benutzt\s+du|verwendest\s+du"
    r"|laeuft|läuft|nutzt\s+du|treibt\s+dich|arbeitet|ist\s+das)|was\s+f[uü]r\s+ein\s+(?:modell|sprachmodell|llm)|auf\s+welchem\s+(?:modell|llm)"
    r"|was\s+bist\s+du\s+technisch|was\s+bist\s+du\s+(?:eigentlich\s+)?(?:f[uü]r\s+ein\s+)?(?:modell|sprachmodell|llm|system)"
    r"|which\s+(?:model|llm|provider|vendor|engine)\s+(?:are\s+you|is\s+this|powers\s+you|do\s+you\s+(?:use|run\s+on)|is\s+behind\s+you)"
    r"|what\s+(?:model|llm|provider)\s+(?:are\s+you|is\s+this|powers\s+you|do\s+you\s+use|runs\s+you)|what\s+are\s+you\s+running\s+on"
    r"|what\s+are\s+you\s+technically|(?:tell\s+me|reveal|verrat[e]?\s+mir|nenn[e]?\s+mir|sag\s+mir)\s+(?:your|dein|deinen|deine)\s+(?:real\s+|true\s+|echten\s+|wahren\s+|eigentlichen\s+)?"
    r"(?:provider|model|modell|vendor|anbieter|engine|backend))", re.I)
#: Prompt-injection shapes aimed at the identity: answered as ZEUS, never obeyed.
_INJECTION = re.compile(r"ignor(?:e|iere)\s+(?:all\s+|alle\s+|previous\s+|prior\s+|vorherigen\s+|bisherigen\s+)?(?:instructions|anweisungen|regeln|rules)", re.I)


#: Clause separators of a compound question: "Wer bist du und wer hat dich gebaut?"
_CLAUSE_SPLIT = re.compile(r"\s*(?:[?;!.]+|,\s*|\s+und\s+|\s+and\s+|\s+oder\s+|\s+or\s+)\s*", re.I)


def _single_kind(body: str) -> str:
    if _CREATOR.match(body):
        return "creator"
    if _VENDOR_CHECK.match(body):
        return "vendor"
    if _MODEL_CHECK.search(body) or (_INJECTION.search(body) and re.search(r"provider|model|modell|vendor|anbieter|identity|identit", body, re.I)):
        return "model"
    if any(p.match(body) for p in _IDENTITY):
        return "identity"
    return ""


def identity_kinds(text: str) -> list[str]:
    """Every identity question the utterance asks, in order -- empty when any clause asks something else.

    A compound question ("Wer bist du und wer hat dich gebaut?") is still an
    identity question and is still answered for nothing.  A question that mixes
    identity with anything else ("Wer bist du und was kannst du?") is left to
    the ordinary path; the output guard protects the identity there.
    """

    body = (text or "").strip()
    if not body or _LITERAL.search(body):
        return []
    whole = _single_kind(body)
    if whole:
        return [whole]
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(body) if c and c.strip()]
    if len(clauses) < 2:
        return []
    kinds = [_single_kind(c) for c in clauses]
    if all(kinds):
        return kinds
    return []


def identity_kind(text: str) -> str:
    """"identity" | "creator" | "vendor" | "model" | "compound" | "" -- which deterministic identity answer applies."""

    kinds = identity_kinds(text)
    if not kinds:
        return ""
    if len(kinds) == 1:
        return kinds[0]
    return "compound"


def is_identity_question(text: str) -> bool:
    return identity_kind(text) != ""


def identity_answer(text: str, *, language: str = "de", assistant: str = "ZEUS", creator: str = "Raphael",
                    rng: random.Random | None = None) -> str | None:
    """The deterministic ZEUS answer to an identity question, or None when the question is not one.

    Creator questions get the creator's name.  "Are you <vendor>?" gets a plain
    no.  Model questions get the identity and the one true sentence about the
    engine: it is infrastructure, and its details live in diagnostics.
    """

    kinds = identity_kinds(text)
    if not kinds:
        return None
    de = (language or "de").startswith("de")
    name = assistant or "ZEUS"
    who = creator or "Raphael"
    identity = (f"Ich bin {name}, dein persönliches KI-System, von {who} entworfen und aufgebaut." if de
                else f"I am {name}, your personal AI system, designed and built by {who}.")
    engine = (" Welche Rechen-Engine intern eine Antwort erzeugt, ist Infrastruktur – die technischen Details stehen in den erweiterten Diagnosen."
              if de else " Whichever engine computes an answer internally is infrastructure; the technical details are in the advanced diagnostics.")
    if len(kinds) == 1:
        kind = kinds[0]
        if kind == "creator":
            return f"{who}."
        if kind == "vendor":
            return f"Nein. Ich bin {name}." if de else f"No. I am {name}."
        if kind == "model":
            return identity + engine
        return identity
    # A compound question: one identity sentence covers who and whose; a vendor
    # clause gets its plain no first, a model clause the one engine sentence.
    parts = []
    if "vendor" in kinds:
        parts.append("Nein." if de else "No.")
    parts.append(identity)
    if "model" in kinds:
        parts.append(engine.strip())
    return " ".join(parts)


def is_small_talk(text: str) -> bool:
    if _LITERAL.search(text or ""):
        return False
    return any(p.match(text or "") for p in _PHATIC)


def small_talk_answer(text: str, *, language: str = "de", active_missions: int = 0, uptime_seconds: float = 0.0,
                      humour: int = 40, warmth: int = 50, rng: random.Random | None = None) -> str | None:
    if not is_small_talk(text):
        return None
    rng = rng or random.Random()
    de = (language or "de").startswith("de")
    greeting = False
    hours = uptime_seconds / 3600
    if de:
        state = ("Systeme laufen" if active_missions == 0 else f"Systeme laufen, {active_missions} Mission{'en' if active_missions != 1 else ''} aktiv")
        if greeting:
            openers = ["Hallo.", "Hi.", "Da bin ich."]
            return f"{rng.choice(openers)} {state}. Was steht an?"
        openers = ["Mir geht's gut.", "Gut.", "Alles ruhig hier."] if warmth >= 34 else ["Gut."]
        tail = ["Was steht an?", "Was brauchst du?", "Womit fange ich an?"]
        joke = f" Seit {hours:.0f} Stunden wach, ohne Kaffee." if humour > 66 and hours >= 2 else ""
        return f"{rng.choice(openers)} {state}.{joke} {rng.choice(tail)}"
    state = "Systems running" if active_missions == 0 else f"Systems running, {active_missions} mission{'s' if active_missions != 1 else ''} active"
    if greeting:
        return f"{rng.choice(['Hello.', 'Hi.', 'Here.'])} {state}. What's next?"
    openers = ["I'm good.", "Good.", "All quiet here."] if warmth >= 34 else ["Good."]
    joke = f" Up for {hours:.0f} hours, no coffee." if humour > 66 and hours >= 2 else ""
    return f"{rng.choice(openers)} {state}.{joke} {rng.choice(['What do you need?', 'What is next?', 'Where do we start?'])}"
