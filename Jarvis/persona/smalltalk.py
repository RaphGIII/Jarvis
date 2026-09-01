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
#: "was bist du technisch", "welches Modell" -- a technical question, answered truthfully by the model.
_TECHNICAL = re.compile(r"(technisch|technically|modell|model|backend|llm|sprachmodell|language\s+model|ki\b|ai\b|programm|software|hardware)", re.I)


def is_identity_question(text: str) -> bool:
    if _TECHNICAL.search(text or "") or _LITERAL.search(text or ""):
        return False
    return any(p.match(text or "") for p in _IDENTITY)


def identity_answer(text: str, *, language: str = "de", assistant: str = "Zeus", rng: random.Random | None = None) -> str | None:
    if not is_identity_question(text):
        return None
    rng = rng or random.Random()
    if (language or "de").startswith("de"):
        return rng.choice([
            f"Ich bin {assistant} – dein persönlicher Assistent. Ich helfe dir bei deinen Projekten, deinem Wissen und allem, was wir gemeinsam aufbauen.",
            f"{assistant}. Dein persönlicher Assistent – für deine Projekte, dein Wissen und alles, was wir zusammen aufbauen.",
        ])
    return rng.choice([
        f"I am {assistant} – your personal assistant. I help you with your projects, your knowledge and everything we build together.",
        f"{assistant}. Your personal assistant – for your projects, your knowledge and everything we build together.",
    ])


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
