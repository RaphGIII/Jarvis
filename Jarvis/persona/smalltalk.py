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
    re.compile(r"^\s*(?:hallo|hi|hey|servus|moin|guten (?:morgen|tag|abend)|hello)\s*(?:zeus)?[\s!.]*$", re.I),
)
#: Literal questions about feelings/consciousness are not small talk.
_LITERAL = re.compile(r"(wirklich|echt|tatsächlich|literally|really|actually).{0,30}(gefühl|emotion|bewusst|conscious|feel)", re.I)


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
    greeting = bool(_PHATIC[2].match(text or "")) and not (_PHATIC[0].match(text or "") or _PHATIC[1].match(text or ""))
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
