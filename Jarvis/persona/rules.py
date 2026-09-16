"""Owner personality rules: the structured, editable layer beside the protected core identity.

The core identity (name ZEUS, creator Raphael, character) is deterministic and protected.  What the
owner adds on top -- how ZEUS relates to him, how it speaks, how it behaves, what it prefers, where
its boundaries lie, what applies in which situation -- is a list of rules with a category, a
normalised meaning and an owner-facing text.  The list is compiled into the provider-independent
PersonalityContract, so the same rules reach Gemini, OpenAI, Anthropic, Groq, Cerebras or OpenRouter.

A rule is a small dict:
  id, category, meaning, text, source_text, enabled, protected, created_at, updated_at, source, version, order

Categories (owner-facing German labels in CATEGORY_LABELS):
  identity, relationship, communication_style, behaviour, response_preference, domain_preference,
  boundary, situational

Rules that mean the same thing merge instead of piling up: the meaning is a normalised token set, and
two rules of one category whose meanings overlap strongly (Jaccard >= 0.6) are one rule -- the newer
text wins, the version counts up.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

CATEGORIES = ("identity", "relationship", "communication_style", "behaviour", "response_preference",
              "domain_preference", "boundary", "situational")

CATEGORY_LABELS = {
    "identity": "Identität",
    "relationship": "Beziehung zu Raphael",
    "communication_style": "Kommunikationsstil",
    "behaviour": "Verhalten",
    "response_preference": "Antwortpräferenzen",
    "domain_preference": "Fachliche Präferenzen",
    "boundary": "Grenzen",
    "situational": "Situative Regeln",
}

_SAVE_PREFIX = re.compile(
    r"^(?:(?:bitte\s+)?(?:merk(?:e)?\s+dir|speicher(?:e)?|notier(?:e)?|[üu]berschreib(?:e)?|behalte?|remember|save|note)[,:]?\s*(?:dass\s+|that\s+)?)+",
    re.I,
)

_CATEGORY_HINTS = [
    ("boundary", re.compile(r"\b(nie|niemals|nicht\s+ohne|auf\s+keinen\s+fall|verboten|grenze|tabu|never|do\s+not|don't|no\s+\w+\s+without)\b", re.I)),
    ("situational", re.compile(r"\b(wenn\s+ich|sobald|falls|während|beim|abends|morgens|nachts|im\s+(?:studium|unterricht|krankenhaus|dienst)|when\s+i|while\s+i|during)\b", re.I)),
    ("relationship", re.compile(r"\b(freund(?:in)?|begleiter(?:in)?|partner(?:in)?|vertraute[rn]?|berater(?:in)?|coach|mentor(?:in)?|lehrer(?:in)?|kumpel|für\s+mich\s+da"
                                r"|h[öo]rst?\s+(?:mir\s+)?zu|zuh[öo]r\w*|gibst?\s+(?:mir\s+)?rat|r[äa]tst|unterst[üu]tzt?\s+mich|companion|friend|listen\s+to\s+me|advice)\b", re.I)),
    ("domain_preference", re.compile(r"\b(medizin\w*|anatomie|physiologie|biochemie|klinik|pharma|studium|klausur|programmier\w*|code|python|schach|musik|physik|mathe\w*|fach\w*|bevorzug\w*|lieber\s+\w+\s+als)\b", re.I)),
    ("communication_style", re.compile(r"\b(ton|tonfall|sprich|sprichst|redest|duz|siez|f[öo]rmlich|locker|humor|witz|ironie|sarkas|kurz\s+und|knapp|ausf[üu]hrlich|warm|sachlich|direkt|h[öo]flich|nüchtern|style|tone|formal|casual)\b", re.I)),
    ("response_preference", re.compile(r"\b(antworte\s+immer|antworten\s+sollen|liste[n]?|stichpunkte|aufz[äa]hlung|beispiel[e]?|format|l[äa]nge|s[äa]tze|absatz|abs[äa]tze|tabelle|quelle[n]?|zitier|structure|bullet|examples?)\b", re.I)),
    ("behaviour", re.compile(r"\b(verhalte?\s+dich|verhalten|proaktiv|frag\w*\s+nach|erinner\w*\s+mich|melde\s+dich|schlag\w*\s+vor|warte\s+auf|handle|behave|remind\s+me|ask\s+me)\b", re.I)),
    ("identity", re.compile(r"\b(du\s+bist|dein\s+name|wer\s+du\s+bist|hei[ßs]t|nenn\w*\s+dich|you\s+are|your\s+name)\b", re.I)),
]

_STOP = {"dass", "und", "auch", "der", "die", "das", "ein", "eine", "einen", "einem", "einer", "mir", "mich", "mein", "meine", "meinen", "meinem",
         "meiner", "du", "dir", "dich", "dein", "deine", "deinen", "deinem", "deiner", "ich", "bist", "ist", "sind", "bin", "wird", "wirst", "zu", "für",
         "mit", "auf", "in", "im", "an", "am", "von", "vom", "the", "and", "also", "you", "me", "my", "your", "are", "is", "a", "an", "to", "of", "for",
         "bitte", "immer", "auch", "noch", "so", "es", "sich", "zeus", "raphael", "raphaels"}


def strip_save_prefix(text: str) -> str:
    body = " ".join(str(text or "").split())
    return _SAVE_PREFIX.sub("", body).strip(" .:,")


def classify_category(text: str) -> str:
    """The first matching category in priority order (boundaries and situations before the rest)."""

    body = strip_save_prefix(text)
    for category, pattern in _CATEGORY_HINTS:
        if pattern.search(body):
            return category
    return "behaviour"


def normalize_meaning(text: str) -> str:
    """A stable, order-independent token form of what the rule says."""

    body = strip_save_prefix(text).lower()
    body = re.sub(r"[^\wäöüß ]+", " ", body)
    tokens = [t for t in body.split() if len(t) > 2 and t not in _STOP]
    stems = sorted({re.sub(r"(en|er|es|em|st|e|s|n)$", "", t) if len(t) > 5 else t for t in tokens})
    return " ".join(stems)


_IMPERATIVES = {
    "sprich": "spricht", "rede": "redet", "antworte": "antwortet", "beantworte": "beantwortet", "sag": "sagt", "sage": "sagt", "erklär": "erklärt",
    "erkläre": "erklärt", "halte": "hält", "halt": "hält", "verhalte": "verhält", "verhalt": "verhält", "sei": "ist", "bleib": "bleibt", "bleibe": "bleibt",
    "erinnere": "erinnert", "erinner": "erinnert", "frag": "fragt", "frage": "fragt", "melde": "meldet", "meld": "meldet", "schlag": "schlägt",
    "schlage": "schlägt", "nutze": "nutzt", "nutz": "nutzt", "benutze": "benutzt", "verwende": "verwendet", "vermeide": "vermeidet", "achte": "achtet",
    "warte": "wartet", "formuliere": "formuliert", "schreib": "schreibt", "schreibe": "schreibt", "fasse": "fasst", "fass": "fasst", "denk": "denkt",
    "denke": "denkt", "mach": "macht", "mache": "macht", "zeig": "zeigt", "zeige": "zeigt", "prüfe": "prüft", "prüf": "prüft", "stell": "stellt",
    "stelle": "stellt", "gib": "gibt", "hör": "hört", "höre": "hört", "nimm": "nimmt", "lass": "lässt", "lasse": "lässt", "beginne": "beginnt",
    "beginn": "beginnt", "fang": "fängt", "fange": "fängt", "such": "sucht", "suche": "sucht", "liefere": "liefert", "liefer": "liefert",
    "begründe": "begründet", "nenne": "nennt", "nenn": "nennt", "unterbrich": "unterbricht", "weise": "weist", "weis": "weist", "bitte": "",
}


def _imperatives_to_third_person(text: str, assistant: str) -> str:
    """"Sprich mit mir locker" -> "ZEUS spricht mit mir locker"; "…, halte dich kurz" -> "…, hält ZEUS sich kurz"."""

    def sub(match: re.Match[str]) -> str:
        lead, verb, reflexive = match.group(1), match.group(2), match.group(3)
        third = _IMPERATIVES.get(verb.lower())
        if not third:
            return match.group(0)
        obj = " sich" if reflexive else ""
        return f"{lead}{assistant} {third}{obj}" if not lead else f"{lead}{third} {assistant}{obj}"

    text = re.sub(r"^bitte[,\s]+", "", text, flags=re.I)
    return re.sub(r"(^|,\s*)([A-Za-zÄÖÜäöüß]+)(\s+dich\b)?", sub, text, count=2)


_SECOND_PERSON = {"bist": "ist", "hast": "hat", "wirst": "wird", "kannst": "kann", "sollst": "soll", "darfst": "darf", "musst": "muss",
                  "willst": "will", "gibst": "gibt", "nimmst": "nimmt", "sprichst": "spricht", "hältst": "hält", "weißt": "weiß", "hilfst": "hilft",
                  "siehst": "sieht", "liest": "liest", "isst": "isst", "vergisst": "vergisst", "lässt": "lässt", "rätst": "rät", "magst": "mag"}
#: Separable verbs whose prefix goes to the end of a main clause: "mir zuhörst" -> "hört Raphael zu".
_SEPARABLE = {"zu": {"hörst", "hoerst"}, "auf": {"passt", "munterst"}, "an": {"rufst", "spornst", "treibst"}, "bei": {"stehst", "bringst"},
              "mit": {"denkst", "fühlst", "hilfst"}, "vor": {"schlägst", "bereitest", "liest"}, "da": {"bist"}, "ein": {"greifst"},
              "nach": {"fragst", "hakst"}, "zurück": {"hältst"}}
_NOT_VERBS = {"fast", "sonst", "selbst", "erst", "zuerst", "meist", "fest", "lust", "angst", "kunst", "dienst", "herbst", "rest", "test", "jetzt",
              "zuletzt", "trost", "gast", "last", "post", "brust", "frust", "geist", "ernst", "arzt", "satz", "platz", "schutz", "netz", "gesetz"}


def _third_person_verb(word: str) -> tuple[str, str] | None:
    """(third-person verb, separated prefix) for a second-person verb at the end of a subordinate clause, else None."""

    low = word.lower()
    if low in _NOT_VERBS or len(low) < 4:
        return None
    for prefix, rests in _SEPARABLE.items():
        if low.startswith(prefix) and low[len(prefix):] in rests:
            third = _third_person_verb(low[len(prefix):])
            return (third[0], prefix) if third else None
    if low in _SECOND_PERSON:
        return _SECOND_PERSON[low], ""
    if low.endswith(("test", "dest")) and len(low) > 5:
        return low[:-3] + "et", ""
    if low.endswith(("sst", "ßt", "tzt", "zt")):
        return low, ""
    if low.endswith("st"):
        return low[:-2] + "t", ""
    return None


def _dass_clause_as_main_clause(body: str, *, assistant: str) -> str | None:
    """"du auch mein Freund bist und mir zuhörst und Rat gibst" -> "ZEUS ist auch mein Freund, hört mir zu und gibt Rat".

    The verbs of a "dass du ..." clause stand last in each coordinated part; a fact about ZEUS puts them
    second.  Parts end at a verb followed by "und", a comma or the end.  None when the shape does not hold.
    """

    words = body.rstrip(" .!?").split()
    if len(words) < 2 or words[0].lower() != "du":
        return None
    parts: list[tuple[str, str, list[str]]] = []
    current: list[str] = []
    for i, raw in enumerate(words[1:], start=1):
        token = raw.rstrip(",")
        comma = raw.endswith(",")
        nxt = words[i + 1].lower() if i + 1 < len(words) else ""
        if token.lower() == "und" and not current:
            continue
        verb = _third_person_verb(token)
        if verb and (comma or nxt in {"und", ""}):
            parts.append((verb[0], verb[1], current))
            current = []
            continue
        current.append(token)
    if current or not parts:
        return None
    phrases = [" ".join(x for x in [verb, *rest, prefix] if x) for verb, prefix, rest in parts]
    joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " und " + phrases[-1]
    return f"{assistant} {joined}"


def owner_text(text: str, *, assistant: str = "ZEUS", creator: str = "Raphael") -> str:
    """The rule in the owner's words, turned into the third person so it reads as a fact about ZEUS.

    "du auch mein Freund bist und mir zuhörst und Rat gibst" -> "ZEUS ist auch Raphaels Freund, hört Raphael zu und gibt Rat."
    """

    body = strip_save_prefix(text)
    if not body:
        return ""
    if re.search(r"\bdass\s+du\b", str(text or ""), re.I):
        body = _dass_clause_as_main_clause(body, assistant=assistant) or body
    out = re.sub(r"\bper\s+du\b", "per __perdu__", body, flags=re.I)
    out = _imperatives_to_third_person(out, assistant)
    swaps = [
        (r"\bdu\s+bist\b", f"{assistant} ist"), (r"\bbist\s+du\b", f"ist {assistant}"), (r"\bdu\s+wirst\b", f"{assistant} wird"),
        (r"\bdu\s+sollst\b", f"{assistant} soll"), (r"\bdu\s+kannst\b", f"{assistant} kann"), (r"\bdu\s+darfst\b", f"{assistant} darf"),
        (r"\bdu\s+musst\b", f"{assistant} muss"), (r"\bdu\b", assistant), (r"\bdich\b", assistant), (r"\bdir\b", assistant),
        (r"\bdeine[nmrs]?\b", f"{assistant}'"), (r"\bdein\b", f"{assistant}'"),
        (r"\bmeine[nmrs]?\b", f"{creator}s"), (r"\bmein\b", f"{creator}s"), (r"\bmir\b", creator), (r"\bmich\b", creator), (r"\bich\b", creator),
        (r"\bzuh[öo]rst\b", "zuhört"), (r"\bh[öo]rst\b", "hört"), (r"\bgibst\b", "gibt"), (r"\blerne\b", "lernt"), (r"\barbeite\b", "arbeitet"),
        (r"\bstudiere\b", "studiert"), (r"\bschlafe\b", "schläft"), (r"\bbin\b", "ist"), (r"\bhabe\b", "hat"), (r"\bkannst\b", "kann"), (r"\bsollst\b", "soll"), (r"\bdarfst\b", "darf"),
        (r"\bmusst\b", "muss"), (r"\bwirst\b", "wird"), (r"\bsprichst\b", "spricht"), (r"\bredest\b", "redet"), (r"\bantwortest\b", "antwortet"),
        (r"\bverh[äa]ltst\b", "verhält"), (r"\bfragst\b", "fragt"), (r"\berinnerst\b", "erinnert"), (r"\bunterst[üu]tzt\b", "unterstützt"),
        (r"\br[äa]tst\b", "rät"), (r"\bbist\b", "ist"), (r"\bhast\b", "hat"),
    ]
    for pattern, repl in swaps:
        out = re.sub(pattern, repl, out, flags=re.I)
    # "ZEUS auch Raphaels Freund ist" -> subject-verb order for the common "..., dass du X bist" shape
    out = re.sub(rf"^{re.escape(assistant)}\s+(.+?)\s+ist\b(?!\s+\w)", rf"{assistant} ist \1", out)
    out = re.sub(rf"^{re.escape(assistant)}\s+(.+?)\s+ist\s+und\b", rf"{assistant} ist \1 und", out)
    out = out.replace("__perdu__", "du")
    out = out[0].upper() + out[1:]
    if not out.endswith((".", "!", "?")):
        out += "."
    return out


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def make_rule(text: str, *, category: str = "", source: str = "OWNER_UI", protected: bool = False, assistant: str = "ZEUS",
              creator: str = "Raphael", source_text: str = "", enabled: bool = True) -> dict[str, Any]:
    body = " ".join(str(text or "").split())
    cat = category if category in CATEGORIES else classify_category(body)
    facing = owner_text(body, assistant=assistant, creator=creator) if source_text or _SAVE_PREFIX.match(body) or re.search(r"\b(du|dir|dich|dein\w*|mir|mich|mein\w*)\b", body, re.I) else (body[0].upper() + body[1:] if body else body)
    now = _now()
    return {"id": f"rule_{uuid.uuid4().hex[:8]}", "category": cat, "meaning": normalize_meaning(body), "text": facing[:400],
            "source_text": (source_text or body)[:400], "enabled": bool(enabled), "protected": bool(protected),
            "created_at": now, "updated_at": now, "source": source, "version": 1}


def normalize_rule(raw: Any, *, index: int = 0, assistant: str = "ZEUS", creator: str = "Raphael") -> dict[str, Any] | None:
    """A rule as the editor or an older document hands it in, completed to the full shape (fields kept when present)."""

    if isinstance(raw, str):
        raw = {"text": raw}
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("text") or "").strip()[:400]
    if not text:
        return None
    now = _now()
    category = str(raw.get("category") or "")
    rule = {
        "id": str(raw.get("id") or f"rule_{uuid.uuid4().hex[:8]}"),
        "category": category if category in CATEGORIES else classify_category(text),
        "meaning": str(raw.get("meaning") or normalize_meaning(text)),
        "text": text,
        "source_text": str(raw.get("source_text") or text)[:400],
        "enabled": bool(raw.get("enabled", True)),
        "protected": bool(raw.get("protected", False)),
        "created_at": str(raw.get("created_at") or now),
        "updated_at": str(raw.get("updated_at") or now),
        "source": str(raw.get("source") or "OWNER_UI"),
        "version": int(raw.get("version", 1) or 1),
        "order": index,
    }
    return rule


def similarity(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def merge_rule(rules: list[dict[str, Any]], new: dict[str, Any], *, threshold: float = 0.6) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Add ``new`` to ``rules`` -- or, when a rule of the same category means the same, update that one.

    Returns (rules, the stored rule, "added" | "updated" | "unchanged").
    """

    out = [dict(r) for r in rules]
    for rule in out:
        if rule.get("category") != new.get("category"):
            continue
        if rule.get("meaning") == new.get("meaning") or similarity(str(rule.get("meaning", "")), str(new.get("meaning", ""))) >= threshold:
            if rule.get("text") == new.get("text") and rule.get("enabled", True):
                return out, rule, "unchanged"
            rule.update({"text": new["text"], "source_text": new.get("source_text", rule.get("source_text", "")), "meaning": new["meaning"],
                         "enabled": True, "protected": bool(rule.get("protected") or new.get("protected")),
                         "updated_at": _now(), "source": new.get("source", rule.get("source", "")), "version": int(rule.get("version", 1) or 1) + 1})
            return out, rule, "updated"
    new = dict(new)
    new["order"] = len(out)
    out.append(new)
    return out, new, "added"


def grouped(rules: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Rules by category, in category order, enabled or not."""

    groups: dict[str, list[dict[str, Any]]] = {c: [] for c in CATEGORIES}
    for rule in sorted((r for r in rules if isinstance(r, dict)), key=lambda r: int(r.get("order", 0) or 0)):
        groups.setdefault(rule.get("category") if rule.get("category") in CATEGORIES else "behaviour", []).append(rule)
    return {c: rs for c, rs in groups.items() if rs}
