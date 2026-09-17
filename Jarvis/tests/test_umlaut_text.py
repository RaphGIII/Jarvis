"""ZEUS speaks German with real umlauts.

Everything ZEUS says or shows to the owner is German, and German is written
with ä ö ü ß. A transliteration such as "Es laeuft" in a reply is a defect
the owner sees (and the TTS voice mispronounces).

The static scan walks the backend packages with ``ast`` and inspects string
literals only (docstrings, comments and developer log calls are not owner
text). The detector is a curated list of transliterated stems, not a naive
ae/oe/ue pattern, so real words such as "Israel", "neue", "Feuer", "Poesie",
"aktuell", "Queue", "true" or "value" never trip it.

Legitimate transliterations remain in input matchers: speech-to-text and old
keyboards may still produce "laeuft", so a regex or keyword list may keep the
ASCII form as long as it ALSO accepts the umlaut form.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = (
    "service", "persona", "owner", "gateway", "capabilities", "jarvis",
    "speech", "voice", "knowledge", "projects", "tools", "skills",
)

# Transliterated stems (lower case). Each one is a German word fragment that
# only exists because an umlaut or ß was spelled out; none of them occurs in
# ordinary German or English words written correctly.
STEMS = (
    # ä
    "laeuf", "aeuss", "haeufig", "raeum", "spaet", "waehl", "naechst", "naeh",
    "erklaer", "aender", "staerk", "faehig", "waehrend", "taeglich", "geraet",
    "zusaetz", "aehnlich", "bestaetig", "maerz", "haett", "laeng", "haeng",
    "jaehr", "waere", "faellt", "gefaell", "zaehl", "erzaehl", "haelt",
    "enthaelt", "klaer", "saetz", "koerper", "abhaeng", "unabhaeng", "gaeng",
    "vorschlaeg", "schlaeg", "traeg", "auftraeg", "naemlich", "spaeter",
    "aelter", "aeltest", "uebertraeg", "waerm", "kaelt", "staend", "laest",
    "waeg", "itaet", "ergaenz", "laeuter", "flaeche", "praez", "schwaech",
    "erklaer", "gefaehr", "saeub", "maechtig", "taet",
    # ö
    "koenn", "moecht", "moeglich", "oeffn", "oeffentl", "hoer", "loesch",
    "loes", "groess", "hoech", "noetig", "benoetig", "erhoeh", "woech",
    "schoen", "gehoer", "zerstoer", "stoer", "boes", "toen", "froehlich",
    "persoenlich", "voellig", "moechte", "koennt", "gewoehn", "ueberpruef",
    "zoesisch", "schoepf", "hoehe", "koepf",
    # ü
    "fuer", "ueb", "rueck", "stueck", "kuenst", "fuehr", "natuerl", "verfueg",
    "schluess", "muess", "wuerd", "duerf", "gueltig", "fuell", "kuerz",
    "frueh", "muede", "buero", "pruef", "gruen", "gruess", "wuensch",
    "guenstig", "fuenf", "spuer", "fuehl", "bruecke", "luecke", "glueck",
    "drueck", "wuerf", "genueg", "suess", "zurueck", "tuer", "ausfuehr",
    "zufaell", "fluess", "schuetz", "nuetz", "unterstuetz", "stuetz",
    "huebsch", "betrueb", "kuend", "zuend", "hinzufueg", "einfueg", "fueg",
    "kuenft", "abstuerz", "stuerz", "begruend", "gruend", "ueblich",
    # ß
    "heiss", "weisst", "schliess", "gemaess", "strasse", "ausserdem",
    "ausserhalb", "draussen", "groesst", "massnahm", "spass", "gruesse",
)

# Stems that are fine inside these longer, correct words.
_FALSE_FRIENDS = (
    "tuerk",       # not a transliteration problem on its own in names
    "abenteuer", "steuer", "feuer", "teuer", "ungeheuer", "neuer",
    "treuer", "bauer", "dauer", "mauer", "trauer", "lauer", "sauer",
    "blueprint", "fueled", "fuel", "cruel", "influenc", "fluent",
    "hoeing", "shoe", "phoen", "poesie", "poet",
    "queue", "sequence", "frequen", "duett", "bluetooth",
    "ueberlingen",
)

_WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")
_UMLAUT = re.compile(r"[äöüÄÖÜß]")


def find_transliterations(text: str) -> list[str]:
    """The words of ``text`` that spell an umlaut or ß out as ae/oe/ue/ss."""

    hits: list[str] = []
    for match in _WORD.finditer(text):
        word = match.group(0).lower()
        if any(friend in word for friend in _FALSE_FRIENDS):
            continue
        if any(stem in word for stem in STEMS):
            hits.append(match.group(0))
    return hits


_LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
_LOG_OWNERS = re.compile(r"^_?(log|logger|LOG|LOGGER|logging)$")


def _is_log_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in _LOG_METHODS:
        owner = func.value
        if isinstance(owner, ast.Name) and _LOG_OWNERS.match(owner.id):
            return True
        if isinstance(owner, ast.Attribute) and _LOG_OWNERS.match(owner.attr):
            return True
    return False


def iter_literals(path: Path):
    """Yield ``(lineno, text, siblings)`` for every non-docstring literal.

    ``siblings`` holds the other string literals of the same list, tuple or
    set, so a keyword list that carries both spellings is recognised.
    """

    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    skip: set[int] = set()
    siblings: dict[int, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                skip.add(id(body[0].value))
        if _is_log_call(node):
            for inner in ast.walk(node):
                skip.add(id(inner))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            elements = [*node.keys, *node.values] if isinstance(node, ast.Dict) else node.elts
            texts = [e.value for e in elements if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            for element in elements:
                if element is not None:
                    siblings[id(element)] = texts
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            yield node.lineno, node.value, siblings.get(id(node), [])


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:/\-]+$")


def _umlaut_form(stem: str) -> str:
    return stem.replace("ae", "ä").replace("oe", "ö").replace("ue", "ü")


def accepted(text: str, siblings: list[str]) -> bool:
    """True for the legitimate uses of a transliteration in a literal."""

    # Identifiers, keys, file names, protocol values: never shown as prose.
    if _IDENTIFIER.match(text) and re.search(r"[_./:]", text):
        return True
    # A regex that also accepts the real umlaut, e.g. ``l(?:ä|ae)uft``.
    if _UMLAUT.search(text) and re.search(r"[|\[]", text):
        return True
    # A keyword list that carries the umlaut spelling next to the ASCII one.
    for word in find_transliterations(text):
        lowered = word.lower()
        for stem in STEMS:
            if stem in lowered:
                umlaut = _umlaut_form(stem)
                if umlaut != stem and any(umlaut in s.lower() for s in siblings):
                    break
                if stem.endswith("ss") or "ss" in stem:
                    sz = umlaut.replace("ss", "ß")
                    if any(sz in s.lower() for s in siblings):
                        break
        else:
            return False
    return True


#: Modules that fold the owner's words (ä -> ae, ö -> oe, ü -> ue, ß -> ss)
#: BEFORE matching them. Their matchers are written in the folded alphabet on
#: purpose: an umlaut can never reach them, so "laeuft" in such a regex or
#: keyword list already matches "läuft". Only matcher-shaped literals (regex
#: syntax, or bare lower-case keywords) are exempt there; a sentence with a
#: capital letter or punctuation is still owner text and still checked.
#: ``test_folded_matchers_accept_both_spellings`` proves the fold is real.
FOLDED_MATCHER_MODULES = {
    "service/routing.py": "read()/asks_about_itself() match on fold(text)",
    "service/intents.py": "parse() matches on fold(body)",
    "service/music.py": "parse() matches on _fold(text); _STOPWORDS compared to _fold(word)",
    "service/claims.py": "find_claims() matches on _fold(text)",
    "service/composer.py": "GOAL_FAMILIES markers compared to _fold(goal/step)",
    "service/calendar.py": "relative-day key is built after replacing ü with ue",
    "gateway/intelligence_class.py": "classify() matches on _fold(text)",
    "capabilities/models.py": "goal parsing matches on _fold(raw)",
    "capabilities/resolver.py": "synonyms keyed by _fold()ed terms",
    "capabilities/registry.py": "BOILERPLATE subtracted from _fold()ed terms",
    "capabilities/service.py": "suggest_id() stopwords compared to _fold(goal) words",
}

_REGEX_SHAPE = re.compile(r"\\[bwsd]|\(\?|[|^$]")
_KEYWORD_SHAPE = re.compile(r"^[a-z0-9 \t\n\-]+$")


def _folded_matcher(rel: str, text: str) -> bool:
    return rel in FOLDED_MATCHER_MODULES and bool(_REGEX_SHAPE.search(text) or _KEYWORD_SHAPE.match(text))


# Explicit, reviewed exceptions: (relative path, literal substring, reason).
ALLOWLIST: tuple[tuple[str, str, str], ...] = (
    ("knowledge/graph.py", "der die das und oder aber wenn dann sonst fuer",
     "lexical embedder stopwords; its tokenizer is [a-z0-9_]+, changing it "
     "invalidates stored vectors (version lexical-v2-256)"),
    ("jarvis/measure_pipeline.py", "Rekursion",
     "developer latency benchmark prompts, never shown to the owner"),
)


def _allowlisted(rel: str, text: str) -> bool:
    return _folded_matcher(rel, text) or any(rel == path and needle in text for path, needle, _ in ALLOWLIST)


def scan() -> list[str]:
    problems: list[str] = []
    for package in PACKAGES:
        base = ROOT / package
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            for lineno, text, sibs in iter_literals(path):
                hits = find_transliterations(text)
                if not hits or accepted(text, sibs) or _allowlisted(rel, text):
                    continue
                problems.append(f"{rel}:{lineno}: {hits} in {text[:120]!r}")
    return problems


# ---------------------------------------------------------------- detector


@pytest.mark.parametrize("word", [
    "Israel", "neue", "Feuer", "Poesie", "aktuell", "Frequenz", "Queue", "true",
    "blue", "value", "Sequenz", "Duett", "Kontinuum", "Michael", "Steuer",
    "Abenteuer", "muss", "dass", "Schluss", "Quelle", "Bauer", "does", "goes",
    "aerial", "Raphael", "maestro", "Mauer", "continue", "issue", "läuft", "Größe",
])
def test_detector_leaves_real_words_alone(word):
    assert find_transliterations(word) == []


@pytest.mark.parametrize("word", [
    "laeuft", "fuer", "ueber", "koennen", "moechte", "waehle", "spaeter",
    "zurueck", "oeffne", "naechste", "Stueck", "Kuenstler", "Lautstaerke",
    "pruefe", "geprueft", "Pruefung", "erklaere", "fuehre", "natuerlich",
    "hoere", "verfuegbar", "Groesse", "schliesse", "heisst", "muessen",
    "wuerde", "haette", "Aenderung", "Uebersicht", "Loeschen", "Schluessel",
])
def test_detector_catches_transliterations(word):
    assert find_transliterations(word) == [word]


def test_input_matchers_that_accept_both_spellings_are_allowed():
    assert accepted(r"\bwas\s+l(?:ä|ae)uft\b", [])
    assert accepted("laeuft", ["läuft", "laeuft"])
    assert accepted("music.laeuft_gerade", [])
    assert not accepted("Es laeuft: Song", [])


# ---------------------------------------------------------------- the scan


def test_backend_string_literals_use_real_umlauts():
    problems = scan()
    assert not problems, "German transliterations in owner-facing text:\n" + "\n".join(problems)


# ---------------------------------------------------------------- behaviour


def _music_outcome(tmp_path, action: str, **kwargs):
    from test_music import PLAYING, WRONG_TRACK, FakeExecution, FakeSession, build
    from service.music import MusicRequest

    state = WRONG_TRACK if kwargs.pop("wrong", False) else PLAYING
    service = build(tmp_path, session=FakeSession(state),
                    execution=FakeExecution(ok=True, output={"ok": True}))
    return service.run(MusicRequest(action, **kwargs))


def test_now_playing_reply_says_laeuft_with_umlaut(tmp_path):
    from service.music import compose

    outcome = _music_outcome(tmp_path, "current")
    reply = compose(outcome, language="de")

    assert outcome.receipt.verified is True
    assert reply.startswith("Es läuft: Lose Yourself - Eminem")
    assert "laeuft" not in reply.lower()
    assert "Belege:" in reply


def test_play_reply_says_laeuft_jetzt_with_umlaut(tmp_path):
    from service.music import compose

    reply = compose(_music_outcome(tmp_path, "play", query="Lose Yourself Eminem"), language="de")

    assert reply.startswith("Läuft jetzt: Lose Yourself - Eminem")


def test_music_checks_are_shown_in_german_and_keep_a_stable_key(tmp_path):
    from service.music import compose

    outcome = _music_outcome(tmp_path, "play", query="Lose Yourself Eminem", wrong=True)
    checks = outcome.receipt.verifications
    reply = compose(outcome, language="de")

    assert {c.key for c in checks} >= {"media_session_exists", "player_is_spotify",
                                       "playback_running", "requested_track_playing"}
    shown = {c.key: c.check for c in checks}
    assert shown["media_session_exists"] == "Eine Mediensitzung ist vorhanden"
    assert shown["playback_running"] == "Windows meldet laufende Wiedergabe"
    assert shown["requested_track_playing"] == "Der angefragte Titel läuft"
    for check in checks:
        assert not re.search(r"\b(the|is playing|reports|exists)\b", check.check), check.check
        assert find_transliterations(check.check + " " + check.observed) == []
    # the UI reads the serialised receipt: German text, machine key alongside
    dumped = outcome.receipt.to_dict()["verifications"]
    assert {"check": "Der angefragte Titel läuft", "key": "requested_track_playing"}.items() <= next(
        item for item in dumped if item.get("key") == "requested_track_playing").items()
    assert "Der angefragte Titel läuft: fehlgeschlagen" in reply
    head = reply.splitlines()[0]
    assert head.startswith("Das hat nicht geklappt"), head
    assert not re.search(r"\b(did not|take effect|reports|playing)\b", head), "a German reply never carries the machine's English headline"


def test_verification_key_survives_the_ledger_round_trip():
    from runtime.receipts import Receipt, Verification

    receipt = Receipt(kind="music.play", executor="x", ok=True,
                      verifications=(Verification("Der Titel hat gewechselt", True, observed="a -> b",
                                                  key="track_changed"),))
    again = Receipt.from_dict(receipt.to_dict())

    assert again.verifications[0].key == "track_changed"
    assert again.verifications[0].check == "Der Titel hat gewechselt"


@pytest.mark.parametrize("text", ["Was läuft gerade?", "Was laeuft gerade?"])
def test_asking_what_is_playing_accepts_both_spellings(text):
    from service.intent import Intent, classify
    from service.music import understand

    assert understand(text).action == "current"
    assert classify(text).intent is Intent.MUSIC


def test_folded_matchers_accept_both_spellings():
    """The exemption in FOLDED_MATCHER_MODULES is only honest if the fold is real."""

    from capabilities.models import _fold as models_fold
    from capabilities.resolver import _fold as resolver_fold
    from gateway.intelligence_class import _fold as class_fold
    from service.claims import _fold as claims_fold
    from service.intents import fold as intents_fold
    from service.music import _fold as music_fold
    from service.routing import fold as routing_fold

    for fold in (models_fold, resolver_fold, class_fold, claims_fold, intents_fold, music_fold, routing_fold):
        assert fold("Läuft Größe Übersicht öffne heißt") == "laeuft groesse uebersicht oeffne heisst"
        assert fold("laeuft groesse") == "laeuft groesse"
