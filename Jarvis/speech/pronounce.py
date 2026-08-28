"""What ZEUS says versus what ZEUS shows: the pronunciation pipeline.

    answer text (displayed unchanged)
        -> language-aware normalisation   (numbers, units, URLs, acronyms)
        -> pronunciation lexicon          (owner entries outrank the defaults)
        -> provider rendering             (piper/espeak: a respelling the
                                           German phonemiser reads correctly)
        -> TTS provider

Only the *spoken* string changes; the transcript, the events and the memory
keep the proper spelling.  The lexicon is data: a built-in seed here, and the
owner's persistent file (``<state>/voice/lexicon.json``) on top, written by
the correction flow ("du sprichst X falsch aus, sag Y").

Why respelling rather than phonemes: the installed piper voices phonemise
with espeak-ng from *text*; ``PiperVoice.synthesize`` takes no SSML and no
inline phonemes.  A respelling that the German espeak rules read the way the
owner wants ("Git-Hab", "Spottifai") is the representation this provider
actually honours.  Another provider gets its own ``render`` -- the entry keeps
``spoken_as`` per provider family so switching voices does not lose the
owner's corrections.

Nothing here claims acoustic quality; the tests check the text that reaches
the provider.  Ears decide the rest.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: Provider families a spoken form can be stored for.
PROVIDERS = ("piper_espeak", "generic")

#: German letter names for spelled-out acronyms (espeak-de reads "GPU" as a word otherwise).
LETTERS_DE = {"a": "A", "b": "Be", "c": "Ze", "d": "De", "e": "E", "f": "Ef", "g": "Ge", "h": "Ha", "i": "I", "j": "Jot", "k": "Ka",
              "l": "El", "m": "Em", "n": "En", "o": "O", "p": "Pe", "q": "Ku", "r": "Er", "s": "Es", "t": "Te", "u": "U", "v": "Vau",
              "w": "We", "x": "Ix", "y": "Ypsilon", "z": "Zett"}
LETTERS_EN = {ch: ch.upper() for ch in "abcdefghijklmnopqrstuvwxyz"}

UNITS_DE = {"ms": "Millisekunden", "s": "Sekunden", "min": "Minuten", "h": "Stunden", "gb": "Gigabyte", "mb": "Megabyte", "kb": "Kilobyte",
            "tb": "Terabyte", "ghz": "Gigahertz", "mhz": "Megahertz", "khz": "Kilohertz", "hz": "Hertz", "fps": "Bilder pro Sekunde",
            "px": "Pixel", "kg": "Kilogramm", "km": "Kilometer", "cm": "Zentimeter", "mm": "Millimeter", "%": "Prozent", "°c": "Grad Celsius",
            "w": "Watt", "kw": "Kilowatt", "v": "Volt", "mib": "Mebibyte", "gib": "Gibibyte"}
UNITS_EN = {"ms": "milliseconds", "s": "seconds", "min": "minutes", "h": "hours", "gb": "gigabytes", "mb": "megabytes", "kb": "kilobytes",
            "tb": "terabytes", "ghz": "gigahertz", "mhz": "megahertz", "hz": "hertz", "fps": "frames per second", "px": "pixels",
            "%": "percent", "°c": "degrees Celsius", "mib": "mebibytes", "gib": "gibibytes"}

#: Acronyms that are read as words, not spelled (stay as they are).
WORD_ACRONYMS = {"nasa", "gpu", "cpu"}  # gpu/cpu are spelled by the seed lexicon below; kept here for completeness
WORD_ACRONYMS = {"nasa", "ram", "rom", "usb", "json", "yaml", "html", "sql"}


@dataclass
class LexiconEntry:
    surface: str
    language: str = "de"            # de | en | * (any)
    spoken_as: dict[str, str] = field(default_factory=dict)   # provider family -> rendering
    scope: str = "global"
    source: str = "seed"            # seed | owner
    note: str = ""
    updated_at: str = ""

    def rendering(self, provider: str) -> str | None:
        return self.spoken_as.get(provider) or self.spoken_as.get("generic")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LexiconEntry":
        return cls(surface=str(data.get("surface", "")), language=str(data.get("language", "de")),
                   spoken_as=dict(data.get("spoken_as") or {}), scope=str(data.get("scope", "global")),
                   source=str(data.get("source", "owner")), note=str(data.get("note", "")), updated_at=str(data.get("updated_at", "")))


def _seed() -> list[LexiconEntry]:
    de = lambda surface, spoken, note="": LexiconEntry(surface, "de", {"piper_espeak": spoken, "generic": spoken}, note=note)
    en = lambda surface, spoken, note="": LexiconEntry(surface, "en", {"piper_espeak": spoken, "generic": spoken}, note=note)
    return [
        de("Zeus", "Zeus", "the product and the name; espeak-de reads 'Zeus' as /tsɔʏs/ (matches ZEUS too: case-insensitive)"),
        de("Spotify", "Spottifai"), de("GitHub", "Git-Hab"), de("OpenAI", "Open-Ei-Ai"), de("GPU", "Ge-Pe-U"), de("CPU", "Ze-Pe-U"),
        de("Whisper", "Wisper"), de("Piper", "Peiper"), de("SelfDev", "Self-Deff"), de("Knowledge Graph", "Nolledsch Graf"),
        de("Knowledge", "Nolledsch"), de("Mission Control", "Mischen Kontroll"), de("Stockfish", "Stockfisch"),
        de("Ollama", "Ollama"), de("Qwen", "Kwenn"), de("Claude", "Klohd"), de("Python", "Peiten"), de("PowerShell", "Pauer-Schell"),
        de("Windows", "Windos"), de("Chrome", "Krohm"), de("Bluetooth", "Blutuhs"), de("Cloud", "Klaud"), de("Update", "Apdeit"),
        de("Feature", "Fietscher"), de("Release", "Rilies"), de("Supervisor", "Supervaiser"), de("Listener", "Lissner"),
        de("Router", "Ruter"), de("Timeout", "Teimaut"), de("Download", "Daunlohd"), de("Upload", "Aplohd"), de("Screenshot", "Skrienschott"),
        de("Voice Studio", "Vois Stjudio"), de("Expert", "Experte"), de("Gateway", "Gehtwee"), de("Capability", "Käpäbiliti"),
        de("Wakeword", "Weikwörd"), de("Wake-Word", "Weikwörd"),
        de("Physikum", "Physikum"), de("Biochemie", "Biochemie"), de("Gluconeogenese", "Glukoneogenese"),
        de("Desoxyribonukleinsäure", "Desoxy-ribo-nuklein-säure"), de("Mitochondrien", "Mitochondrien"), de("Carbonsäure", "Karbonsäure"),
        de("Physiologie", "Physiologie"),
        en("Zeus", "Zoos", "English speakers say /zuːs/"), en("Qwen", "Kwen"), en("Ollama", "Oh-lama"),
    ]


class Lexicon:
    """Seed entries plus the owner's file; owner entries win on the same surface+language."""

    def __init__(self, owner_path: str | Path | None = None) -> None:
        self.owner_path = Path(owner_path) if owner_path else None
        self.entries: dict[tuple[str, str], LexiconEntry] = {}
        for entry in _seed():
            self.entries[(entry.surface.lower(), entry.language)] = entry
        self._load_owner()

    def _load_owner(self) -> None:
        if self.owner_path is None or not self.owner_path.is_file():
            return
        try:
            data = json.loads(self.owner_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for row in data.get("entries", []) if isinstance(data, dict) else []:
            try:
                entry = LexiconEntry.from_dict(row)
            except Exception:  # noqa: BLE001
                continue
            if entry.surface:
                entry.source = "owner"
                self.entries[(entry.surface.lower(), entry.language)] = entry

    def save(self) -> None:
        if self.owner_path is None:
            return
        self.owner_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [e.to_dict() for e in self.entries.values() if e.source == "owner"]
        tmp = self.owner_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "entries": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.owner_path)

    def set(self, surface: str, spoken: str, *, language: str = "de", provider: str = "piper_espeak", note: str = "") -> LexiconEntry:
        """An owner correction: how ``surface`` is to be spoken from now on."""

        surface = surface.strip()
        key = (surface.lower(), language)
        entry = self.entries.get(key)
        if entry is None or entry.source != "owner":
            entry = LexiconEntry(surface, language, {}, source="owner")
        entry.spoken_as[provider] = spoken.strip()
        entry.spoken_as.setdefault("generic", spoken.strip())
        entry.note = note or entry.note
        entry.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry.source = "owner"
        self.entries[key] = entry
        self.save()
        return entry

    def remove(self, surface: str, *, language: str = "de") -> bool:
        key = (surface.strip().lower(), language)
        entry = self.entries.get(key)
        if entry is None or entry.source != "owner":
            return False
        del self.entries[key]
        for seed in _seed():
            if (seed.surface.lower(), seed.language) == key:
                self.entries[key] = seed
        self.save()
        return True

    def lookup(self, surface: str, language: str) -> LexiconEntry | None:
        return self.entries.get((surface.lower(), language)) or self.entries.get((surface.lower(), "*"))

    def owner_entries(self) -> list[LexiconEntry]:
        return [e for e in self.entries.values() if e.source == "owner"]

    def all(self) -> list[LexiconEntry]:
        return sorted(self.entries.values(), key=lambda e: (e.language, e.surface.lower()))

    def apply(self, text: str, *, language: str, provider: str = "piper_espeak") -> str:
        """Replace known surfaces (longest first, whole-word, case-insensitive) by their spoken form."""

        candidates = [e for e in self.entries.values() if e.language in (language, "*") and e.rendering(provider)]
        candidates.sort(key=lambda e: -len(e.surface))
        out = text
        for entry in candidates:
            pattern = re.compile(r"(?<![\w-])" + re.escape(entry.surface) + r"(?![\w-])", re.I)
            out = pattern.sub(entry.rendering(provider), out)
        return out


# --------------------------------------------------------------------------
# Normalisation (language-aware, provider-independent)
# --------------------------------------------------------------------------

_URL = re.compile(r"\bhttps?://([^\s/]+)(/[^\s]*)?", re.I)
_UNIT = re.compile(r"(?<![\w])(\d+(?:[.,]\d+)?)\s?(ms|min|mib|gib|gb|mb|kb|tb|ghz|mhz|khz|hz|fps|px|kg|km|cm|mm|kw|%|°c|s|h|w|v)(?![\w])", re.I)
_ACRONYM = re.compile(r"\b([A-Z]{2,5})(?![a-z])\b")
_CAMEL = re.compile(r"\b([A-Z][a-z]+)([A-Z][a-z]+)\b")
_NUMBER_DE = re.compile(r"(\d+)\.(\d{3})(?!\d)")


def spell(acronym: str, language: str) -> str:
    letters = LETTERS_DE if language.startswith("de") else LETTERS_EN
    return "-".join(letters.get(ch.lower(), ch) for ch in acronym)


def normalize(text: str, *, language: str = "de") -> str:
    """Numbers, units, URLs and acronyms into words the phonemiser reads right."""

    de = language.startswith("de")
    out = text
    # URLs: "https://lichess.org/analysis" -> "lichess Punkt org Schrägstrich analysis"
    def url(m: re.Match) -> str:
        host = m.group(1).replace(".", " Punkt " if de else " dot ")
        path = (m.group(2) or "").strip("/")
        if path:
            host += (" Schrägstrich " if de else " slash ") + path.replace("/", " Schrägstrich " if de else " slash ").replace("-", " ").replace("_", " ")
        return host
    out = _URL.sub(url, out)
    # units after numbers
    units = UNITS_DE if de else UNITS_EN
    out = _UNIT.sub(lambda m: f"{m.group(1)} {units.get(m.group(2).lower(), m.group(2))}", out)
    # thousands separators the German way stay numbers; the phonemiser reads "12.500" as twelve point five hundred
    if de:
        out = _NUMBER_DE.sub(lambda m: f"{m.group(1)}{m.group(2)}", out)
    # acronyms: spelled out unless they are read as words
    out = _ACRONYM.sub(lambda m: m.group(1) if m.group(1).lower() in WORD_ACRONYMS else spell(m.group(1), language), out)
    return out


@dataclass
class Spoken:
    displayed: str
    spoken: str
    language: str
    provider: str
    changed: bool

    def to_dict(self) -> dict[str, Any]:
        return {"displayed": self.displayed, "spoken": self.spoken, "language": self.language, "provider": self.provider, "changed": self.changed}


class Pronouncer:
    """The pipeline: normalise, then lexicon, for one provider family."""

    def __init__(self, lexicon: Lexicon | None = None, *, provider: str = "piper_espeak") -> None:
        self.lexicon = lexicon or Lexicon()
        self.provider = provider

    def render(self, text: str, *, language: str = "de") -> Spoken:
        language = (language or "de")[:2].lower()
        # lexicon first on the original spelling (so "GPU" maps by the entry,
        # not by the generic speller), then normalisation for the rest
        spoken = self.lexicon.apply(text, language=language, provider=self.provider)
        spoken = normalize(spoken, language=language)
        spoken = re.sub(r"[ \t]{2,}", " ", spoken).strip()
        return Spoken(text, spoken, language, self.provider, spoken != text.strip())


#: The pronunciation acceptance set (§10): surfaces the tests check end to end.
ACCEPTANCE_SET = ["Zeus", "Spotify", "GitHub", "OpenAI", "GPU", "CPU", "Whisper", "Piper", "SelfDev", "Knowledge Graph",
                  "Mission Control", "Stockfish", "Physikum", "Biochemie", "Gluconeogenese", "Desoxyribonukleinsäure",
                  "Mitochondrien", "Carbonsäure", "Physiologie"]
