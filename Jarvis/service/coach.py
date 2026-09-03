"""The language coach: a real adaptive tutor inside the conversation.

"Zeus, lass uns 10 Minuten Französisch üben." starts a SESSION: the coach
speaks in the target language, the owner answers (typed or spoken), and each
turn is evaluated by FAST_LOCAL with a structured schema — meaning, one
concise correction, new vocabulary.  The learner model persists: vocabulary
strengths with spaced-repetition due dates, recurring mistakes, session
history.  A session ends on request or after the planned number of turns,
writes its summary into the Wissen library, and never overcorrects — one
main correction per turn, not a red-pen bath.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LANGUAGES = {"franzoesisch": "Französisch", "französisch": "Französisch", "french": "Französisch",
             "englisch": "Englisch", "english": "Englisch",
             "spanisch": "Spanisch", "spanish": "Spanisch",
             "italienisch": "Italienisch", "latein": "Latein"}

#: strength 0..4 → review interval in days
_INTERVALS = {0: 0.5, 1: 1, 2: 3, 3: 7, 4: 14}

TURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verstanden": {"type": "string"},
        "korrektur": {"type": "string"},
        "fehler_art": {"type": "string", "enum": ["", "grammatik", "vokabel", "wortstellung", "aussprache", "keine"]},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "antwort": {"type": "string"},
        "neue_vokabeln": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["korrektur", "score", "antwort"],
}

TURN_PROMPT = """Du bist ein geduldiger {language}-Lehrer für einen deutschen Muttersprachler (Niveau: {level}).
Ihr führt ein kurzes Gespräch auf {language}. Thema: {topic}.

Bisheriger Dialog:
{dialog}

Der Schüler sagt: "{answer}"

Bewerte NUR den letzten Satz des Schülers:
- verstanden: was der Schüler sagen wollte (deutsch, 1 Satz).
- korrektur: die EINE wichtigste Verbesserung, kurz und konkret (deutsch erklärt, mit der richtigen {language}-Form). Leer, wenn der Satz gut war.
- fehler_art: grammatik | vokabel | wortstellung | keine.
- score: 0..1 (1 = fehlerfrei und natürlich).
- antwort: deine nächste Gesprächszeile auf {language} — kurz, dem Niveau angemessen, treibt das Gespräch weiter. Bei sehr niedrigem Niveau füge in Klammern die deutsche Übersetzung an.
- neue_vokabeln: bis zu 3 {language}-Wörter aus deiner Antwort, die der Schüler lernen sollte (Form: "wort = bedeutung").

JSON:"""

OPENERS = {
    "Französisch": [("Bonjour ! Comment ça va aujourd'hui ?", "Begrüßung"),
                     ("Qu'est-ce que tu as fait ce week-end ?", "Wochenende"),
                     ("Tu préfères le café ou le thé ? Pourquoi ?", "Vorlieben"),
                     ("Imagine: nous sommes au restaurant. Tu veux commander.", "Restaurant-Rollenspiel")],
    "Englisch": [("Hi! How was your day so far?", "Begrüßung"),
                  ("Let's do a roleplay: you are at the airport check-in.", "Flughafen-Rollenspiel"),
                  ("Tell me about a project you are working on.", "Projekte")],
    "Spanisch": [("¡Hola! ¿Cómo estás hoy?", "Begrüßung"),
                  ("¿Qué te gusta hacer el fin de semana?", "Freizeit")],
    "Italienisch": [("Ciao! Come stai oggi?", "Begrüßung")],
    "Latein": [("Salve! Quid agis hodie?", "Begrüßung")],
}


@dataclass
class CoachSession:
    language: str
    minutes: int = 10
    topic: str = ""
    started_at: float = field(default_factory=time.time)
    turns: list[dict[str, Any]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    new_vocab: list[str] = field(default_factory=list)
    mistakes: list[str] = field(default_factory=list)

    @property
    def expired(self) -> bool:
        return time.time() - self.started_at > self.minutes * 60

    def dialog_text(self, limit: int = 8) -> str:
        lines = []
        for t in self.turns[-limit:]:
            lines.append(f"Lehrer: {t.get('coach', '')}")
            if t.get("student"):
                lines.append(f"Schüler: {t['student']}")
        return "\n".join(lines)


class LanguageCoach:
    """Learner model + session flow; evaluation is the caller's provider."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        self.session: CoachSession | None = None

    # -- learner model ---------------------------------------------------

    def _model_path(self, language: str) -> Path:
        return self.root / f"learner_{language.lower()}.json"

    def learner(self, language: str) -> dict[str, Any]:
        try:
            return json.loads(self._model_path(language).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"language": language, "level": "Anfänger", "vocabulary": {}, "mistakes": [], "sessions": []}

    def _save_learner(self, model: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._model_path(model["language"]).write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    def due_vocabulary(self, language: str, limit: int = 8) -> list[str]:
        model = self.learner(language)
        now = time.time()
        due = [(w, v) for w, v in model.get("vocabulary", {}).items() if v.get("due", 0) <= now]
        due.sort(key=lambda item: item[1].get("strength", 0))
        return [w for w, _ in due[:limit]]

    # -- sessions --------------------------------------------------------

    def start(self, language: str, minutes: int = 10) -> dict[str, Any]:
        opener, topic = random.choice(OPENERS.get(language, OPENERS["Englisch"]))
        review = self.due_vocabulary(language, limit=4)
        self.session = CoachSession(language=language, minutes=minutes, topic=topic)
        self.session.turns.append({"coach": opener})
        model = self.learner(language)
        intro = (f"Los geht's — {minutes} Minuten {language} (Niveau: {model.get('level', 'Anfänger')}). "
                 + (f"Zur Wiederholung fällig: {', '.join(review)}. " if review else "")
                 + "Sag „Übung beenden“, wenn du aufhören willst.\n\n" + opener)
        return {"ok": True, "text": intro, "topic": topic, "review": review}

    def evaluate_turn(self, provider: Any, answer: str) -> dict[str, Any]:
        session = self.session
        if session is None:
            return {"ok": False, "error": "keine aktive Übung"}
        model = self.learner(session.language)
        prompt = TURN_PROMPT.format(language=session.language, level=model.get("level", "Anfänger"),
                                    topic=session.topic, dialog=session.dialog_text(), answer=answer)
        from brain.json_utils import lenient_json_loads

        if hasattr(provider, "generate_structured"):
            raw = provider.generate_structured(prompt, TURN_SCHEMA, max_tokens=350, temperature=0.4)
        else:
            raw = provider.generate(prompt, max_tokens=350)
        payload = lenient_json_loads(str(raw))
        if not isinstance(payload, dict) or not str(payload.get("antwort", "")).strip():
            return {"ok": False, "error": "die Bewertung kam nicht durch"}
        correction = str(payload.get("korrektur", "")).strip()
        score = max(0.0, min(1.0, float(payload.get("score") or 0.5)))
        reply = str(payload.get("antwort", "")).strip()
        vocab = [str(v).strip() for v in (payload.get("neue_vokabeln") or []) if str(v).strip()][:3]
        session.turns[-1]["student"] = answer
        session.turns[-1]["score"] = score
        if correction:
            session.turns[-1]["korrektur"] = correction
            session.mistakes.append(correction)
        session.scores.append(score)
        session.new_vocab.extend(v for v in vocab if v not in session.new_vocab)
        session.turns.append({"coach": reply})
        lines = []
        if correction and score < 0.9:
            lines.append(f"✎ {correction}")
        lines.append(reply)
        return {"ok": True, "text": "\n".join(lines), "score": score, "done": session.expired}

    def finish(self) -> dict[str, Any]:
        session, self.session = self.session, None
        if session is None:
            return {"ok": False, "error": "keine aktive Übung"}
        model = self.learner(session.language)
        now = time.time()
        for entry in session.new_vocab:
            word = entry.split("=", 1)[0].strip()
            if not word:
                continue
            item = model["vocabulary"].get(word, {"strength": 0, "meaning": entry.partition("=")[2].strip()})
            item["strength"] = min(4, int(item.get("strength", 0)) + 1)
            item["last_seen"] = now
            item["due"] = now + _INTERVALS[item["strength"]] * 86400
            model["vocabulary"][word] = item
        model["mistakes"] = (model.get("mistakes", []) + session.mistakes)[-40:]
        average = round(sum(session.scores) / len(session.scores), 2) if session.scores else None
        record = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "minutes": session.minutes,
                  "turns": len(session.scores), "average_score": average,
                  "new_vocab": session.new_vocab, "mistakes": session.mistakes[:6], "topic": session.topic}
        model["sessions"] = (model.get("sessions", []) + [record])[-30:]
        # a simple adaptive level: sustained high scores move the learner up
        recent = [s.get("average_score") or 0 for s in model["sessions"][-3:]]
        if len(recent) == 3 and min(recent) >= 0.8 and model.get("level") == "Anfänger":
            model["level"] = "Mittelstufe"
        self._save_learner(model)
        lines = [f"Übung beendet — {session.language}, {len(session.scores)} Antworten."]
        if average is not None:
            lines.append(f"Durchschnitt: {round(average * 100)}%.")
        if session.new_vocab:
            lines.append("Neue Vokabeln: " + "; ".join(session.new_vocab[:6]) + ".")
        if session.mistakes:
            lines.append("Woran wir arbeiten: " + " | ".join(session.mistakes[:3]))
        lines.append("Nächste Wiederholung ist eingeplant.")
        return {"ok": True, "text": "\n".join(lines), "record": record, "language": session.language,
                "summary_note": "\n".join([f"# {session.language}-Übung {record['at']}", "",
                                           f"Thema: {session.topic} · {record['turns']} Antworten · Ø {average}",
                                           "", "## Neue Vokabeln",
                                           *([f"- {v}" for v in session.new_vocab] or ["- (keine)"]),
                                           "", "## Hauptfehler",
                                           *([f"- {m}" for m in session.mistakes[:6]] or ["- (keine)"])])}
