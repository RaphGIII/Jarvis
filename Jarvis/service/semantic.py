"""The semantic control plane: FAST_LOCAL understands the goal, tools execute.

Before this module, natural language reached the tools through lexical
matchers: keyword tables, stem regexes and term-overlap capability scoring.
Those remain as FAST PATHS for unambiguous commands ("wie spät ist es?"), but
they are no longer the intelligence.  When the deterministic layer cannot
produce a typed action, the request comes HERE: a schema-constrained local
model call turns the owner's words into one goal from a CLOSED set of
operations.  The model cannot invent a tool — the schema's enum makes a
wrong tool unrepresentable, not merely discouraged — and it never executes
anything itself; the typed dispatchers in ``service.core`` do, with the same
verification they always had.

Latency: one FAST_LOCAL structured generation (~1–3 s warm).  The prompt is
deliberately small — prompt size, not model size, dominates wall clock on
this machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

#: The closed set of goal operations the planner may choose.  Every entry has
#: a dispatcher in service.core; adding one here without a dispatcher would
#: reintroduce the dead-end this layer exists to remove.
OPERATIONS = (
    "web.open",
    "web.search",
    "research",
    "app.open",
    "file.open",
    "folder.open",
    "project.open",
    "music.control",
    "system.open_view",
    "system.tell_time",
    "system.tell_date",
    "knowledge.search",
    "calendar.create",
    "calendar.query",
    "image.generate",
    "fs.count",
    "fs.largest",
    "fs.list",
    "fs.open",
    "web.read_summary",
    "capability.run",
    "clarify",
    "capability.missing",
    "delegate",
    "conversation",
)

#: Soft task features the planner estimates in the same call (0..1 each).
#: They may only RAISE the routing vector (D_final = max(rule, semantic)) and
#: decide how much thinking the answer gets; ambiguity is not among them
#: because the planner already says "clarify" when it is really ambiguous.
SOFT_FEATURES = ("reasoning_depth", "context_dependency", "long_horizon", "novelty")

GOAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": list(OPERATIONS)},
        "target": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "question": {"type": "string"},
        "reason": {"type": "string"},
        "features": {
            "type": "object",
            "properties": {name: {"type": "number", "minimum": 0.0, "maximum": 1.0} for name in SOFT_FEATURES},
        },
    },
    "required": ["operation", "target", "confidence", "reason"],
}


class SemanticAuthorityUnavailable(RuntimeError):
    """No configured reasoning provider can decide what the owner means.

    Raised instead of asking the legacy local model.  That model may answer
    prose; it never infers intent, chooses a goal, a plan or a capability, and
    it never stands in after a provider failure.
    """


def is_provider_outage(exc: BaseException) -> bool:
    """Whether an exception from a decision call means "no semantic provider", not "bad answer"."""

    if isinstance(exc, SemanticAuthorityUnavailable):
        return True
    if hasattr(exc, "decision"):  # GatewayRefused: no route in this mode / budget / health
        return True
    status = getattr(exc, "status", None)
    return bool(getattr(status, "is_outage", False))


@dataclass
class SemanticGoal:
    """What the owner wants, as one typed goal.  A proposal, never a result."""

    operation: str
    target: str = ""
    confidence: float = 0.0
    question: str = ""
    reason: str = ""
    elapsed_ms: float = 0.0
    #: The planner's estimate of the soft task features, 0..1 each.
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "target": self.target,
                "confidence": round(self.confidence, 2), "question": self.question,
                "reason": self.reason, "elapsed_ms": round(self.elapsed_ms, 1),
                "features": {k: round(v, 2) for k, v in self.features.items()}}


PROMPT = """Du bist die semantische Steuerung von ZEUS, einem lokalen Assistenten auf einem Windows-PC.
Bestimme das ZIEL der Anfrage und wähle GENAU EIN Werkzeug. Du führst nichts aus.

Werkzeuge:
- web.open: eine Website im Browser öffnen. target = Name oder URL. "Öffne Wikipedia." -> web.open "Wikipedia". "Bring mich zu GitHub." -> web.open "GitHub".
- web.search: im Internet suchen. target = Suchanfrage.
- research: Frage nach aktuellem Weltgeschehen/Nachrichten, braucht frische Quellen. target = Frage. "Was ist heute passiert?" -> research.
- app.open: ein installiertes Programm starten. target = Programmname. "Mach Spotify auf." -> app.open "Spotify".
- file.open: eine bestimmte Datei öffnen. target = Pfad oder ihr Name.
- folder.open: einen Ordner öffnen/zeigen. target = Pfad oder sein Name. "Zeig mir den Ordner mit meinen Uni-Sachen." -> folder.open "Uni-Sachen".
- project.open: ein ZEUS-Projekt öffnen. Nur wenn eines aus der Projektliste gemeint ist. target = Projektname.
- music.control: Musik abspielen, pausieren, weiter, lauter. target = was geschehen/gespielt werden soll. "Spiel Rammstein." -> music.control "Rammstein abspielen".
- system.open_view: eine ZEUS-Ansicht öffnen (projects, files, knowledge, missions, activity, owner, voice). target = Ansicht. "Zeig mir meine Dateien/Projekte." -> system.open_view "files"/"projects".
- system.tell_time: Uhrzeit sagen. system.tell_date: Datum sagen. target = "".
- knowledge.search: im gespeicherten Wissen von ZEUS suchen. target = Suchbegriff.
- calendar.create: einen Termin eintragen. target = die Anfrage wörtlich. "Leg morgen Nachmittag zwei Stunden fürs Physikum frei." -> calendar.create.
- calendar.query: nach Terminen fragen. target = die Frage wörtlich.
- image.generate: ein Bild lokal erzeugen. target = die Bildbeschreibung. "Erzeuge mir ein Bild von einem Adler." -> image.generate "ein Adler".
- fs.count: Ordner/Dateien zählen. target = Ordnername oder Pfad. "Wie viele Unterordner hat mein Jarvis-Ordner?" -> fs.count "Jarvis". Du HAST vollen Dateisystem-Zugriff.
- fs.largest: den größten Ordner/die größte Datei finden. target = Pfad oder Laufwerk. "Welcher Ordner auf D: ist am größten?" -> fs.largest "D:\\".
- fs.list: den direkten Inhalt eines Ordners zeigen. target = Pfad, Name oder "dein Repo". "Was ist direkt in deinem Repo?" -> fs.list "dein Repo".
- fs.open: einen Ordner im Explorer öffnen. target = Pfad, Name oder "dein Repo".
- Dateisystem-Fragen (Ordner, Dateien, Größe, "dein Repo", Laufwerk D:) sind IMMER fs.*-Werkzeuge — NIEMALS screen/capture/file.read. Behaupte nie fehlenden Zugriff.
- web.read_summary: einen Artikel aus den letzten Suchergebnissen (oder eine URL) WIRKLICH lesen und zusammenfassen. target = Bezug oder URL.
- capability.run: eine INSTALLIERTE ZEUS-Fähigkeit aus der Liste unten ausführen, wenn sie die Anfrage wirklich erfüllt – auch bei anderer Wortwahl ("Fingerprint der Datei" = Prüfsumme). target = capability_id EXAKT aus der Liste. Nie eine Fähigkeit erfinden; passt keine, wähle etwas anderes.
- clarify: WIRKLICH mehrdeutig - stelle GENAU EINE kurze Frage (Feld question).
- capability.missing: eine echte Handlung, für die es hier kein Werkzeug gibt (Gerät steuern, Datei konvertieren, E-Mail senden ...). target = das Ziel.
- delegate: etwas ERSTELLEN oder ÄNDERN (Datei, Notiz, Projekt, Code) - der Ausführungsplaner übernimmt. target = "".
- conversation: reine Unterhaltung oder Wissensfrage ohne Handlung. target = "".

Regeln:
- Bekannte Websites (Wikipedia, YouTube, GitHub, Amazon ...) sind web.open - NIEMALS app.open, außer die Liste installierter Apps enthält sie.
- Bei "öffne X": entscheide semantisch, ob X Website, App, Datei, Ordner oder Projekt ist. Nutze die Kontextlisten.
- Mehrschritt ("Öffne Spotify und spiel Rammstein"): wähle das dominante Ziel (hier music.control).
- target MUSS wörtlich aus der Anfrage oder den Kontextlisten stammen - NIEMALS erfunden.
- Erfinde keine Pfade und keine Ziele. Ist der Ort unbekannt, aber das Ziel klar, wähle das Werkzeug trotzdem - die Ausführung fragt nach dem Ort.
- confidence ehrlich: über 0.85 nur, wenn die Absicht eindeutig ist.
- reason: ein kurzer Satz.
- features: schätze je 0..1 — reasoning_depth (wie viel Denkarbeit eine gute Antwort braucht: Faktenfrage 0.1, Herleitung/Planung 0.7+), context_dependency (wie stark die Anfrage auf Gesprächskontext, Projekte oder Vorheriges Bezug nimmt), long_horizon (mehrere Schritte über längere Zeit), novelty (etwas, das ZEUS so noch nicht kann).

{context}Anfrage: {request}
JSON:"""


def _context_lines(*, apps: Iterable[str] = (), projects: Iterable[str] = (),
                   aliases: Iterable[dict[str, Any]] = (), guidance: str = "",
                   capabilities: Iterable[dict[str, Any]] = ()) -> str:
    lines: list[str] = []
    apps = [a for a in apps if a][:8]
    projects = [p for p in projects if p][:8]
    aliases = list(aliases)[:5]
    capabilities = list(capabilities)[:15]
    if apps:
        lines.append("Passende installierte Apps: " + ", ".join(apps))
    if projects:
        lines.append("Passende ZEUS-Projekte: " + ", ".join(projects))
    if capabilities:
        rows = []
        for cap in capabilities:
            examples = "; ".join(str(e) for e in (cap.get("examples") or [])[:2])
            rows.append(f"  - {cap.get('id')}: {str(cap.get('description', ''))[:120]}" + (f" (z.B. {examples})" if examples else ""))
        lines.append("Installierte ZEUS-Fähigkeiten (capability.run, target = id):\n" + "\n".join(rows))
    for entry in aliases:
        lines.append(f"Owner-Alias: „{entry.get('name')}“ = {entry.get('kind')} {entry.get('value')}")
    if guidance.strip():
        lines.append("Frühere Korrekturen des Owners (haben Vorrang):\n" + guidance.strip())
    return ("\n".join(lines) + "\n\n") if lines else ""


class SemanticPlanner:
    """One structured FAST_LOCAL call: owner words in, one typed goal out."""

    def plan(self, request: str, provider: Any, *,
             apps: Iterable[str] = (), projects: Iterable[str] = (),
             aliases: Iterable[dict[str, Any]] = (), guidance: str = "",
             capabilities: Iterable[dict[str, Any]] = ()) -> SemanticGoal | None:
        request = str(request or "").strip()
        if not request:
            return None
        prompt = PROMPT.format(context=_context_lines(apps=apps, projects=projects, aliases=aliases, guidance=guidance,
                                                      capabilities=capabilities),
                               request=request)
        started = time.perf_counter()
        raw = ""
        try:
            if hasattr(provider, "generate_structured"):
                raw = provider.generate_structured(prompt, GOAL_SCHEMA, max_tokens=220, temperature=0.0)
            else:
                raw = provider.generate(prompt, max_tokens=220, temperature=0.0)
        except TypeError:
            try:
                raw = provider.generate(prompt)
            except Exception as exc:  # noqa: BLE001
                if is_provider_outage(exc):
                    raise
                return None
        except Exception as exc:  # noqa: BLE001 - a bad answer means no goal; a missing provider is the caller's to report
            if is_provider_outage(exc):
                raise
            return None

        from brain.json_utils import lenient_json_loads

        try:
            payload = lenient_json_loads(str(raw))
        except Exception:  # noqa: BLE001
            payload = None
        if not isinstance(payload, dict):
            return None
        operation = str(payload.get("operation") or "").strip()
        if operation not in OPERATIONS:
            return None
        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        features: dict[str, float] = {}
        raw_features = payload.get("features")
        if isinstance(raw_features, dict):
            for name in SOFT_FEATURES:
                try:
                    value = float(raw_features.get(name))
                except (TypeError, ValueError):
                    continue
                features[name] = max(0.0, min(1.0, value))
        return SemanticGoal(
            operation=operation,
            target=str(payload.get("target") or "").strip()[:400],
            confidence=confidence,
            question=str(payload.get("question") or "").strip()[:300],
            reason=str(payload.get("reason") or "").strip()[:300],
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            features=features,
        )
