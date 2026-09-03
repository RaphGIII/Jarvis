"""The owner speech corpus: real recordings with owner-verified transcripts.

This is the ground truth every STT decision must be measured against.  Each
entry is one utterance: the audio file as recorded, the transcript the OWNER
confirmed (typed or corrected), and the conditions.  The benchmark harness
(speech/benchmark.py, run in the speech venv) replays these recordings
through candidate models and reports WER/CER/entity accuracy/latency —
numbers on THIS voice on THIS machine, never synthetic claims.

Metrics live here too (pure Python, no model dependencies) so the core, the
benchmark and the tests share one definition of WER.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any

#: The recording script: broad coverage of what the owner actually says.
#: Categories from the sprint: normal sentences, tempo variants, commands,
#: technical vocabulary, project names, medical vocabulary, mixed DE/EN,
#: numbers, URLs, application names.
PHRASES: tuple[tuple[str, str], ...] = (
    ("normal", "Heute war ein langer Tag und ich möchte noch kurz meine Notizen durchsehen."),
    ("normal", "Kannst du mir sagen, was morgen ansteht?"),
    ("normal", "Ich habe das Gefühl, dass wir das Projekt bald abschließen können."),
    ("normal", "Das Wetter soll am Wochenende deutlich besser werden."),
    ("normal", "Erinnere mich bitte daran, die Unterlagen mitzunehmen."),
    ("schnell", "Mach schnell Spotify auf und spiel meine Playlist."),
    ("schnell", "Zeig mir sofort die Projekte, ich habe es eilig."),
    ("langsam", "Öffne … bitte … ganz in Ruhe … den Kalender."),
    ("befehl", "Öffne Wikipedia."),
    ("befehl", "Mach Spotify auf."),
    ("befehl", "Spiel Rammstein."),
    ("befehl", "Wie spät ist es?"),
    ("befehl", "Trag morgen um 14 Uhr Lernen ein."),
    ("befehl", "Stopp."),
    ("befehl", "Zeig mir meine Dateien."),
    ("befehl", "Such im Internet nach den neuesten Nachrichten."),
    ("technik", "Der Server läuft auf Port 8420 und antwortet über die API."),
    ("technik", "Das Repository liegt auf dem Laufwerk D unter Jarvis Recovery."),
    ("technik", "Ollama lädt das Modell Qwen in den Grafikspeicher."),
    ("technik", "Der Supervisor startet den Core neu, wenn der Health-Check fehlschlägt."),
    ("technik", "Whisper transkribiert die Aufnahme und Piper spricht die Antwort."),
    ("projekt", "Öffne das Projekt Physikum."),
    ("projekt", "Wie steht es um die Mission Control?"),
    ("projekt", "Zeus, zeig mir die Galaxy mit allen Capabilities."),
    ("medizin", "Die Biochemie-Klausur behandelt Glykolyse und Citratzyklus."),
    ("medizin", "Anatomie und Physiologie sind die Grundlage für das Physikum."),
    ("medizin", "Der Patient bekommt zweimal täglich zehn Milligramm."),
    ("medizin", "Die Differentialdiagnose umfasst Pneumonie und Bronchitis."),
    ("gemischt", "Ich pushe den Branch später auf GitHub und mache einen Pull Request."),
    ("gemischt", "Das Update vom Voice Studio ist ready für den Test."),
    ("gemischt", "Downloade das File und speichere es im Ordner Downloads."),
    ("zahlen", "Der Termin ist am dritten Oktober um Viertel nach neun."),
    ("zahlen", "Die Datei hat zweitausenddreihundertfünfzig Zeilen und achtzehn Fehler."),
    ("zahlen", "Rechne 17 mal 23 und sag mir das Ergebnis."),
    ("url", "Öffne w w w Punkt wikipedia Punkt org im Browser."),
    ("url", "Die Seite heißt github Punkt com Schrägstrich anthropics."),
    ("app", "Starte den Editor und den Taschenrechner."),
    ("app", "Öffne Microsoft Edge und danach den Explorer."),
    ("app", "Schließe Spotify und öffne stattdessen Stockfish."),
    ("frage", "Was ist heute in der Welt passiert?"),
    ("normal", "Kannst du mir kurz zusammenfassen, was wir gestern besprochen haben?"),
    ("normal", "Ich bin gleich zurück, mach solange einfach weiter."),
    ("normal", "Das klingt gut, aber lass uns zuerst das andere Thema abschließen."),
    ("normal", "Am Wochenende möchte ich endlich wieder Zeit zum Lernen finden."),
    ("normal", "Kannst du das bitte noch einmal langsamer erklären?"),
    ("normal", "Ich verstehe den Unterschied zwischen den beiden Varianten nicht."),
    ("normal", "Lass uns das morgen früh direkt als Erstes angehen."),
    ("normal", "Der Termin gestern hat länger gedauert als geplant."),
    ("normal", "Ich habe darüber nachgedacht und bin anderer Meinung."),
    ("normal", "Wie war das noch mal mit der Einstellung von vorhin?"),
    ("befehl", "Öffne meinen Uni-Planer."),
    ("befehl", "Zeig mir die Projekte."),
    ("befehl", "Mach die Musik leiser."),
    ("befehl", "Pausiere die Wiedergabe."),
    ("befehl", "Erstelle ein Bild von einem Sonnenuntergang über den Bergen."),
    ("befehl", "Lösche den letzten Eintrag wieder."),
    ("befehl", "Zeig mir den Ordner mit meinen Dokumenten."),
    ("befehl", "Öffne den Kalender und zeig mir nächste Woche."),
    ("befehl", "Erinnere mich in zwanzig Minuten an die Pause."),
    ("befehl", "Fass mir den Artikel zusammen."),
    ("befehl", "Wie viele Unterordner hat mein Jarvis-Ordner?"),
    ("befehl", "Speichere das bitte im Wissen ab."),
    ("dateien", "Öffne die Datei Hausarbeit Punkt Docx im Ordner Studium."),
    ("dateien", "Verschiebe die Präsentation in den Ordner Archiv."),
    ("dateien", "Wo liegt die letzte Version vom Lernplan?"),
    ("dateien", "Erstelle einen neuen Ordner namens Prüfungsvorbereitung."),
    ("dateien", "Zeig mir alle PDFs aus dem Download-Ordner."),
    ("dateien", "Auf Laufwerk D liegt der Ordner Jarvis Recovery."),
    ("technik", "Starte den Server neu und prüfe die Logdatei."),
    ("technik", "Der Commit gestern hat den Bug in der Pipeline behoben."),
    ("technik", "Die API antwortet mit einem Fehler vierhundertdrei."),
    ("technik", "Das Backup läuft jeden Abend um zweiundzwanzig Uhr."),
    ("technik", "Der Arbeitsspeicher ist fast voll, schließe ein paar Programme."),
    ("technik", "Die Grafikkarte hat acht Gigabyte Speicher."),
    ("technik", "Installiere das Update erst nach dem Wochenende."),
    ("technik", "Der Selfdev-Lauf hat alle Tests bestanden."),
    ("projekt", "Wie steht es um das Projekt Physikum?"),
    ("projekt", "Leg ein neues Projekt namens Sommerplanung an."),
    ("projekt", "Welche Aufgaben sind im Projekt Zeus noch offen?"),
    ("projekt", "Verschiebe die Deadline auf Ende September."),
    ("projekt", "Markiere das Projekt als abgeschlossen."),
    ("medizin", "Die Glykolyse findet im Zytoplasma statt."),
    ("medizin", "Erkläre mir den Unterschied zwischen Arterien und Venen."),
    ("medizin", "Das Herzzeitvolumen steigt unter Belastung deutlich an."),
    ("medizin", "Die Niere reguliert den Wasser- und Elektrolythaushalt."),
    ("medizin", "Wiederhole mit mir die Hirnnerven von eins bis zwölf."),
    ("medizin", "Adenosintriphosphat ist der universelle Energieträger der Zelle."),
    ("medizin", "Der Patient klagt über Schmerzen im rechten Oberbauch."),
    ("medizin", "Morgen schreibe ich die Klausur in Physiologie."),
    ("gemischt", "Merge den Branch erst nach dem Code-Review."),
    ("gemischt", "Das Feature ist deployed, aber der Load-Balancer zickt noch."),
    ("gemischt", "Schick mir den Link später per Mail."),
    ("gemischt", "Wir machen heute Abend ein kurzes Standup-Meeting."),
    ("gemischt", "Der Stream startet um acht, hast du den Login?"),
    ("gemischt", "Ich habe das File schon gedownloadet und entpackt."),
    ("zahlen", "Der Kurs liegt bei dreiundzwanzig Komma sieben Prozent."),
    ("zahlen", "Meine Matrikelnummer ist vier fünf acht zwei sieben eins."),
    ("zahlen", "Das Meeting geht von halb zehn bis Viertel vor zwölf."),
    ("zahlen", "Im Jahr zweitausendvierundzwanzig war das noch anders."),
    ("zahlen", "Nimm die Version eins Punkt zwei Punkt siebzehn."),
    ("zahlen", "Die Datei ist ungefähr dreihundertfünfzig Megabyte groß."),
    ("datum", "Trag den Termin für den fünften Oktober um vierzehn Uhr ein."),
    ("datum", "Nächsten Dienstag habe ich von neun bis elf Vorlesung."),
    ("datum", "Verschiebe alles vom Freitag auf den Montag danach."),
    ("datum", "Am einunddreißigsten Dezember ist Silvester."),
    ("url", "Geh auf hackernews Punkt com und zeig mir die Top-Beiträge."),
    ("url", "Die Doku findest du unter docs Punkt python Punkt org."),
    ("url", "Öffne amboss Punkt com für die Lernkarten."),
    ("app", "Starte Blender und öffne das letzte Projekt."),
    ("app", "Mach Discord auf und stell mich auf abwesend."),
    ("app", "Öffne die Einstellungen von Windows."),
    ("schnell", "Schnell, mach den Kalender auf, ich bin spät dran."),
    ("schnell", "Spotify an, Playlist Fokus, Lautstärke runter, danke."),
    ("schnell", "Zeig mir sofort den letzten Screenshot."),
    ("langsam", "Bitte … trage … den Termin … ganz genau … so ein."),
    ("langsam", "Ich diktiere jetzt … langsam … einen wichtigen Satz."),
    ("leise", "Kannst du mich auch verstehen, wenn ich leise spreche?"),
    ("leise", "Der Termin morgen ist mir wichtig, vergiss ihn nicht."),
    ("leise", "Mach die Musik bitte etwas leiser, danke."),
    ("frage", "Was bedeutet das Wort Ambivalenz?"),
    ("frage", "Wie lange dauert der Weg zur Uni mit dem Rad?"),
    ("frage", "Welche Termine habe ich diese Woche noch?"),
    ("frage", "Was ist der Unterschied zwischen RAM und VRAM?"),
    ("zeus", "Zeus, wach auf."),
    ("zeus", "Zeus, zeig dich auf dem Fernseher."),
    ("zeus", "Zeus, lass uns zehn Minuten Französisch üben."),
    ("zeus", "Zeus, starte den Lernmodus."),
    ("zeus", "Zeus, was läuft gerade im Hintergrund?"),
    ("zeus", "Zeus, brich die Bilderzeugung ab."),
)


def _fold(text: str) -> str:
    lowered = str(text or "").lower().replace("ß", "ss")
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        lowered = lowered.replace(a, b)
    out = "".join(ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", out)


def _words(text: str) -> list[str]:
    return [w for w in _fold(text).split() if w]


def _levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(truth: str, hypothesis: str) -> float:
    """Word error rate on folded words; 0.0 is perfect, can exceed 1.0."""

    ref = _words(truth)
    if not ref:
        return 0.0 if not _words(hypothesis) else 1.0
    return _levenshtein(ref, _words(hypothesis)) / len(ref)


def cer(truth: str, hypothesis: str) -> float:
    ref = list(_fold(truth).replace(" ", ""))
    if not ref:
        return 0.0 if not _fold(hypothesis).strip() else 1.0
    return _levenshtein(ref, list(_fold(hypothesis).replace(" ", ""))) / len(ref)


def entity_accuracy(truth: str, hypothesis: str, entities: tuple[str, ...] = ()) -> float | None:
    """Of the known entities the owner said, the fraction the STT got right.

    None when the truth contains no tracked entity — an average over Nones
    would be a synthetic number.
    """

    if not entities:
        from speech.normalize import BUILTIN_ENTITIES

        entities = BUILTIN_ENTITIES
    truth_f, hyp_f = _fold(truth), _fold(hypothesis)
    present = [e for e in entities if _fold(e).strip() and _fold(e).strip() in truth_f]
    if not present:
        return None
    hit = sum(1 for e in present if _fold(e).strip() in hyp_f)
    return hit / len(present)


class SpeechCorpus:
    """Recordings + verified transcripts under one directory, JSONL-indexed."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.audio_dir = self.root / "recordings"
        self.index = self.root / "corpus.jsonl"
        self._lock = threading.Lock()

    def add(self, audio: bytes, *, ext: str, ground_truth: str, category: str = "",
            device: str = "", conditions: str = "", held_out: bool = False) -> dict[str, Any]:
        if not audio:
            raise ValueError("no audio")
        if not str(ground_truth).strip():
            raise ValueError("the verified transcript is the whole point — it cannot be empty")
        ext = "." + str(ext).lstrip(".").lower() if ext else ".webm"
        entry_id = "sp_" + uuid.uuid4().hex[:10]
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        path = self.audio_dir / f"{entry_id}{ext}"
        path.write_bytes(audio)
        entry = {"id": entry_id, "audio": str(path), "ground_truth": str(ground_truth).strip(),
                 "category": category, "device": device, "conditions": conditions,
                 "held_out": bool(held_out), "bytes": len(audio),
                 "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with self._lock:
            with self.index.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def add_base64(self, payload: str, **kwargs: Any) -> dict[str, Any]:
        raw = base64.b64decode(str(payload or "").split(",")[-1].encode("ascii"), validate=False)
        return self.add(raw, **kwargs)

    def list(self) -> list[dict[str, Any]]:
        try:
            lines = self.index.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        removed = set()
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("removed"):
                removed.add(entry.get("id"))
            else:
                out.append(entry)
        return [e for e in out if e.get("id") not in removed]

    def delete(self, entry_id: str) -> bool:
        entries = {e["id"]: e for e in self.list()}
        if entry_id not in entries:
            return False
        try:
            Path(entries[entry_id]["audio"]).unlink(missing_ok=True)
        except OSError:
            pass
        with self._lock:
            with self.index.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": entry_id, "removed": True}) + "\n")
        return True

    def stats(self) -> dict[str, Any]:
        entries = self.list()
        by_cat: dict[str, int] = {}
        for e in entries:
            by_cat[e.get("category") or "—"] = by_cat.get(e.get("category") or "—", 0) + 1
        return {"count": len(entries), "held_out": sum(1 for e in entries if e.get("held_out")),
                "by_category": by_cat, "phrases_total": len(PHRASES)}
