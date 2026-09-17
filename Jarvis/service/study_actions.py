"""Study actions in the conversation: find it, show it, or answer from it -- never from nowhere.

The core hands every ``study.*`` action here.  Locating and listing need no model: the
index answers, the viewer opens at the location, and the message names document and
place.  Answering, summarizing, quizzing, explaining and comparing go through the normal
conversational path with exactly the owner's material as their source, and the answer
carries its sources.

The document context
--------------------
"This page", "the marked section", "davor", "darunter", "daneben", "noch einen Treffer"
refer to one structured document context -- the place the viewer shows or ZEUS last
found: document, source, page / slide / section / paragraph, selection, the phrases
marked, the query, zoom and viewer mode, and the list of places the last search found.
The viewer keeps it current; it is saved beside the study index, so it survives a
reload of the interface and a restart of ZEUS.

Direct open
-----------
"Zeig mir die Seite zu X" opens the dominant place at once.  Only when two places are
genuinely close (different documents, similar relevance) does ZEUS list them instead of
choosing for the owner.  A handwriting match that recognition is unsure about is named
"Möglicher Treffer" -- it opens, but never pretends certainty.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from service.events import EventType
from service.state import JarvisState

_KIND_TYPES = {
    "skript": {"pdf", "docx", "pptx"}, "folien": {"pptx", "pdf"}, "vorlesung": {"pdf", "pptx", "docx"},
    "notizen": {"markdown", "text", "docx", "pdf", "image"}, "mitschrift": {"markdown", "text", "docx", "pdf", "image"},
    "unterlagen": {"pdf", "docx", "pptx", "markdown", "text", "image"},
}
_WHEN_DAYS = {"heute": 1, "gestern": 2, "vorgestern": 3, "letzter woche": 14}
CONTEXT_FIELDS = ("document_id", "source_path", "source_type", "unit", "page", "slide", "section", "paragraph", "selection", "highlights", "focus",
                  "at", "last_query", "topic", "zoom", "mode", "results", "result_index", "found", "title", "source", "updated_at")
#: how long a found place stays "the place" for follow-ups
CONTEXT_TTL = 6 * 3600
#: a second place this close to the best one (relevance) makes the choice the owner's
DOMINANCE_GAP = 0.2


def compact(result: dict[str, Any]) -> dict[str, Any]:
    """A result as the conversation and the interface carry it."""

    doc = result["document"]
    best = result["best"]
    source = {"document_id": doc["id"], "title": doc.get("title") or "", "filename": doc.get("filename") or "", "source_type": doc.get("source_type"),
              "goodnotes": bool((doc.get("flags") or {}).get("goodnotes")), "subject": doc.get("subject") or "", "location": best["location"],
              "excerpt": best["excerpt"], "highlights": best["highlights"], "focus": best.get("focus") or "", "relevance": result.get("relevance", 1.0),
              "phrases": list(best.get("phrases") or [])[:6], "at": best.get("at"), "match": best.get("match") or "",
              "handwriting": bool(best.get("handwriting")), "recognised": bool(best.get("recognised")), "certainty": best.get("certainty") or "sure"}
    if best.get("figure"):
        source["figure"] = best["figure"]
    return source


def view_params(source: dict[str, Any]) -> dict[str, str]:
    """The Studium viewer's address for a source: document, unit, the phrase marked first, every other phrase, where the hit is."""

    location = source.get("location") or {}
    params = {"doc": str(source["document_id"]), "unit": str(location.get("unit") or 1)}
    if source.get("focus"):
        params["focus"] = str(source["focus"])[:200]
    others = [p for p in (source.get("phrases") or []) if p and p != source.get("focus")]
    if others:
        params["hl"] = "|".join(str(p)[:80] for p in others[:5])
    if source.get("at") is not None:
        params["at"] = str(int(source["at"]))
    if location.get("paragraph") is not None and location.get("page") is None and location.get("slide") is None:
        params["para"] = str(location["paragraph"])
    if (source.get("figure") or {}).get("number"):
        params["fig"] = str(source["figure"]["number"])
    return params


def dominant(results: list[dict[str, Any]]) -> bool:
    """One place clearly answers: the only one, or clearly ahead of the best place in another document."""

    if not results:
        return False
    best = results[0]
    others = [r for r in results[1:] if r["document_id"] != best["document_id"]]
    if not others:
        return True
    if best.get("match") == "phrase" and others[0].get("match") != "phrase":
        return True
    return float(best.get("relevance") or 1.0) - float(others[0].get("relevance") or 0.0) >= DOMINANCE_GAP


def place_word(source: dict[str, Any]) -> str:
    """"Handschriftlicher Treffer" / "Möglicher handschriftlicher Treffer" / "Möglicher Treffer" -- or nothing for printed text."""

    if source.get("certainty") == "possible":
        return "Möglicher handschriftlicher Treffer" if source.get("handwriting") else "Möglicher Treffer"
    if source.get("handwriting"):
        return "Handschriftlicher Treffer"
    return ""


class StudyActions:
    def __init__(self, core: Any) -> None:
        self.core = core
        self.focus: dict[str, Any] = {}
        self._loaded = False

    # -- the document context ------------------------------------------------------------

    @property
    def _context_path(self) -> Path:
        return Path(self.core.study.data_dir) / "context.json"

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            data = json.loads(self._context_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.focus = data
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        try:
            self._context_path.parent.mkdir(parents=True, exist_ok=True)
            self._context_path.write_text(json.dumps({k: self.focus.get(k) for k in CONTEXT_FIELDS if k in self.focus}, ensure_ascii=False),
                                          encoding="utf-8")
        except OSError:
            pass

    def context(self) -> dict[str, Any]:
        self._load()
        return dict(self.focus)

    def has_context(self) -> bool:
        self._load()
        if not self.focus.get("document_id"):
            return False
        if time.time() - float(self.focus.get("updated_at") or 0) > CONTEXT_TTL:
            return False
        return self.core.study.index.document(self.focus["document_id"]) is not None

    def set_focus(self, document_id: str, unit: int, *, selection: str = "", topic: str = "", source: str = "viewer", **viewer: Any) -> dict[str, Any]:
        """The place is now this one.  The viewer also reports zoom, mode, marked phrases, paragraph and the query."""

        self._load()
        doc = self.core.study.index.document(document_id)
        if doc is None:
            return {"ok": False, "error": "Dokument nicht gefunden"}
        unit = int(unit or 1)
        record = self.core.study.index.unit(document_id, unit) or {}
        kind = record.get("kind") or ""
        same_doc = self.focus.get("document_id") == document_id
        focus = {**(self.focus if same_doc else {}), "document_id": document_id, "unit": unit, "source_path": doc.get("stored_path"),
                 "source_type": doc.get("source_type"), "title": doc.get("title"), "source": source, "updated_at": time.time(),
                 "page": unit if kind == "page" else None, "slide": unit if kind == "slide" else None,
                 "section": record.get("title") if kind == "section" else None, "selection": str(selection or "")[:6000]}
        if topic:
            focus["topic"] = topic
        elif not same_doc:
            focus["topic"] = self.focus.get("topic", "")
        for key in ("zoom", "mode", "paragraph", "highlights", "focus", "at", "last_query"):
            if viewer.get(key) not in (None, ""):
                focus[key] = viewer[key]
        self.focus = focus
        self._save()
        return {"ok": True, "focus": {k: v for k, v in self.focus.items() if k != "results"}}

    def _remember_results(self, results: list[dict[str, Any]], query: str, topic: str, index: int = 0) -> None:
        self._load()
        self.focus = {**self.focus, "results": results[:12], "result_index": index, "last_query": query, "topic": topic or self.focus.get("topic", ""),
                      "updated_at": time.time()}
        self._save()

    def _focus_label(self) -> str:
        self._load()
        doc = self.core.study.index.document(self.focus["document_id"]) if self.focus.get("document_id") else None
        unit = self.core.study.index.unit(self.focus["document_id"], int(self.focus.get("unit") or 1)) if doc else None
        if not doc or not unit:
            return ""
        kind = unit["kind"]
        where = f"Seite {unit['number']}" if kind == "page" else f"Folie {unit['number']}" if kind == "slide" else (unit.get("title") or f"Abschnitt {unit['number']}")
        return f"{doc['filename'] or doc['title']}, {where}"

    # -- running -----------------------------------------------------------------------

    def run(self, action: Any, text: str, scope: str) -> None:
        self._load()
        operation = action.operation
        handler = {
            "study.locate": self._locate, "study.search": self._search, "study.which_page": self._which_page, "study.figure": self._figure,
            "study.open_recent": self._open_recent, "study.answer": self._answer, "study.summarize": self._summarize, "study.quiz": self._quiz,
            "study.explain": self._explain, "study.compare": self._compare, "study.missing": self._missing,
            "study.page": self._page, "study.return": self._return, "study.view": self._view, "study.next_hit": self._next_hit,
            "study.below": self._below, "study.beside": self._beside,
        }.get(operation)
        self.core.emit(EventType.TOOL, {"summary": f"study: {operation}" + (f" '{action.target}'" if action.target else ""), "source": "study",
                                        "study_action": action.to_dict()}, scope=scope)
        if handler is None:
            self._say(f"Das kann Studium noch nicht: {operation}.", scope)
            return
        handler(action, text, scope)

    def run_in_context(self, text: str, context: dict[str, Any], scope: str) -> None:
        """A question asked from the viewer ("Ask ZEUS"): the visible page or the selection is the only direct source."""

        focused = self.set_focus(str(context.get("document_id") or ""), int(context.get("unit") or 1),
                                 selection=str(context.get("selection") or ""), source="ask")
        if not focused.get("ok"):
            self._say("Diese Stelle finde ich nicht mehr in deinen Unterlagen.", scope, final=JarvisState.WAITING)
            return
        from study.intents import parse_study_operation

        action = parse_study_operation(text, has_material=True, has_context=True)
        if action is not None and action.operation in {"study.explain", "study.summarize", "study.quiz", "study.beside", "study.below"}:
            self.run(action, text, scope)
            return
        instruction = ("Beantworte die Frage des Besitzers ausschließlich anhand des folgenden Auszugs aus seinen Unterlagen. "
                       "Wenn der Auszug die Antwort nicht enthält, sag das klar.")
        other = re.search(r"(?i)\b(?:mit|und|zu)\s+(?:der\s+)?(seite|folie)\s+(\d{1,4})\b", text)
        if other:
            # "Vergleiche das mit Seite 12": the open place and exactly that other page of the same document -- nothing else
            self._from_focus(text, scope, instruction=instruction.replace("des folgenden Auszugs", "der beiden folgenden Auszüge"),
                             extra_unit=int(other.group(2)))
            return
        self._from_focus(text, scope, instruction=instruction, with_figures=bool(re.search(r"(?i)\b(abbildung|grafik|schema|diagramm|bild)\b", text)))

    # -- speaking ----------------------------------------------------------------------

    def _say(self, sentence: str, scope: str, *, sources: list[dict[str, Any]] | None = None, final: JarvisState = JarvisState.IDLE,
             context: str = "") -> None:
        meta: dict[str, Any] = {}
        request_id = self.core._last_user_meta().get("request_id")
        if request_id:
            meta["request_id"] = request_id
        if sources:
            meta["study_sources"] = sources
        self.core._deliver(sentence, scope=scope, backend="study", final_state=final, meta=meta, context_text=context or f"[study] {sentence[:160]}")

    def _open(self, source: dict[str, Any], scope: str) -> None:
        location = source["location"]
        params = view_params(source)
        self.core.emit(EventType.NOTIFICATION, {"kind": "open_view", "view": "study", "params": params, "text": ""}, scope=scope)
        self.set_focus(source["document_id"], int(location.get("unit") or 1), topic=self.focus.get("topic") or source.get("focus") or "",
                       source="locate", focus=source.get("focus") or "", at=source.get("at"), highlights=source.get("phrases") or [],
                       paragraph=location.get("paragraph"))
        self.focus["found"] = {"source": source, "params": params}
        self._save()

    def _open_params(self, params: dict[str, str], scope: str) -> None:
        self.core.emit(EventType.NOTIFICATION, {"kind": "open_view", "view": "study", "params": params, "text": ""}, scope=scope)

    def _empty_library(self, scope: str) -> bool:
        if self.core.study.index.stats()["documents"]:
            return False
        self._say("Im Studium liegt noch kein Material. Füge unter Studium › Material hinzufügen deine Skripte und Notizen hinzu – "
                  "dann finde ich jede Stelle darin.", scope, final=JarvisState.WAITING)
        return True

    def _results(self, query: str, scope: str, *, limit: int = 8, document_id: str = "", filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.core.state.set(JarvisState.RESEARCHING, detail="durchsucht deine Unterlagen", scope=scope)
        found = self.core.study.search(query, limit=limit, document_id=document_id, filters=filters)
        return [compact(r) for r in found.get("results") or []]

    def _query_for(self, action: Any, text: str) -> tuple[str, str, dict[str, Any]]:
        """The search words, the topic, and the filters an action implies (GoodNotes, handwriting; the open topic for "dazu")."""

        arguments = action.arguments or {}
        topic = action.target or ""
        query = arguments.get("query") or text
        if arguments.get("from_context"):
            topic = self.focus.get("topic") or ""
            query = topic
        filters: dict[str, Any] = {}
        if arguments.get("goodnotes"):
            filters["goodnotes"] = True
        if arguments.get("handwriting"):
            filters["handwriting"] = True
        if arguments.get("figure") and topic and arguments.get("from_context"):
            query = f"Abbildung zu {topic}"
        return query, topic, filters

    # -- finding -------------------------------------------------------------------------

    def _locate(self, action: Any, text: str, scope: str) -> None:
        if self._empty_library(scope):
            return
        query, topic, filters = self._query_for(action, text)
        if not topic:
            self._say("Zu welchem Thema soll ich suchen?", scope, final=JarvisState.WAITING)
            return
        results = self._results(query, scope, limit=6, filters=filters or None)
        if not results and filters:
            # "meine GoodNotes-Notizen dazu" with no GoodNotes material: say so, then look everywhere
            wider = self._results(query, scope, limit=6)
            where = "in deinen GoodNotes-Unterlagen" if filters.get("goodnotes") else "handschriftlich"
            if not wider:
                self._say(f"Zu „{topic}“ finde ich {where} nichts – und auch sonst keine Stelle.", scope, final=JarvisState.WAITING)
                return
            self._remember_results(wider, query, topic)
            self._say(f"Zu „{topic}“ finde ich {where} nichts. In deinen anderen Unterlagen: {self._name(wider[0])}.", scope, sources=wider[:4],
                      final=JarvisState.WAITING)
            return
        if not results:
            self._say(f"Zu „{topic}“ finde ich in deinen Unterlagen keine Stelle.", scope, final=JarvisState.WAITING)
            return
        self._remember_results(results, query, topic)
        best = results[0]
        if not dominant(results):
            names = " oder ".join(self._name(r) for r in results[:2])
            self._say(f"Zu „{topic}“ passen mehrere Stellen gleich gut: {names}. Welche soll ich öffnen?", scope, sources=results[:4],
                      final=JarvisState.WAITING)
            return
        self.focus["topic"] = topic
        self._open(best, scope)
        self._say(f"{self._lead(best)}{self._name(best)}.", scope, sources=results[:3])

    @staticmethod
    def _name(source: dict[str, Any]) -> str:
        return f"{source.get('filename') or source.get('title')} · {source['location']['label']}"

    @staticmethod
    def _lead(source: dict[str, Any]) -> str:
        word = place_word(source)
        if word:
            return f"{word}: "
        # found by meaning, not by the words: say so, the owner judges whether it is the place
        return "Wörtlich steht es nirgends; inhaltlich am nächsten: " if source.get("match") == "semantic" else ""

    def _search(self, action: Any, text: str, scope: str) -> None:
        if self._empty_library(scope):
            return
        query, topic, filters = self._query_for(action, text)
        if not topic:
            self._say("Wonach soll ich in deinen Unterlagen suchen?", scope, final=JarvisState.WAITING)
            return
        results = self._results(query, scope, limit=8, filters=filters or None)
        if not results:
            where = " handschriftlich" if filters.get("handwriting") else ""
            self._say(f"Zu „{topic}“ finde ich{where} nichts in deinen Unterlagen.", scope, final=JarvisState.WAITING)
            return
        self._remember_results(results, query, topic)
        documents = len({r["document_id"] for r in results})
        if len(results) == 1 or (filters.get("handwriting") and dominant(results)):
            # one place, or one clear handwritten place: open it -- the owner asked where it is
            self.focus["topic"] = topic
            self._open(results[0], scope)
            self._say(f"{self._lead(results[0])}{self._name(results[0])}.", scope, sources=results[:3])
            return
        lead = f"{len(results)} Stellen in {documents} Dokumenten zu „{topic}“."
        self._say(lead, scope, sources=results[:6])

    def _which_page(self, action: Any, text: str, scope: str) -> None:
        label = self._focus_label()
        if not label:
            self._say("Ich habe gerade keine Stelle aus deinen Unterlagen offen.", scope, final=JarvisState.WAITING)
            return
        self._say(f"Das war {label}.", scope)

    def _figure(self, action: Any, text: str, scope: str) -> None:
        topic = self.focus.get("topic") or ""
        if not topic:
            self._say("Zu welchem Thema soll ich die Abbildung suchen?", scope, final=JarvisState.WAITING)
            return
        results = self._results(f"Abbildung zu {topic}", scope, limit=5)
        results = sorted(results, key=lambda r: 0 if r.get("figure") else 1)
        if not results:
            self._say(f"Eine Abbildung zu „{topic}“ finde ich in deinen Unterlagen nicht.", scope, final=JarvisState.WAITING)
            return
        self._remember_results(results, f"Abbildung zu {topic}", topic)
        self._open(results[0], scope)
        caption = (results[0].get("figure") or {}).get("caption") or ""
        self._say(f"{self._name(results[0])}" + (f" – {caption}" if caption else "") + ".", scope, sources=results[:3])

    def _open_recent(self, action: Any, text: str, scope: str) -> None:
        if self._empty_library(scope):
            return
        days = _WHEN_DAYS.get(str(action.arguments.get("when") or "gestern"), 2)
        types = _KIND_TYPES.get(action.target, set())
        cutoff = time.time() - days * 86400
        candidates = []
        for doc in self.core.study.documents(limit=400):
            stamp = max(float(doc.get("opened_at") or 0), float(doc.get("updated_at") or 0), float(doc.get("indexed_at") or 0))
            if stamp >= cutoff and (not types or doc["source_type"] in types):
                candidates.append((stamp, doc))
        if not candidates:
            self._say("Aus diesem Zeitraum finde ich kein Material.", scope, final=JarvisState.WAITING)
            return
        candidates.sort(key=lambda item: item[0], reverse=True)
        doc = candidates[0][1]
        source = {"document_id": doc["id"], "title": doc["title"], "filename": doc["filename"], "source_type": doc["source_type"],
                  "goodnotes": bool((doc.get("flags") or {}).get("goodnotes")), "subject": doc.get("subject") or "",
                  "location": {"unit": 1, "page": 1 if doc["source_type"] == "pdf" else None, "label": "Anfang"}, "excerpt": "", "highlights": [],
                  "focus": "", "relevance": 1.0}
        self._open(source, scope)
        self._say(f"{doc['filename'] or doc['title']} ist offen.", scope, sources=[source])

    # -- navigating the open document -------------------------------------------------------

    def _require_context(self, scope: str) -> bool:
        if self.has_context():
            return True
        self._say("Ich habe gerade keine Stelle aus deinen Unterlagen offen. Frag nach einem Thema, dann öffne ich die Seite.", scope,
                  final=JarvisState.WAITING)
        return False

    def _page(self, action: Any, text: str, scope: str) -> None:
        if not self._require_context(scope):
            return
        doc = self.core.study.index.document(self.focus["document_id"])
        total = int(doc.get("units") or 1)
        current = int(self.focus.get("unit") or 1)
        target = int(action.arguments["number"]) if action.arguments.get("number") else current + int(action.arguments.get("step") or 0)
        word = "Folie" if doc.get("source_type") == "pptx" else "Seite" if doc.get("source_type") in {"pdf", "image"} else "Abschnitt"
        if target < 1 or target > total:
            edge = "die erste" if target < 1 else "die letzte"
            self._say(f"Das ist schon {edge} {word} ({current} von {total}).", scope, final=JarvisState.WAITING)
            return
        self._open_params({"doc": doc["id"], "unit": str(target)}, scope)
        self.set_focus(doc["id"], target, source="navigate")
        self._say(f"{word} {target} von {total}.", scope)

    def _return(self, action: Any, text: str, scope: str) -> None:
        if not self._require_context(scope):
            return
        found = self.focus.get("found") or {}
        if not found.get("params"):
            label = self._focus_label()
            self._say(f"Offen ist {label}." if label else "Ich habe keine gefundene Stelle, zu der ich zurück kann.", scope)
            return
        source = found["source"]
        self._open_params(found["params"], scope)
        self.set_focus(source["document_id"], int(source["location"].get("unit") or 1), source="return", focus=source.get("focus") or "",
                       at=source.get("at"), highlights=source.get("phrases") or [])
        self._say(f"Zurück zur Stelle: {self._name(source)}.", scope, sources=[source])

    def _view(self, action: Any, text: str, scope: str) -> None:
        if not self._require_context(scope):
            return
        command = {"fit": True} if action.arguments.get("fit") else {"zoom": float(action.arguments.get("zoom") or 0.25)}
        params = {"doc": self.focus["document_id"], "unit": str(self.focus.get("unit") or 1)}
        for key in ("focus", "at"):
            if self.focus.get(key) not in (None, ""):
                params[key] = str(self.focus[key])
        level = None if command.get("fit") else max(0.5, min(3.0, round(float(self.focus.get("zoom") or 1.0) + command["zoom"], 2)))
        # an open viewer applies the command; with the viewer closed (typed in the chat) the page opens again at that zoom
        self.core.emit(EventType.NOTIFICATION, {"kind": "study_view", "params": params, "command": command, "level": level, "text": ""}, scope=scope)
        self.focus["zoom"] = level
        self._save()
        self._say("Ganze Seite." if command.get("fit") else ("Größer." if command["zoom"] > 0 else "Kleiner."), scope)

    def _next_hit(self, action: Any, text: str, scope: str) -> None:
        if not self._require_context(scope):
            return
        results = list(self.focus.get("results") or [])
        index = int(self.focus.get("result_index") or 0) + 1
        if not results and self.focus.get("topic"):
            results = self._results(self.focus["topic"], scope, limit=8)
        # the same document's other places count as hits too
        extra = self.core.study.search(self.focus.get("last_query") or self.focus.get("topic") or "", limit=1,
                                       document_id=self.focus["document_id"]) if self.focus.get("topic") else {}
        for result in (extra.get("results") or [])[:1]:
            for hit in result.get("more") or []:
                candidate = compact({"document": result["document"], "best": hit, "relevance": 0.5})
                if all((r["document_id"], r["location"].get("unit")) != (candidate["document_id"], candidate["location"].get("unit")) for r in results):
                    results.append(candidate)
        if index >= len(results):
            self._say("Weitere Treffer finde ich dazu nicht.", scope, final=JarvisState.WAITING)
            return
        source = results[index]
        self._open(source, scope)
        self.focus["results"] = results[:12]
        self.focus["result_index"] = index
        self._save()
        self._say(f"Treffer {index + 1} von {len(results)}: {self._lead(source)}{self._name(source)}.", scope, sources=[source])

    def _below(self, action: Any, text: str, scope: str) -> None:
        if not self._require_context(scope):
            return
        unit = self.core.study.index.unit(self.focus["document_id"], int(self.focus.get("unit") or 1)) or {}
        body = unit.get("text") or ""
        at = self.focus.get("at")
        focus = str(self.focus.get("focus") or "")
        if at is None and focus and focus in body:
            at = body.index(focus)
        if at is None or not body:
            self._say("Ich weiß nicht, unter welcher Stelle ich lesen soll – frag zuerst nach einem Thema.", scope, final=JarvisState.WAITING)
            return
        start = int(at) + len(focus)
        # from the end of the found line: the next sentences, as they stand on the page
        line_end = body.find("\n", start)
        following = body[line_end + 1 if line_end >= 0 else start:].strip()
        sentences = re.findall(r"[^.!?\n]+[.!?]?", following)
        quote = " ".join(s.strip() for s in sentences[:3] if s.strip())[:500]
        if not quote:
            self._say("Darunter steht auf dieser Seite nichts mehr.", scope)
            return
        self._say(f"Darunter steht: „{quote}“", scope, sources=[self._focus_source_now()])

    def _beside(self, action: Any, text: str, scope: str) -> None:
        """Handwriting on the same page, nearest to the found place: quoted as recognised, marked as uncertain when it is."""

        if not self._require_context(scope):
            return
        study = self.core.study
        doc_id = self.focus["document_id"]
        number = int(self.focus.get("unit") or 1)
        unit = study.index.unit(doc_id, number) or {}
        blocks = [b for b in unit.get("blocks") or [] if b.get("box") and (b.get("kind") == "handwriting" or b.get("role") in {"annotation", "label"})]
        if not blocks:
            self._say("Neben dieser Stelle erkenne ich keine handschriftliche Notiz.", scope)
            return
        anchor_y = 0.5
        focus = str(self.focus.get("focus") or "")
        if focus:
            located = study.locate(doc_id, number, [focus], at=self.focus.get("at"))
            primary = next((m for m in located.get("matches") or [] if m.get("primary")), None)
            if primary and primary["rects"]:
                x, y, w, h = primary["rects"][0]
                anchor_y = y + h / 2
        nearest = sorted(blocks, key=lambda b: abs((b["box"][1] + b["box"][3] / 2) - anchor_y))[:2]
        quotes = " / ".join(str(b.get("text") or "").strip() for b in nearest if str(b.get("text") or "").strip())
        unsure = any(b.get("confidence") is not None and float(b["confidence"]) < 0.6 for b in nearest)
        lead = "Daneben steht handschriftlich (unsicher erkannt)" if unsure else "Daneben steht handschriftlich"
        self._say(f"{lead}: „{quotes[:400]}“", scope, sources=[self._focus_source_now()])

    def _focus_source_now(self) -> dict[str, Any]:
        doc = self.core.study.index.document(self.focus["document_id"]) or {}
        from study.model import Location

        unit = self.core.study.index.unit(self.focus["document_id"], int(self.focus.get("unit") or 1)) or {}
        kind = unit.get("kind")
        location = Location(page=unit.get("number") if kind == "page" else None, slide=unit.get("number") if kind == "slide" else None,
                            heading_path=[h for h in (unit.get("title") or "").split(" › ") if h] if kind == "section" else [], unit=int(unit.get("number") or 1))
        return {"document_id": doc.get("id"), "title": doc.get("title") or "", "filename": doc.get("filename") or "", "source_type": doc.get("source_type"),
                "goodnotes": bool((doc.get("flags") or {}).get("goodnotes")), "subject": doc.get("subject") or "", "location": location.to_dict(),
                "excerpt": "", "highlights": [], "focus": self.focus.get("focus") or "", "relevance": 1.0}

    # -- answering from material -----------------------------------------------------------

    def _material(self, results: list[dict[str, Any]], *, max_chars: int = 9000) -> str:
        study = self.core.study
        parts: list[str] = []
        used = 0
        for result in results:
            location = result["location"]
            unit = study.index.unit(result["document_id"], int(location.get("unit") or 1))
            if unit is None:
                continue
            start = max(0, int(location.get("char_start") or 0) - 300)
            end = min(len(unit["text"]), int(location.get("char_end") or 0) + 300)
            passage = unit["text"][start:end].strip()
            block = f"[{result['filename'] or result['title']} · {location['label']}]\n{passage}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def _conversational(self, text: str, scope: str, *, instruction: str, material: str, sources: list[dict[str, Any]]) -> None:
        self.core._answer_conversationally(text, scope, study={"instruction": instruction, "material": material, "sources": sources})

    def _answer(self, action: Any, text: str, scope: str) -> None:
        if self._empty_library(scope):
            return
        results = self._results(action.arguments.get("query") or text, scope, limit=6)
        if not results:
            self._say(f"Zu „{action.target}“ steht nichts in deinen Unterlagen.", scope, final=JarvisState.WAITING)
            return
        self.focus = {**self.focus, "topic": action.target}
        self._conversational(text, scope, sources=results[:4], material=self._material(results),
                             instruction=("Antworte ausschließlich anhand der folgenden Auszüge aus den Unterlagen des Besitzers. "
                                          "Nenne bei jeder Aussage kurz Dokument und Stelle in Klammern. Erfinde nichts dazu."))

    def _focus_context(self, scope: str) -> dict[str, Any] | None:
        self._load()
        if not self.focus.get("document_id"):
            self._say("Öffne zuerst eine Seite im Studium oder frag nach einer Stelle – dann weiß ich, welchen Abschnitt du meinst.", scope,
                      final=JarvisState.WAITING)
            return None
        context = self.core.study.context(self.focus["document_id"], int(self.focus.get("unit") or 1), selection=self.focus.get("selection") or "")
        if not context.get("ok") or not context.get("has_text"):
            self._say("Auf dieser Stelle ist kein lesbarer Text – bei handschriftlichen Seiten brauche ich dafür eine Texterkennung.", scope,
                      final=JarvisState.WAITING)
            return None
        return context

    def _focus_source(self, context: dict[str, Any]) -> dict[str, Any]:
        doc = context["document"]
        return {"document_id": doc["id"], "title": doc["title"], "filename": doc["filename"], "source_type": doc["source_type"], "goodnotes": False,
                "subject": "", "location": context["location"], "excerpt": context["text"][:220], "highlights": [], "focus": "", "relevance": 1.0}

    def _from_focus(self, text: str, scope: str, *, instruction: str, extra_unit: int | None = None, with_figures: bool = False) -> None:
        context = self._focus_context(scope)
        if context is None:
            return
        label = f"{context['document']['filename'] or context['document']['title']} · {context['location']['label']}"
        scope_word = "markierter Abschnitt" if context["scope"] == "selection" else "Stelle"
        material = f"[{label} – {scope_word}]\n{context['text']}"
        sources = [self._focus_source(context)]
        if with_figures:
            figures = self.core.study.index.figures(context["document"]["id"], int(self.focus.get("unit") or 1))
            if figures:
                material += "\n\n[ABBILDUNGEN AUF DIESER STELLE]\n" + "\n".join(f"- {f['caption'] or f['kind']}: {f['text'][:300]}" for f in figures)
        if extra_unit is not None:
            other = self.core.study.context(context["document"]["id"], extra_unit)
            if other.get("ok") and other.get("has_text"):
                material += f"\n\n[{context['document']['filename'] or context['document']['title']} · {other['location']['label']} – Vergleichsstelle]\n{other['text']}"
                sources.append(self._focus_source(other))
        self._conversational(text, scope, instruction=instruction, material=material, sources=sources)

    def _summarize(self, action: Any, text: str, scope: str) -> None:
        self._from_focus(text, scope, instruction="Fasse ausschließlich den folgenden Auszug aus den Unterlagen des Besitzers zusammen: knapp, strukturiert, prüfungsrelevant.")

    def _explain(self, action: Any, text: str, scope: str) -> None:
        self._from_focus(text, scope, instruction=("Erkläre ausschließlich den folgenden Abschnitt aus den Unterlagen des Besitzers. "
                                                   "Bleib bei genau diesem Abschnitt; greife nicht auf andere Kapitel vor."))

    def _quiz(self, action: Any, text: str, scope: str) -> None:
        count = action.arguments.get("count") or (10 if action.arguments.get("multiple_choice") else 5)
        difficulty = action.arguments.get("difficulty") or "prüfungsnah"
        style = "Multiple-Choice-Fragen mit je fünf Antwortmöglichkeiten (A–E), genau eine richtig" if action.arguments.get("multiple_choice") else "Prüfungsfragen"
        self._from_focus(text, scope, instruction=(f"Erstelle {count} {difficulty}e {style} ausschließlich aus dem folgenden Auszug. "
                                                   "Gib die Lösungen erst am Ende gesammelt an, jeweils mit kurzer Begründung."))

    def _compare(self, action: Any, text: str, scope: str) -> None:
        context = self._focus_context(scope)
        if context is None:
            return
        topic = self.focus.get("topic") or context["location"].get("label") or ""
        probe = topic if topic and not topic.startswith(("Seite", "Folie", "Abschnitt")) else context["text"][:300]
        others = [r for r in self._results(probe, scope, limit=8) if r["document_id"] != context["document"]["id"]][:4]
        if not others:
            self._say("Zu dieser Stelle finde ich keine anderen Notizen zum Vergleichen.", scope, final=JarvisState.WAITING)
            return
        label = f"{context['document']['filename']} · {context['location']['label']}"
        material = f"[AUSGANGSSTELLE: {label}]\n{context['text']}\n\n[ANDERE UNTERLAGEN]\n{self._material(others, max_chars=6000)}"
        self._conversational(text, scope, sources=[self._focus_source(context)] + others, material=material,
                             instruction=("Vergleiche die Ausgangsstelle mit den anderen Unterlagen des Besitzers: Übereinstimmungen, Widersprüche, "
                                          "was nur in einer Quelle steht. Nenne jeweils Dokument und Stelle. Nutze nur diese Auszüge."))

    def _missing(self, action: Any, text: str, scope: str) -> None:
        context = self._focus_context(scope)
        if context is None:
            return
        doc_id = context["document"]["id"]
        headings = [u.get("title") for u in self.core.study.index.units(doc_id) if u.get("title")][:60]
        probe = " ".join(headings[:6]) or context["text"][:300]
        notes = [r for r in self._results(probe, scope, limit=8) if r["document_id"] != doc_id][:4]
        material = (f"[SKRIPT: {context['document']['filename']} – Gliederung]\n" + "\n".join(f"- {h}" for h in headings)
                    + f"\n\n[AKTUELLE STELLE]\n{context['text'][:3000]}\n\n[NOTIZEN DES BESITZERS]\n{self._material(notes, max_chars=5000) or '(keine passenden Notizen gefunden)'}")
        self._conversational(text, scope, sources=[self._focus_source(context)] + notes, material=material,
                             instruction=("Liste die Themen aus dem Skript auf, die in den Notizen des Besitzers fehlen oder nur angerissen sind. "
                                          "Stütze dich nur auf diese Auszüge und sag, wenn die Notizen zu wenig Material für ein sicheres Urteil bieten."))
