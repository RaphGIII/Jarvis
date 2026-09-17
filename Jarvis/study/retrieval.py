"""Hybrid retrieval over the study index: what the owner asked about, where it is.

1. The query is read: the topic is separated from the command around it ("Zeig mir die
   Seite zum Frank-Starling-Mechanismus" -> "Frank-Starling-Mechanismus"), and intent
   hints are kept (a figure was asked for; "von gestern" limits by date; "in meinen
   Folien" prefers slides).
2. Candidates come from up to four lists: the word index (BM25; exact names, prefixes),
   the trigram index (substrings; German compounds, hyphenation), optional expansion
   terms (synonyms from the reasoning pool), and the semantic neighbours of the question
   (local embeddings, computed by the service -- this module stays pure).
3. Every candidate is scored on evidence, not on rank alone: the exact phrase (hyphens,
   spaces and case ignored), the share of the topic's words present, the heading, a
   file name or title that names the topic, the figure hint, the type hint -- plus the
   rank fusion of the lexical lists and the semantic closeness.  Exact names and
   phrases stay strongest; meaning lifts what the words alone rank too low and finds
   what is said in other words.  A place found only by meaning must be clearly closer
   than the rest of the library, or it is not a source.
4. Metadata filters (subject, course, semester, module, topic, type, provider) restrict
   the candidates before scoring.
5. Results are grouped by document and ranked by relevance, never by file type; each
   carries its location, an excerpt with highlight spans, the phrase the viewer marks
   first and every other phrase worth marking on the same page.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from study.index import StudyIndex
from study.model import Location

_STOP_WORDS = {
    # command words around the topic
    "zeig", "zeige", "mir", "öffne", "öffnen", "such", "suche", "finde", "find", "wo", "habe", "hab", "hatte", "ich", "etwas", "was", "infos",
    "information", "informationen", "seite", "seiten", "stelle", "abschnitt", "folie", "folien", "notiz", "notizen", "unterlagen", "material",
    "skript", "script", "vorlesung", "vorlesungen", "dokument", "datei", "stand", "nochmal", "noch", "mal", "genau", "bitte", "meine", "meinen",
    "meinem", "meiner", "mein", "notiert", "geschrieben", "aufgeschrieben", "welche", "welcher", "welches", "war", "das", "gibt", "es",
    "abbildung", "grafik", "mitschrift",
    # question and function words
    "warum", "wieso", "weshalb", "wodurch", "womit", "wofür", "wann", "wer", "wen", "wem", "wenn", "dass", "ob", "ist", "sind", "wird", "werden",
    "wurde", "hat", "haben", "kann", "können", "muss", "soll", "mehr", "weniger", "sehr", "auch", "nur", "nicht", "kein", "keine", "man", "sich",
    "er", "sie", "wir", "ihr", "du", "dir", "dich", "uns", "so", "als", "bis", "durch", "gegen", "ohne", "um", "unter", "vor", "zwischen", "erklär",
    "erkläre", "erklärt", "beschreibe", "nenne", "bedeutet", "heißt",
    # articles and prepositions
    "der", "die", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "zu", "zum", "zur", "über", "von", "vom", "mit", "im",
    "in", "an", "am", "auf", "und", "oder", "für", "bei", "aus", "nach", "wie", "the", "of", "on", "to", "about", "show", "me", "where",
    "page", "notes", "open", "my", "a", "an", "and", "for", "is", "what", "why", "how",
}
_PAGE_Q = re.compile(r"(?i)\b(seite|page|folie|slide)\b")
_FIGURE_Q = re.compile(r"(?i)\b(abbildung|abb\.?|grafik|schaubild|schema|diagramm|figure|bild|zeichnung|skizze)\b")
_WHEN = {"heute": 1, "gestern": 2, "vorgestern": 3, "diese woche": 7, "letzte woche": 14, "letzten woche": 14}
#: "in meinen Folien" -> slides first.  Soft: a better place in another format still wins.
_TYPE_HINTS: list[tuple[re.Pattern[str], set[str], bool]] = [
    (re.compile(r"(?i)\b(folien|folie|präsentation|slides?|powerpoint)\b"), {"pptx"}, False),
    (re.compile(r"(?i)\b(notizen|notiz|mitschrift|goodnotes|handschrift\w*)\b"), {"markdown", "text", "image"}, True),
    (re.compile(r"(?i)\b(skript|script)\b"), {"pdf", "docx"}, False),
    (re.compile(r"(?i)\bword(?:-?dokument)?\b"), {"docx"}, False),
]

_TOPIC_PATTERNS = [
    re.compile(r"(?i)\b(?:seite|stelle|abschnitt|folie|abbildung|grafik|notizen?|skript|unterlagen|material|kapitel|zusammenfassung)"
               r"(?:\s+mit\s+der\s+abbildung)?\s+(?:zu[mr]?|über|ueber|zu\s+den|mit)\s+(?P<t>.+)$"),
    re.compile(r"(?i)\bwo\s+(?:habe|hab|hatte)\s+ich\s+(?:etwas|was|infos?|notizen)?\s*(?:zu[mr]?|über|ueber)\s+(?P<t>.+)$"),
    re.compile(r"(?i)\bwas\s+(?:habe|hab)\s+ich\s+(?:zu[mr]?|über|ueber)\s+(?P<t>.+?)(?:\s+(?:notiert|geschrieben|aufgeschrieben|gelernt))?\s*$"),
    re.compile(r"(?i)\bwo\s+stand\s+(?:nochmal\s+|noch\s+mal\s+)?(?:etwas\s+)?(?:zu[mr]?\s+|über\s+)?(?P<t>.+)$"),
    re.compile(r"(?i)\bwo\s+(?:steht|stehen|finde\s+ich)\s+(?:in\s+meine[mnr]?\s+\w+\s+)?(?:etwas\s+|was\s+|infos?\s+)?(?:zu[mr]?|über|ueber)\s+(?P<t>.+)$"),
    re.compile(r"(?i)^(?:zeig(?:e)?\s+mir|öffne|oeffne|such(?:e)?|finde?)\s+(?:bitte\s+)?(?:(?:die|den|das|meine)\s+)?(?P<t>.+)$"),
]

#: Semantic gate, calibrated on multilingual-e5 cosine similarities (compressed around 0.75-0.87): a place found only
#: by meaning must be close in absolute terms AND clearly closer than the library's median.  Off-topic questions
#: ("Rezept für Apfelkuchen") reach 0.78 at best with a margin of 0.03; paraphrases of real content 0.83+ / 0.05+.
SEMANTIC_MIN_SIMILARITY = 0.83
SEMANTIC_MIN_MARGIN = 0.045
SEMANTIC_WEIGHT = 1.4
SEMANTIC_TOP_K = 40
#: A place that shares only some of the question's words ("Rezept" in "Rezeptoren") must also be near in meaning:
#: below this similarity and without a margin over the median it is an accident of spelling, not a source.
SEMANTIC_VETO_BELOW = 0.80


def fold(text: str) -> str:
    """Lower case, diacritics removed, every run of non-alphanumerics one space."""

    decomposed = unicodedata.normalize("NFKD", text.lower().replace("ß", "ss"))
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]+", " ", plain).strip()


_STOP = {fold(word) for word in _STOP_WORDS} | {"oeffne", "ueber", "fuer"}


def fold_with_index(text: str) -> tuple[str, list[int]]:
    """``fold`` plus, for every folded character, the index of the original character it came from."""

    out: list[str] = []
    index: list[int] = []
    space = True
    for i, ch in enumerate(text):
        piece = unicodedata.normalize("NFKD", ch.lower().replace("ß", "ss"))
        piece = "".join(c for c in piece if not unicodedata.combining(c))
        for c in piece:
            if c.isalnum() and c.isascii():
                out.append(c)
                index.append(i)
                space = False
            elif not space:
                out.append(" ")
                index.append(i)
                space = True
    while out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


@dataclass
class Query:
    raw: str
    topic: str
    terms: list[str]
    figure: bool = False
    page: bool = False
    since_days: int = 0
    expansions: list[str] = field(default_factory=list)
    types: set[str] = field(default_factory=set)
    notes: bool = False


def read_query(text: str) -> Query:
    raw = " ".join(str(text or "").split())
    body = raw.rstrip(" ?.!")
    since = 0
    for word, days in _WHEN.items():
        if re.search(r"(?i)\b(?:von\s+)?" + word + r"\b", body):
            since = days
            body = re.sub(r"(?i)\s*\b(?:von\s+)?" + word + r"\b", "", body)
    topic = body
    for pattern in _TOPIC_PATTERNS:
        match = pattern.search(body)
        if match and match.group("t").strip():
            topic = match.group("t").strip()
            break
    topic = re.sub(r"(?i)^(?:(?:der|die|das|den|dem)\s+)?(?:abbildung|grafik|seite)\s+(?:zu[mr]?|über)\s+", "", topic).strip(" ?.!\"'„“")
    topic = re.sub(r"(?i)^(?:der|die|das|den|dem|des|ein|eine|einen|einem|einer)\s+", "", topic).strip()
    figure = bool(_FIGURE_Q.search(raw))
    terms = [t for t in fold(topic).split() if len(t) >= 2 and t not in _STOP]
    if not terms:
        terms = [t for t in fold(body).split() if len(t) >= 3 and t not in _STOP]
    types: set[str] = set()
    notes = False
    # the type hint lives in the command around the topic ("in meinen Folien"), never in the topic itself ("Folien zur Motorik" is a topic)
    position = raw.lower().find(topic.lower()) if topic else -1
    command = raw[:position] if position > 0 else (raw if topic.strip(" ?.!") == body.strip() else "")
    for pattern, kinds, is_notes in _TYPE_HINTS:
        if pattern.search(command):
            types |= kinds
            notes = notes or is_notes
    return Query(raw=raw, topic=topic, terms=list(dict.fromkeys(terms)), figure=figure, page=bool(_PAGE_Q.search(raw)), since_days=since,
                 types=types, notes=notes)


def _word_query(terms: list[str]) -> str:
    parts = []
    for term in terms:
        safe = re.sub(r"[^0-9a-z]", "", term)
        if not safe:
            continue
        parts.append(f'"{safe}"*' if len(safe) >= 4 else f'"{safe}"')
    return " OR ".join(parts)


def _trigram_query(terms: list[str]) -> str:
    pieces: list[str] = []
    for term in terms:
        safe = re.sub(r"[^0-9a-z]", "", term)
        if len(safe) < 3:
            continue
        pieces.append(safe)
        if len(safe) >= 12:  # a compound: its halves find the hyphenated or spaced spelling too
            middle = len(safe) // 2
            pieces.extend([safe[:middle + 1], safe[middle - 1:]])
    return " OR ".join(f'"{p}"' for p in dict.fromkeys(pieces))


def _stem(term: str) -> str:
    return term[:6] if len(term) > 7 else term


def _fuzzy_coverage(terms: list[str], folded: str) -> float:
    """Share of the terms that appear in near spelling (recognised handwriting: "Starlinq", "Sarkomerlange", "Frank Stariing")."""

    from difflib import SequenceMatcher

    if not terms:
        return 0.0
    words = folded.split()
    squashed = folded.replace(" ", "")
    hits = 0
    for term in terms:
        if len(term) < 5:
            if re.search(r"(?<![0-9a-z])" + re.escape(term), folded):
                hits += 1
            continue
        grams = {term[i:i + 3] for i in range(len(term) - 2)}
        found = False
        # candidates: single words and pairs of neighbouring words (a hyphen or a line break split the term)
        for i, word in enumerate(words):
            for candidate in (word, word + (words[i + 1] if i + 1 < len(words) else "")):
                if abs(len(candidate) - len(term)) > max(2, len(term) // 4):
                    continue
                shared = sum(1 for g in grams if g in candidate)
                if shared < max(2, len(grams) * 0.3):  # seen live: "Aidoteron" shares only 3 of 8 trigrams with "Aldosteron"
                    continue
                if SequenceMatcher(None, term, candidate).ratio() >= 0.8:
                    found = True
                    break
            if found:
                break
        if not found and len(term) >= 10 and SequenceMatcher(None, term, squashed).find_longest_match(0, len(term), 0, len(squashed)).size >= len(term) - 2:
            found = True
        hits += found
    return hits / len(terms)


def _coverage(terms: list[str], folded: str, squashed: str) -> float:
    if not terms:
        return 0.0
    hits = 0
    for term in terms:
        stem = _stem(term)
        if re.search(r"(?<![0-9a-z])" + re.escape(stem), folded) or (len(term) >= 8 and term in squashed):
            hits += 1
    return hits / len(terms)


def _phrase_span(folded: str, phrase: str) -> tuple[int, int, list[int]] | None:
    """Where the folded phrase stands in folded text, ignoring spaces and hyphens when it is a compound."""

    if len(phrase) >= 4:
        position = folded.find(phrase)
        if position >= 0:
            return position, position + len(phrase), []
    squashed_phrase = phrase.replace(" ", "")
    if len(squashed_phrase) >= 8:
        keep = [i for i, ch in enumerate(folded) if ch != " "]
        at = "".join(folded[i] for i in keep).find(squashed_phrase)
        if at >= 0:
            return keep[at], keep[at + len(squashed_phrase) - 1] + 1, keep
    return None


def _near_phrase(folded: str, phrase: str) -> tuple[int, int, list[int]] | None:
    """The written form of a phrase with one word misheard or misspelt ("Frank Stalin Mechanismus" -> "Frank-Starling-Mechanismus").

    The phrase's words must stand side by side in the text, all but one matching (by stem), the first and the last
    matching; three words at least, so a two-word coincidence is never taken for the phrase."""

    wanted = phrase.split()
    if len(wanted) < 3:
        return None
    words = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[0-9a-z]+", folded)]
    n = len(wanted)

    def same(word: str, target: str) -> bool:
        return word.startswith(_stem(target)) or (len(word) >= 5 and target.startswith(_stem(word)))

    for i in range(0, len(words) - n + 1):
        window = words[i:i + n]
        hits = [same(window[j][2], wanted[j]) for j in range(n)]
        if hits[0] and hits[-1] and sum(hits) >= n - 1:
            return window[0][0], window[-1][1], []
    return None


def _excerpt(text: str, query: Query, *, width: int = 280) -> tuple[str, list[list[int]], str, int]:
    """A window of the chunk around the best hit, highlight spans inside it, the phrase to mark on the page, and where the hit starts."""

    folded, index = fold_with_index(text)
    phrase = fold(query.topic)
    focus = ""
    span = _phrase_span(folded, phrase)
    if span is None:
        span = _near_phrase(folded, phrase)
    if span is not None:
        start_raw, end_raw = index[span[0]], index[span[1] - 1] + 1
        focus = text[start_raw:end_raw]
    else:
        start_raw = end_raw = -1
        for term in sorted(query.terms, key=len, reverse=True):
            found = re.search(r"(?<![0-9a-z])" + re.escape(_stem(term)), folded)
            if found:
                start_raw = index[found.start()]
                end = found.start()
                while end < len(folded) and folded[end] != " ":
                    end += 1
                end_raw = index[end - 1] + 1
                focus = text[start_raw:end_raw]
                break
    if start_raw < 0:
        snippet = text[:width].strip().replace("\n", " ")
        return snippet + ("…" if len(text) > width else ""), [], "", -1
    left = max(0, start_raw - width // 3)
    right = min(len(text), left + width)
    left = max(0, min(left, right - width))
    while left > 0 and not text[left - 1].isspace():
        left -= 1
    while right < len(text) and not text[right - 1].isspace():
        right += 1
    window = text[left:right]
    spans: list[list[int]] = []
    window_folded, window_index = fold_with_index(window)
    needles = ([phrase] if len(phrase) >= 4 and window_folded.find(phrase) >= 0 else []) + [_stem(t) for t in query.terms if len(t) >= 3]
    for needle in needles:
        for found in re.finditer(r"(?<![0-9a-z])" + re.escape(needle), window_folded):
            end = found.end()
            while end < len(window_folded) and window_folded[end] != " " and needle != phrase:
                end += 1
            spans.append([window_index[found.start()], window_index[end - 1] + 1])
    if focus and window.find(focus) >= 0:
        spans.append([window.find(focus), window.find(focus) + len(focus)])
    spans.sort()
    merged: list[list[int]] = []
    for item in spans:
        if merged and item[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], item[1])
        else:
            merged.append(item)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    shift = len(prefix)
    # a heading and its first paragraph are separate lines in the file; in one excerpt line they read as a sentence without a break
    clean = re.sub(r"\n\s*\n", " · ", window).replace("\n", " ")
    shift_map = _collapse_map(window)
    return (prefix + clean + suffix, [[shift_map[a] + shift, shift_map[min(b, len(window))] + shift] for a, b in merged],
            focus.replace("\n", " ").strip(), start_raw)


def _collapse_map(window: str) -> list[int]:
    """Index map from the raw window to the excerpt text where blank-line breaks became " · " (3 characters)."""

    mapping = [0] * (len(window) + 1)
    out = 0
    i = 0
    while i < len(window):
        match = re.match(r"\n\s*\n", window[i:])
        if match:
            for j in range(match.end()):
                mapping[i + j] = out
            out += 3
            i += match.end()
            continue
        mapping[i] = out
        out += 1
        i += 1
    mapping[len(window)] = out
    return mapping


def highlight_phrases(unit_text: str, query: Query, focus: str, *, limit: int = 6) -> list[str]:
    """Every phrase worth marking on the page: the focus first, then the topic's own words as they are written there."""

    phrases: list[str] = []
    if focus:
        phrases.append(focus)
    folded, index = fold_with_index(unit_text)
    seen = {fold(focus)} if focus else set()
    exact = bool(focus) and fold(focus) == fold(query.topic)
    if exact and len(query.terms) <= 1:
        return phrases
    for term in sorted(query.terms, key=len, reverse=True):
        if len(term) < 5:
            continue
        for found in re.finditer(r"(?<![0-9a-z])" + re.escape(_stem(term)) + r"[0-9a-z]*", folded):
            word = unit_text[index[found.start()]:index[found.end() - 1] + 1]
            key = fold(word)
            if key and key not in seen and not any(key in s for s in seen):
                seen.add(key)
                phrases.append(word)
            break
        if len(phrases) >= limit:
            break
    return phrases


def _location(chunk: dict[str, Any], *, index: StudyIndex | None = None, hit: int = -1, blocks: dict[tuple[str, int], list] | None = None) -> Location:
    heading = [h for h in str(chunk.get("heading") or "").split(" › ") if h] if chunk.get("page") is None and chunk.get("slide") is None else []
    location = Location(page=chunk.get("page"), slide=chunk.get("slide"), heading_path=heading, paragraph=chunk.get("paragraph"),
                        unit=int(chunk.get("unit") or 1), char_start=int(chunk.get("char_start") or 0), char_end=int(chunk.get("char_end") or 0))
    if index is None or hit < 0 or chunk.get("page") is not None or chunk.get("slide") is not None:
        return location
    # A section chunk spans several paragraphs: the location is the paragraph the hit is in, not the chunk's first one.
    key = (chunk["document_id"], location.unit)
    if blocks is not None and key not in blocks:
        unit = index.unit(*key)
        blocks[key] = (unit or {}).get("blocks") or []
    position = location.char_start + hit
    for block in (blocks or {}).get(key, []):
        if int(block.get("char_start") or 0) <= position < int(block.get("char_end") or 0) and block.get("paragraph") is not None:
            location.paragraph = block["paragraph"]
            location.heading_path = list(block.get("heading_path") or location.heading_path)
            break
    return location


def _passes_filters(doc: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, wanted in filters.items():
        if wanted in (None, "", [], ()):
            continue
        if key in {"source_type", "type"}:
            allowed = {str(w).lower() for w in (wanted if isinstance(wanted, (list, tuple, set)) else [wanted])}
            if str(doc.get("source_type") or "").lower() not in allowed:
                return False
        elif key == "goodnotes":
            if bool((doc.get("flags") or {}).get("goodnotes")) != bool(wanted):
                return False
        elif key in {"course", "subject", "semester", "module", "topic", "provider"}:
            options = {str(w).lower() for w in (wanted if isinstance(wanted, (list, tuple, set)) else [wanted])}
            if str(doc.get(key) or "").lower() not in options:
                return False
    return True


def search(index: StudyIndex, text: str, *, limit: int = 8, expansions: list[str] | None = None, document_id: str = "",
           filters: dict[str, Any] | None = None, per_document: int = 3, semantic: list[tuple[int, float]] | None = None,
           semantic_median: float | None = None, similarity_for: Callable[[list[int]], dict[int, float]] | None = None) -> dict[str, Any]:
    """Ranked sources for a natural-language question, grouped by document.

    ``semantic`` are the question's nearest chunks as (chunk rowid, cosine similarity), best first, and
    ``semantic_median`` the median similarity of the question to the whole library, ``similarity_for`` the similarity of
    any other chunk (for word matches outside the semantic neighbours) -- all from the service.
    """

    started = time.perf_counter()
    query = read_query(text)
    query.expansions = [fold(e) for e in (expansions or []) if fold(e) and fold(e) not in {fold(query.topic)}][:8]
    semantic = list(semantic or [])
    if not query.terms and not semantic:
        return {"ok": True, "query": query.raw, "topic": query.topic, "results": [], "reason": "no searchable words"}
    expansion_terms = [t for e in query.expansions for t in e.split() if t not in _STOP and len(t) >= 3 and t not in query.terms]
    lists = {
        "word": index.match_word(_word_query(query.terms)) if query.terms else [],
        "tri": index.match_trigram(_trigram_query(query.terms)) if query.terms else [],
        "exp": index.match_word(_word_query(list(dict.fromkeys(expansion_terms)))) if expansion_terms else [],
    }
    weights = {"word": 1.0, "tri": 0.75, "exp": 0.45}
    fused: dict[int, float] = {}
    for name, hits in lists.items():
        for rank, (rowid, _score) in enumerate(hits):
            fused[rowid] = fused.get(rowid, 0.0) + weights[name] / (1.0 + 0.2 * rank)
    closeness: dict[int, float] = {}
    top_similarity = semantic[0][1] if semantic else 0.0
    median = semantic_median if semantic_median is not None else (semantic[-1][1] if semantic else 0.0)
    spread = max(0.0, top_similarity - median)
    trust = max(0.3, min(1.0, spread / 0.06))  # a flat similarity profile is weak evidence
    for rowid, similarity in semantic:
        closeness[rowid] = similarity
        fused.setdefault(rowid, 0.0)
    if query.terms and any(len(t) >= 5 for t in query.terms):
        # recognised handwriting and scans: near spellings the word indexes cannot find become candidates too (bounded)
        for rowid, text in index.recognised_chunks(limit=4000, document_id=document_id):
            if rowid not in fused and _fuzzy_coverage(query.terms, fold(text)) >= 0.5:
                fused[rowid] = 0.3
    if semantic and similarity_for is not None:
        unknown = [rowid for rowid in fused if rowid not in closeness]
        if unknown:
            closeness.update(similarity_for(unknown))
    chunks = index.chunks(list(fused))
    phrase = fold(query.topic)
    squashed_phrase = phrase.replace(" ", "")
    documents: dict[str, dict[str, Any]] = {}
    cutoff = time.time() - query.since_days * 86400 if query.since_days else 0
    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for rowid, fusion in fused.items():
        chunk = chunks.get(rowid)
        if chunk is None or (document_id and chunk["document_id"] != document_id):
            continue
        doc = documents.get(chunk["document_id"])
        if doc is None:
            doc = index.document(chunk["document_id"]) or {}
            documents[chunk["document_id"]] = doc
        if not doc or not _passes_filters(doc, filters):
            continue
        if cutoff and max(float(doc.get("updated_at") or 0), float(doc.get("created_at") or 0), float(doc.get("opened_at") or 0)) < cutoff:
            continue
        folded = fold(chunk["text"])
        squashed = folded.replace(" ", "")
        heading = fold(f"{chunk.get('heading') or ''} {doc.get('title') or ''}")
        coverage = _coverage(query.terms, folded, squashed)
        recognised = chunk.get("ocr_confidence") is not None or bool(chunk.get("handwriting"))
        if filters and filters.get("handwriting") and not recognised:
            continue  # "Wo habe ich das handschriftlich notiert?": only recognised pages count
        fuzzy_only = False
        if recognised and coverage < 1:
            # recognised handwriting is spelled imperfectly: a near spelling of a word counts, and says it was near
            fuzzy = _fuzzy_coverage(query.terms, folded)
            if fuzzy > coverage:
                fuzzy_only = coverage == 0
                coverage = fuzzy
        similarity = closeness.get(rowid)
        relative = 0.0
        if similarity is not None and spread > 0:
            relative = max(0.0, min(1.0, (similarity - median) / spread))
        semantic_only = coverage == 0 and not any(t in folded for t in expansion_terms)
        exact = len(phrase) >= 4 and (phrase in folded or (len(squashed_phrase) >= 8 and squashed_phrase in squashed))
        if semantic_only:
            if similarity is None or similarity < SEMANTIC_MIN_SIMILARITY or similarity - median < SEMANTIC_MIN_MARGIN:
                continue
        elif (semantic and not exact and coverage < 0.75 and similarity is not None and similarity < SEMANTIC_VETO_BELOW
              and similarity - median < SEMANTIC_MIN_MARGIN):
            continue
        evidence = 1.3 * coverage
        if exact:
            evidence += 1.6
        evidence += 0.5 * _coverage(query.terms, heading, heading.replace(" ", ""))
        names = fold(f"{doc.get('filename') or ''} {doc.get('title') or ''}")
        if len(phrase) >= 4 and (phrase in names or (len(squashed_phrase) >= 8 and squashed_phrase in names.replace(" ", ""))):
            evidence += 1.0  # the file itself is named after what was asked
        if query.figure and chunk.get("figure_no"):
            evidence += 1.2  # a located figure (caption + region), not only a page that has an image somewhere
        elif query.figure and chunk.get("figure"):
            evidence += 0.7
        if query.page and (chunk.get("page") is not None or chunk.get("slide") is not None):
            evidence += 0.3  # "die Seite zu ..." asks for something with pages
        # "meine Notizen" / "in meinen Folien" name the kind of material outright: among places that say the same, that kind wins;
        # a better place in another kind still ranks, just after it
        if query.notes and ((doc.get("flags") or {}).get("goodnotes") or doc.get("source_type") in query.types):
            evidence += 1.2
        elif query.types and doc.get("source_type") in query.types:
            evidence += 1.0
        meaning = SEMANTIC_WEIGHT * trust * relative * (1.0 if coverage < 1 else 0.5)
        confidence = chunk.get("ocr_confidence")
        if recognised:
            # words read from a scan or handwriting count by how sure the recognition was; printed text keeps full weight
            evidence *= 0.55 + 0.45 * (float(confidence) if confidence is not None else 0.6)
            if fuzzy_only:
                evidence *= 0.85
        score = evidence + 0.35 * fusion + meaning
        match = "phrase" if exact else ("terms" if coverage >= 0.99 and not fuzzy_only else ("semantic" if semantic_only else "partial"))
        certainty = "sure"
        # recognised text is only sure where every searched word stands there letter for letter; a stem or near spelling
        # ("Natriumrueckresorotion", "Nieren: musiogie") is a possible match the owner checks on the page
        literal = bool(query.terms) and all(t in squashed for t in query.terms)
        if recognised and (fuzzy_only or semantic_only or not (exact or literal) or (confidence is not None and float(confidence) < 0.6)):
            certainty = "possible"
        scored.append((score, chunk, {"coverage": round(coverage, 2), "exact": exact, "match": match,
                                      "similarity": round(similarity, 4) if similarity is not None else None,
                                      "handwriting": bool(chunk.get("handwriting")), "recognised": recognised, "certainty": certainty,
                                      "ocr_confidence": confidence}))
    scored.sort(key=lambda item: item[0], reverse=True)

    grouped: dict[str, dict[str, Any]] = {}
    unit_blocks: dict[tuple[str, int], list] = {}
    unit_texts: dict[tuple[str, int], str] = {}
    for score, chunk, why in scored:
        doc_id = chunk["document_id"]
        entry = grouped.get(doc_id)
        if entry is not None and len(entry["more"]) >= per_document - 1:
            continue
        excerpt, spans, focus, hit_at = _excerpt(chunk["text"], query)
        location = _location(chunk, index=index, hit=hit_at, blocks=unit_blocks)
        if entry is not None and location.unit in entry["_units"]:
            continue
        key = (doc_id, location.unit)
        if key not in unit_texts:
            unit_texts[key] = ((index.unit(*key) or {}).get("text") or "")
        at = location.char_start + max(0, hit_at)
        hit = {"location": location.to_dict(), "excerpt": excerpt, "highlights": spans, "focus": focus, "at": at,
               "phrases": highlight_phrases(unit_texts[key], query, focus), "score": round(score, 3), **why}
        if chunk.get("figure_no"):
            figure = next((f for f in index.figures(doc_id, location.unit) if int(f["number"]) == int(chunk["figure_no"])), None)
            if figure is not None:
                hit["figure"] = {"number": figure["number"], "kind": figure["kind"], "box": figure["box"], "caption": figure["caption"]}
        if entry is None:
            doc = documents[doc_id]
            grouped[doc_id] = {"document": {k: doc.get(k) for k in ("id", "title", "filename", "source_type", "course", "subject", "semester",
                                                                    "module", "topic", "provider", "units", "flags")},
                               "score": score, "best": hit, "more": [], "_units": {location.unit}, "_chunk": chunk}
            continue
        entry["_units"].add(location.unit)
        entry["more"].append(hit)
        entry["score"] += 0.08 * score / max(1.0, len(entry["more"]))
    results = sorted(grouped.values(), key=lambda r: r["score"], reverse=True)[:limit]
    top = results[0]["score"] if results else 1.0
    for result in results:
        result.pop("_units", None)
        result.pop("_chunk", None)
        result["relevance"] = round(max(0.05, min(1.0, result["score"] / top)), 3)
        result["score"] = round(result["score"], 3)
    return {"ok": True, "query": query.raw, "topic": query.topic, "figure": query.figure, "since_days": query.since_days,
            "types": sorted(query.types), "expanded": query.expansions, "results": results, "semantic": bool(semantic),
            "seconds": round(time.perf_counter() - started, 3)}
