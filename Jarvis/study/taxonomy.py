"""Where a document belongs: course, subject, semester, module, topic -- inferred, never forced.

Inference reads the folder path, the file name and the first text of the document.
It proposes; the owner corrects.  A category the owner set is marked confirmed and
never overwritten by a later inference.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

#: subject -> course.  Medicine first: the owner studies it; everything else stays general.
SUBJECTS: dict[str, str] = {
    "Anatomie": "Medizin", "Neuroanatomie": "Medizin", "Histologie": "Medizin", "Embryologie": "Medizin", "Physiologie": "Medizin",
    "Biochemie": "Medizin", "Biologie": "Medizin", "Chemie": "Medizin", "Physik": "Medizin", "Psychologie": "Medizin",
    "Soziologie": "Medizin", "Pharmakologie": "Medizin", "Pathologie": "Medizin", "Mikrobiologie": "Medizin", "Immunologie": "Medizin",
    "Innere Medizin": "Medizin", "Chirurgie": "Medizin", "Neurologie": "Medizin", "Kardiologie": "Medizin", "Radiologie": "Medizin",
    "Genetik": "Medizin", "Hygiene": "Medizin", "Epidemiologie": "Medizin", "Rechtsmedizin": "Medizin", "Pädiatrie": "Medizin",
    "Gynäkologie": "Medizin", "Psychiatrie": "Medizin", "Dermatologie": "Medizin", "Anästhesie": "Medizin", "Notfallmedizin": "Medizin",
}
_SUBJECT_ALIASES = {"anat": "Anatomie", "physio": "Physiologie", "biochem": "Biochemie", "histo": "Histologie", "pharma": "Pharmakologie",
                    "patho": "Pathologie", "mikrobio": "Mikrobiologie", "immuno": "Immunologie", "neuroanat": "Neuroanatomie"}

#: topic -> the words that point to it.  Used for the topic proposal and for browsing, never for access.
TOPICS: dict[str, tuple[str, ...]] = {
    "Herz": ("herz", "kardial", "myokard", "ventrikel", "frank-starling", "systole", "diastole", "koronar"),
    "Kreislauf": ("kreislauf", "blutdruck", "gefäß", "gefäss", "arterie", "vene", "kapillar", "hämodynamik"),
    "Niere": ("niere", "nephron", "glomerul", "tubulus", "aldosteron", "renin", "harn"),
    "Lunge & Atmung": ("lunge", "atmung", "ventilation", "alveol", "respirat", "compliance"),
    "Nervensystem": ("nerv", "neuron", "synapse", "basalganglien", "kortex", "rückenmark", "plexus", "hirn"),
    "Stoffwechsel": ("citratzyklus", "glykolyse", "atmungskette", "stoffwechsel", "enzym", "atp", "gluconeogenese", "glukoneogenese"),
    "Blut & Immunsystem": ("blut", "erythrozyt", "leukozyt", "immun", "antikörper", "lymph"),
    "Verdauung": ("magen", "darm", "leber", "pankreas", "verdauung", "galle"),
    "Hormone": ("hormon", "endokrin", "schilddrüse", "insulin", "cortisol", "hypophyse"),
    "Bewegungsapparat": ("muskel", "knochen", "gelenk", "sehne", "skelett"),
}

_SEMESTER = re.compile(r"(?i)\b(?:(\d{1,2})\s*\.?\s*(?:fach)?(?:semester|sem)\b|(?:semester|sem)\s*(\d{1,2})\b|(ws|wise|ss|sose)\s*(\d{2}(?:/\d{2})?|\d{4}(?:/\d{2,4})?))")


def _norm(value: str) -> str:
    return value.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")


def subject_of(text: str) -> str:
    folded = _norm(text)
    for subject in sorted(SUBJECTS, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(_norm(subject)), folded):
            return subject
    for alias, subject in _SUBJECT_ALIASES.items():
        if re.search(r"(?<![a-z])" + alias + r"(?![a-z])", folded):
            return subject
    return ""


def semester_of(text: str) -> str:
    match = _SEMESTER.search(text)
    if not match:
        return ""
    if match.group(1) or match.group(2):
        return f"{int(match.group(1) or match.group(2))}. Semester"
    term = match.group(3).lower()
    return ("WiSe " if term in {"ws", "wise"} else "SoSe ") + match.group(4)


def topic_of(text: str) -> str:
    folded = text.lower()
    best, hits = "", 0
    for topic, words in TOPICS.items():
        count = sum(folded.count(word) for word in words)
        if count > hits:
            best, hits = topic, count
    return best if hits >= 2 else ""


def infer(path: str | Path, *, title: str = "", sample: str = "", root: str | Path | None = None) -> dict[str, Any]:
    """A proposal: {course, subject, semester, module, topic}.  Empty strings where nothing points anywhere."""

    target = Path(path)
    parts = list(target.parts[:-1])
    if root is not None:
        try:
            parts = list(target.parent.relative_to(Path(root)).parts)
        except ValueError:
            pass
    folder_text = " / ".join(parts[-5:])
    head = f"{folder_text} {target.stem} {title}"
    subject = subject_of(head) or subject_of(sample[:3000])
    semester = semester_of(head) or semester_of(sample[:1500])
    course = SUBJECTS.get(subject, "")
    if not course and re.search(r"(?i)\b(medizin|medicine|klinik)\b", head):
        course = "Medizin"
    module = ""
    for part in reversed(parts[-4:]):
        cleaned = part.strip()
        if not cleaned or _norm(cleaned) in {_norm(subject), _norm(course), "studium", "uni", "unterlagen"} or _SEMESTER.search(cleaned):
            continue
        if subject and _norm(subject) in _norm(cleaned):
            continue
        module = cleaned
        break
    topic = topic_of(f"{target.stem} {title} {sample[:6000]}")
    return {"course": course, "subject": subject, "semester": semester, "module": module, "topic": topic}
