"""The privacy router: what may leave the machine, and to whom.

Runs before any model call.  Context is handed over as *chunks*, each with a
source label, and every chunk is classified on its own:

    PUBLIC   -- general knowledge, the owner's plain question, code snippets
    PRIVATE  -- the owner's documents, mail, memory, screen, finances, health records
    SECRET   -- credentials, tokens, passwords: never sent anywhere

A provider that may use request content for product improvement (the free
tier) receives PUBLIC chunks only.  A provider under a no-training contract
may receive PRIVATE chunks.  SECRET never leaves.

Two things this module is deliberately not: it is not a filter that "cleans"
private text so it can go to the free lane (redaction of a medical record is
still a medical record), and it is not the place where the owner's *question*
is judged too sensitive to answer -- only where it may be answered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class Sensitivity(str, Enum):
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"
    SECRET = "SECRET"

    @property
    def rank(self) -> int:
        return {"PUBLIC": 0, "PRIVATE": 1, "SECRET": 2}[self.value]


#: Chunk sources that are private by construction, whatever they contain.
PRIVATE_SOURCES: frozenset[str] = frozenset({
    "memory", "document", "documents", "email", "mail", "screen", "screen_history", "recording", "recordings",
    "finance", "medical_record", "calendar", "contacts", "filesystem", "conversation_archive", "owner_profile",
    "knowledge_private", "project_files",
})

#: Chunk sources that are secrets by construction.
SECRET_SOURCES: frozenset[str] = frozenset({"secret", "secrets", "credentials", "api_key", "password", "token"})

#: Sources that carry the owner's own words or system-generated public text.
PUBLIC_SOURCES: frozenset[str] = frozenset({"owner_message", "instruction", "system_context", "capability_catalog",
                                            "conversation", "web", "public_knowledge", "code"})

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"(?i)\b(passwor[dt]|passwd|kennwort|pin)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b(api[_ -]?key|secret|token|bearer)\b\s*[:=]\s*[A-Za-z0-9_\-\.]{8,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

_FINANCE_PATTERNS = [
    re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b"),  # IBAN
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),  # card-number shaped
    re.compile(r"(?i)\b(kontostand|kontoauszug|gehaltsabrechnung|steuerbescheid|account balance|bank statement|payslip)\b"),
]

_MEDICAL_RECORD_PATTERNS = [
    re.compile(r"(?i)\b(befund|arztbrief|diagnose[:]|laborwert|laborbericht|patient(?:in)?[:]|krankschreibung|"
               r"entlassungsbrief|medical record|lab report|discharge summary|diagnosis[:])\b"),
    re.compile(r"(?i)\b[A-Z]\d{2}(?:\.\d{1,2})?\b(?=.*\b(diagnose|diagnosis|icd)\b)"),  # ICD code near "diagnosis"
    re.compile(r"(?i)\b\d{1,3}(?:[.,]\d+)?\s?(mg/dl|mmol/l|µmol/l|ng/ml|g/dl)\b"),
]

_PRIVATE_DOCUMENT_PATTERNS = [
    re.compile(r"(?i)^(from|to|cc|subject|von|an|betreff)\s*:", re.MULTILINE),
    re.compile(r"(?i)[A-Za-z]:\\Users\\[^\\\s]+\\(Documents|Dokumente|Desktop|Downloads|Pictures|Bilder)\\"),
]

_EMAIL_ADDRESS = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str = "owner_message"
    #: A caller that knows better may pin the classification.
    sensitivity: Sensitivity | None = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "label": self.label, "chars": len(self.text),
                "sensitivity": self.sensitivity.value if self.sensitivity else None}


@dataclass
class PrivacyDecision:
    #: Chunks the chosen route may receive, in the original order.
    allowed: list[Chunk]
    #: Chunks withheld, with the reason each was withheld.
    withheld: list[tuple[Chunk, str]]
    #: Highest sensitivity seen across *all* chunks.
    ceiling: Sensitivity
    #: Whether a may-train provider may receive this request at all.
    free_lane_allowed: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def sensitive(self) -> bool:
        return self.ceiling is not Sensitivity.PUBLIC

    def prompt_text(self) -> str:
        return "\n\n".join(chunk.text for chunk in self.allowed if chunk.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ceiling": self.ceiling.value, "free_lane_allowed": self.free_lane_allowed,
            "allowed": [chunk.to_dict() for chunk in self.allowed],
            "withheld": [{**chunk.to_dict(), "reason": reason} for chunk, reason in self.withheld],
            "reasons": list(self.reasons),
        }


def classify_text(text: str) -> tuple[Sensitivity, str]:
    """Content-based classification of one piece of text."""

    body = str(text or "")
    if not body.strip():
        return Sensitivity.PUBLIC, ""
    for pattern in _SECRET_PATTERNS:
        if pattern.search(body):
            return Sensitivity.SECRET, "credential-shaped content"
    for pattern in _FINANCE_PATTERNS:
        if pattern.search(body):
            return Sensitivity.PRIVATE, "financial data"
    for pattern in _MEDICAL_RECORD_PATTERNS:
        if pattern.search(body):
            return Sensitivity.PRIVATE, "medical record"
    for pattern in _PRIVATE_DOCUMENT_PATTERNS:
        if pattern.search(body):
            return Sensitivity.PRIVATE, "private document or mail"
    if len(_EMAIL_ADDRESS.findall(body)) >= 2:
        return Sensitivity.PRIVATE, "personal addresses"
    return Sensitivity.PUBLIC, ""


def classify(chunk: Chunk) -> tuple[Sensitivity, str]:
    if chunk.sensitivity is not None:
        return chunk.sensitivity, "declared by caller"
    source = str(chunk.source or "").strip().lower()
    if source in SECRET_SOURCES:
        return Sensitivity.SECRET, f"source {source!r}"
    content, why = classify_text(chunk.text)
    if content is Sensitivity.SECRET:
        return content, why
    if source in PRIVATE_SOURCES:
        return Sensitivity.PRIVATE, f"source {source!r}"
    return content, why


class PrivacyRouter:
    """Decides, per provider, which chunks may be sent."""

    def decide(self, chunks: Iterable[Chunk], *, provider_may_train: bool) -> PrivacyDecision:
        allowed: list[Chunk] = []
        withheld: list[tuple[Chunk, str]] = []
        ceiling = Sensitivity.PUBLIC
        reasons: list[str] = []
        for chunk in chunks:
            level, why = classify(chunk)
            if level.rank > ceiling.rank:
                ceiling = level
            if level is Sensitivity.SECRET:
                withheld.append((chunk, f"secret: {why}"))
                reasons.append(f"{chunk.source}: {why}")
                continue
            if level is Sensitivity.PRIVATE and provider_may_train:
                withheld.append((chunk, f"private context withheld from a may-train provider: {why}"))
                reasons.append(f"{chunk.source}: {why}")
                continue
            allowed.append(chunk)
        # The free lane is closed to the request as a whole when the owner's own
        # words are sensitive: sending "the question minus the private part" is
        # not the request the owner made.
        owner_sensitive = any(
            classify(chunk)[0] is not Sensitivity.PUBLIC
            for chunk in chunks
            if str(chunk.source).lower() in {"owner_message", ""}
        ) if isinstance(chunks, (list, tuple)) else False
        free_lane_allowed = not owner_sensitive and ceiling is not Sensitivity.SECRET
        return PrivacyDecision(allowed=allowed, withheld=withheld, ceiling=ceiling,
                               free_lane_allowed=free_lane_allowed, reasons=reasons)

    def ceiling(self, chunks: Iterable[Chunk]) -> Sensitivity:
        top = Sensitivity.PUBLIC
        for chunk in chunks:
            level, _ = classify(chunk)
            if level.rank > top.rank:
                top = level
        return top
