"""Evidence, first-class: what actually proves this?

Seven kinds, in rising order of what they are allowed to support:

    CLAIM              someone said so (a model, a page, a prompt)
    OWNER_STATEMENT    the owner said so -- authoritative about intent, not about the world
    MODEL_INFERENCE    a model concluded it; never proof of anything outside the model
    TOOL_OBSERVATION   a tool read the world (a file listing, a process table, an HTTP status)
    EXTERNAL_SOURCE    a document or page, with its provenance and freshness
    EXECUTION_RECEIPT  a tool did something and an *independent* check observed the effect
    VERIFIED_FACT      an observation confirmed by a verifier that is not the writer

And two verdicts that must never be confused:

    EXECUTION_VERIFIED  the action happened and its effect was observed
    GOAL_SATISFIED      the owner's intent was met -- a separate question, often only
                        the owner can answer it (that is what Korrigieren is for)

The rule that matters most: the thing that created an effect may not be the
thing that confirms it.  ``Verifier.confirm`` refuses when the observer is the
writer.  A receipt from :mod:`runtime.receipts` is turned into evidence with
its verifications intact, so what is already checked is not re-trusted.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceKind(str, Enum):
    CLAIM = "claim"
    OWNER_STATEMENT = "owner_statement"
    MODEL_INFERENCE = "model_inference"
    TOOL_OBSERVATION = "tool_observation"
    EXTERNAL_SOURCE = "external_source"
    EXECUTION_RECEIPT = "execution_receipt"
    VERIFIED_FACT = "verified_fact"

    @property
    def strength(self) -> int:
        return list(EvidenceKind).index(self)

    @property
    def proves_execution(self) -> bool:
        return self in {EvidenceKind.EXECUTION_RECEIPT, EvidenceKind.VERIFIED_FACT}


@dataclass
class Evidence:
    kind: EvidenceKind
    claim: str
    #: Who produced it: a tool name, a model tier, "owner", a URL's host.
    source: str = ""
    #: Who confirmed it independently, if anyone.
    verifier: str = ""
    observed: str = ""
    receipt_id: str = ""
    url: str = ""
    fetched_at: str = ""
    confidence: float = 0.0
    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:10]}")
    at: str = field(default_factory=_now)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def execution_verified(self) -> bool:
        return self.kind.proves_execution and bool(self.verifier) and self.verifier != self.source

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["kind"] = self.kind.value
        out["execution_verified"] = self.execution_verified
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        known["kind"] = EvidenceKind(known.get("kind", "claim"))
        return cls(**known)


def claim(text: str, *, source: str = "model") -> Evidence:
    return Evidence(EvidenceKind.CLAIM, text, source=source, confidence=0.2)


def owner_statement(text: str) -> Evidence:
    return Evidence(EvidenceKind.OWNER_STATEMENT, text, source="owner", confidence=0.9)


def inference(text: str, *, tier: str = "FAST_LOCAL", confidence: float = 0.4) -> Evidence:
    return Evidence(EvidenceKind.MODEL_INFERENCE, text, source=tier, confidence=confidence)


def observation(text: str, *, tool: str, observed: str = "", data: dict[str, Any] | None = None) -> Evidence:
    return Evidence(EvidenceKind.TOOL_OBSERVATION, text, source=tool, observed=observed, confidence=0.8, data=dict(data or {}))


def external(text: str, *, url: str, fetched_at: str = "", quote: str = "", authority: int = 0) -> Evidence:
    host = url.split("/")[2] if "://" in url else url
    return Evidence(EvidenceKind.EXTERNAL_SOURCE, text, source=host, url=url, fetched_at=fetched_at or _now(),
                    observed=quote, confidence=min(0.9, 0.4 + authority / 10), data={"authority": authority})


def from_receipt(receipt: Any) -> Evidence:
    """A :class:`runtime.receipts.Receipt` as evidence, keeping what checked it.

    A receipt that verified nothing is a claim by its executor; one whose
    checks passed is an execution receipt whose verifier is the checks.
    """

    data = receipt.to_dict() if hasattr(receipt, "to_dict") else dict(receipt)
    checks = data.get("verifications") or []
    passed = [c for c in checks if c.get("passed")]
    verified = bool(data.get("verified")) and bool(passed)
    kind = EvidenceKind.EXECUTION_RECEIPT if verified else EvidenceKind.CLAIM
    return Evidence(
        kind, str(data.get("detail") or data.get("kind") or "action"), source=str(data.get("executor", "")),
        verifier=", ".join(str(c.get("check", "")) for c in passed)[:200] if verified else "",
        observed="; ".join(str(c.get("observed", "")) for c in passed)[:400], receipt_id=str(data.get("id", "")),
        confidence=0.85 if verified else 0.3, data={"ok": bool(data.get("ok")), "kind": data.get("kind")},
    )


class Verifier:
    """Independent confirmation.  The observer must not be the writer."""

    def __init__(self, name: str) -> None:
        self.name = name

    def confirm(self, evidence: Evidence, check: Callable[[], tuple[bool, str]]) -> Evidence:
        """Run ``check`` (an observation that does not come from ``evidence.source``)
        and return a VERIFIED_FACT on success, or the original evidence unchanged."""

        if evidence.source == self.name:
            raise ValueError(f"{self.name} produced this evidence and may not verify it")
        ok, observed = check()
        if not ok:
            return Evidence(evidence.kind, evidence.claim, source=evidence.source, verifier="", observed=observed,
                            receipt_id=evidence.receipt_id, confidence=min(evidence.confidence, 0.2),
                            data={**evidence.data, "refuted_by": self.name})
        return Evidence(EvidenceKind.VERIFIED_FACT, evidence.claim, source=evidence.source, verifier=self.name, observed=observed,
                        receipt_id=evidence.receipt_id, url=evidence.url, confidence=max(evidence.confidence, 0.9),
                        data={**evidence.data, "verified_from": evidence.evidence_id})


@dataclass
class Verdict:
    """The two questions, answered separately."""

    execution_verified: bool
    goal_satisfied: bool | None  # None = unknown (only the owner can say)
    basis: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verdict(evidence: list[Evidence], *, goal_check: Callable[[], bool | None] | None = None) -> Verdict:
    """What the evidence supports.  Execution needs a receipt or a verified fact
    with an independent verifier; goal satisfaction needs a goal check (or the
    owner) and is never inferred from execution alone."""

    proof = [e for e in evidence if e.execution_verified]
    execution = bool(proof)
    goal: bool | None = None
    if goal_check is not None:
        try:
            goal = goal_check()
        except Exception:  # noqa: BLE001
            goal = None
    basis = [f"{e.kind.value} by {e.source} verified by {e.verifier}" for e in proof][:6]
    confidence = max((e.confidence for e in proof), default=0.0) if execution else max((e.confidence for e in evidence), default=0.0) * 0.5
    return Verdict(execution_verified=execution, goal_satisfied=goal, basis=basis, confidence=round(confidence, 2))


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
