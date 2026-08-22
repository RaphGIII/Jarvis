from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass
class DevelopmentExperience:
    fingerprint: str
    goal: str
    spec: dict[str, Any]
    plan: str
    implementation: list[dict[str, str]]
    failures: list[dict[str, Any]] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    final_code: list[dict[str, str]] = field(default_factory=list)
    public_success: bool = False
    internal_verification_success: bool = False
    reviewer_approved: bool = False
    hidden_success: bool | None = None
    promotion_success: bool = False
    execution_success: bool = False
    second_call_success: bool = False
    token_usage: dict[str, int] = field(default_factory=dict)
    repair_cycles: int = 0
    blind_repair_cycles: int = 0
    failure_classes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def final_success(self) -> bool:
        return bool(
            self.public_success
            and self.internal_verification_success
            and self.reviewer_approved
            and self.hidden_success
            and self.promotion_success
            and self.execution_success
            and self.second_call_success
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "goal": self.goal,
            "spec": self.spec,
            "plan": self.plan,
            "implementation": self.implementation,
            "failures": self.failures,
            "repairs": self.repairs,
            "final_code": self.final_code,
            "public_success": self.public_success,
            "internal_verification_success": self.internal_verification_success,
            "reviewer_approved": self.reviewer_approved,
            "hidden_success": self.hidden_success,
            "promotion_success": self.promotion_success,
            "execution_success": self.execution_success,
            "second_call_success": self.second_call_success,
            "final_success": self.final_success,
            "token_usage": self.token_usage,
            "repair_cycles": self.repair_cycles,
            "blind_repair_cycles": self.blind_repair_cycles,
            "failure_classes": self.failure_classes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DevelopmentExperience":
        return cls(
            fingerprint=str(data.get("fingerprint", "")),
            goal=str(data.get("goal", "")),
            spec=dict(data.get("spec") or {}),
            plan=str(data.get("plan", "")),
            implementation=list(data.get("implementation") or []),
            failures=list(data.get("failures") or []),
            repairs=list(data.get("repairs") or []),
            final_code=list(data.get("final_code") or []),
            public_success=bool(data.get("public_success", False)),
            internal_verification_success=bool(data.get("internal_verification_success", False)),
            reviewer_approved=bool(data.get("reviewer_approved", False)),
            hidden_success=data.get("hidden_success"),
            promotion_success=bool(data.get("promotion_success", False)),
            execution_success=bool(data.get("execution_success", False)),
            second_call_success=bool(data.get("second_call_success", False)),
            token_usage=dict(data.get("token_usage") or {}),
            repair_cycles=int(data.get("repair_cycles", 0)),
            blind_repair_cycles=int(data.get("blind_repair_cycles", 0)),
            failure_classes=list(data.get("failure_classes") or []),
            created_at=str(data.get("created_at", datetime.now(timezone.utc).isoformat())),
        )


class DevelopmentMemory:
    """Persistent practical build memory for future software generation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def fingerprint(self, goal: str, spec: dict[str, Any]) -> str:
        payload = json.dumps({"goal": _normalize(goal), "spec": spec}, sort_keys=True)
        return sha256(payload.encode("utf-8", errors="ignore")).hexdigest()

    def record(self, experience: DevelopmentExperience) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(experience.to_dict(), sort_keys=True) + "\n")

    def retrieve(
        self,
        goal: str,
        spec: dict[str, Any],
        *,
        failure_text: str = "",
        limit: int = 3,
    ) -> list[DevelopmentExperience]:
        goal_terms = _terms(f"{goal} {json.dumps(spec, sort_keys=True)}")
        failure_classes = set(classify_failure(failure_text))
        scored: list[tuple[float, DevelopmentExperience]] = []
        for exp in self.load_all():
            target_terms = _terms(f"{exp.goal} {json.dumps(exp.spec, sort_keys=True)}")
            overlap = len(goal_terms & target_terms) / max(1, len(goal_terms))
            failure_overlap = len(failure_classes & set(exp.failure_classes)) * 0.3
            success_bonus = 0.2 if exp.final_success else 0.0
            partial_penalty = -0.1 if exp.public_success and not exp.final_success else 0.0
            score = overlap + failure_overlap + success_bonus + partial_penalty
            if score > 0:
                scored.append((score, exp))
        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [exp for _, exp in scored[:limit]]

    def load_all(self) -> list[DevelopmentExperience]:
        if not self.path.exists():
            return []
        return [
            DevelopmentExperience.from_dict(json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def classify_failure(text: str) -> list[str]:
    lowered = text.lower()
    classes = []
    markers = {
        "ModuleNotFoundError": "module not found",
        "SyntaxError": "syntaxerror",
        "AssertionError": "assert",
        "ImportError": "importerror",
        "KeyError": "keyerror",
        "TypeError": "typeerror",
        "ValueError": "valueerror",
        "wrong_output_schema": "expected",
        "empty_input": "empty",
        "csv_edge_case": "csv",
        "off_by_one": "off-by-one",
    }
    for name, marker in markers.items():
        if marker in lowered:
            classes.append(name)
    return classes or (["test_failure"] if text.strip() else [])


def _terms(text: str) -> set[str]:
    return {term for term in re.split(r"[^a-z0-9]+", text.lower()) if len(term) > 2}


def _normalize(text: str) -> str:
    return " ".join(sorted(_terms(text)))
