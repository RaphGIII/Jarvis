from __future__ import annotations

import json
import re
from typing import Any

from capabilities.models import CapabilityResolution
from capabilities.registry import CapabilityRegistry


class CapabilityResolver:
    """Detects whether an installed capability can satisfy a user goal."""

    def __init__(self, registry: CapabilityRegistry, brain: Any | None = None, confidence_threshold: float = 0.45) -> None:
        self.registry = registry
        self.brain = brain
        self.confidence_threshold = confidence_threshold

    def resolve(self, goal: str) -> CapabilityResolution:
        matches = self.registry.find(goal, limit=1)
        if matches:
            manifest = matches[0]
            score = self._score(goal, manifest.capability_id, manifest.description)
            if score >= self.confidence_threshold:
                return CapabilityResolution(
                    status="available",
                    capability_id=manifest.capability_id,
                    reason="Matched installed capability registry description.",
                    confidence=score,
                    manifest=manifest,
                )
        qwen = self._qwen_resolve(goal)
        if qwen is not None:
            return qwen
        return CapabilityResolution(status="missing", reason="No installed capability matched the request.", confidence=0.0)

    def _qwen_resolve(self, goal: str) -> CapabilityResolution | None:
        if self.brain is None or not self.registry.all():
            return None
        catalog = [
            {
                "capability_id": manifest.capability_id,
                "description": manifest.description,
                "input_schema": manifest.input_schema,
                "output_schema": manifest.output_schema,
            }
            for manifest in self.registry.all()
            if manifest.status == "active"
        ]
        prompt = (
            "Return JSON only. Decide if one installed capability can satisfy the user goal.\n"
            "Schema: {\"status\":\"available|missing\",\"capability_id\":\"...\",\"reason\":\"...\",\"confidence\":0.0}\n"
            f"Goal: {goal}\n"
            f"Installed capabilities: {json.dumps(catalog, sort_keys=True)}"
        )
        try:
            raw = self.brain.generate(prompt, max_tokens=300, temperature=0.0, top_p=1.0)
            data = json.loads(_extract_json(raw))
        except Exception:
            return None
        capability_id = str(data.get("capability_id") or "")
        confidence = _clamp_float(data.get("confidence", 0.0), 0.0, 1.0)
        manifest = self.registry.get(capability_id) if capability_id else None
        if data.get("status") == "available" and manifest is not None and confidence >= self.confidence_threshold:
            return CapabilityResolution("available", capability_id, str(data.get("reason", "")), confidence, manifest)
        return CapabilityResolution("missing", capability_id or None, str(data.get("reason", "Qwen found no suitable capability.")), confidence)

    @staticmethod
    def _score(goal: str, capability_id: str, description: str) -> float:
        goal_terms = _terms(goal)
        target_terms = _terms(f"{capability_id} {description}")
        if not goal_terms:
            return 0.0
        overlap = len(goal_terms & target_terms) / len(goal_terms)
        exact = 0.35 if capability_id.lower() in goal.lower() else 0.0
        return min(1.0, overlap + exact)


def _terms(text: str) -> set[str]:
    return {term for term in re.split(r"[^a-z0-9]+", text.lower()) if len(term) > 2}


def _extract_json(text: str) -> str:
    match = re.search(r"(\{.*\})", text.strip(), flags=re.DOTALL)
    return match.group(1) if match else text


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = low
    return max(low, min(high, numeric))
