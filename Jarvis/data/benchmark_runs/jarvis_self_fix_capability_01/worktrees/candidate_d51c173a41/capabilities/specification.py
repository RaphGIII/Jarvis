from __future__ import annotations

import json
import re
from typing import Any

from capabilities.models import SkillSpecification


def skill_specification_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "capability_id": {"type": "string"},
            "objective": {"type": "string"},
            "functional_requirements": {"type": "array", "items": {"type": "string"}},
            "inputs": {"type": "object", "additionalProperties": True},
            "outputs": {"type": "object", "additionalProperties": True},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "allowed_dependencies": {"type": "array", "items": {"type": "string"}},
            "permissions": {"type": "array", "items": {"type": "string"}},
            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
            "public_tests": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            "proposed_file_structure": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["capability_id", "objective", "acceptance_criteria", "public_tests", "proposed_file_structure"],
    }


class SkillSpecificationGenerator:
    """Creates a structured skill specification for a missing capability."""

    def __init__(self, brain: Any | None = None) -> None:
        self.brain = brain

    def generate(self, goal: str) -> SkillSpecification:
        if self.brain is not None:
            spec = self._generate_with_brain(goal)
            if spec is not None:
                return spec
        return self._fallback_spec(goal)

    def _generate_with_brain(self, goal: str) -> SkillSpecification | None:
        prompt = (
            "Return JSON only. Create a safe local Jarvis SkillSpecification for this missing capability.\n"
            "The skill must expose main.py with def run(payload: dict) -> dict.\n"
            "Use only Python standard library dependencies unless the request absolutely requires more.\n"
            "Do not request credentials, network, browser, email, or external permissions unless the goal explicitly requires them.\n"
            f"Goal: {goal}"
        )
        try:
            if hasattr(self.brain, "generate_structured"):
                raw = self.brain.generate_structured(
                    prompt,
                    skill_specification_json_schema(),
                    max_tokens=700,
                    temperature=0.2,
                    top_p=0.9,
                )
            else:
                raw = self.brain.generate(prompt, max_tokens=700, temperature=0.2, top_p=0.9)
            spec = SkillSpecification.from_dict(json.loads(_extract_json(raw)))
        except Exception:
            return None
        return spec if not spec.validate() else None

    @staticmethod
    def _fallback_spec(goal: str) -> SkillSpecification:
        slug = re.sub(r"[^a-z0-9]+", ".", goal.lower()).strip(".")[:48] or "local.utility"
        if not slug.startswith("local."):
            slug = f"local.{slug}"
        return SkillSpecification(
            capability_id=slug,
            objective=goal,
            functional_requirements=["Implement a deterministic local utility for the requested transformation."],
            inputs={"type": "object", "additionalProperties": True},
            outputs={"type": "object", "additionalProperties": True},
            constraints=["No network access.", "Use Python standard library only.", "Expose run(payload: dict) -> dict."],
            permissions=[],
            acceptance_criteria=["Public examples pass.", "Hidden verifier passes.", "Original request executes successfully."],
            public_tests=[
                {
                    "name": "returns_object",
                    "input": {"text": "hello"},
                    "expected_keys": ["result"],
                }
            ],
            proposed_file_structure=["main.py"],
        )


def _extract_json(text: str) -> str:
    match = re.search(r"(\{.*\})", text.strip(), flags=re.DOTALL)
    return match.group(1) if match else text
