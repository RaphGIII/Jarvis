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

    def __init__(self, brain: Any | None = None, *, attempts: int = 3) -> None:
        self.brain = brain
        self.attempts = max(1, attempts)

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
            "The public_tests entries must use the exact keys the implementation will read from `payload` and "
            "the exact keys it will write into the returned dict, matching the goal's actual inputs/outputs "
            "(do not reuse an unrelated example like {\"text\": \"hello\"} unless the goal is literally about text).\n"
            "Use only Python standard library dependencies unless the request absolutely requires more.\n"
            "Do not request credentials, network, browser, email, or external permissions unless the goal explicitly requires them.\n"
            f"Goal: {goal}"
        )
        last_error: str | None = None
        for _ in range(self.attempts):
            raw = self._request_raw(prompt)
            if raw is None:
                return None
            try:
                spec = SkillSpecification.from_dict(json.loads(_extract_json(raw)))
            except Exception as exc:
                last_error = f"Response was not valid JSON: {exc}"
                prompt = self._retry_prompt(goal, last_error)
                continue
            errors = spec.validate()
            if not errors:
                return spec
            last_error = "; ".join(errors)
            prompt = self._retry_prompt(goal, last_error)
        return None

    def _request_raw(self, prompt: str) -> str | None:
        if hasattr(self.brain, "generate_structured"):
            try:
                return self.brain.generate_structured(
                    prompt,
                    skill_specification_json_schema(),
                    max_tokens=700,
                    temperature=0.2,
                    top_p=0.9,
                )
            except NotImplementedError:
                pass
            except Exception:
                return None
        try:
            return self.brain.generate(prompt, max_tokens=700, temperature=0.2, top_p=0.9)
        except Exception:
            return None

    @staticmethod
    def _retry_prompt(goal: str, error: str) -> str:
        return (
            "Return JSON only. Create a safe local Jarvis SkillSpecification for this missing capability.\n"
            "The skill must expose main.py with def run(payload: dict) -> dict.\n"
            "The public_tests entries must use the exact keys the implementation will read from `payload` and "
            "the exact keys it will write into the returned dict, matching the goal's actual inputs/outputs.\n"
            "Use only Python standard library dependencies unless the request absolutely requires more.\n"
            "Do not request credentials, network, browser, email, or external permissions unless the goal explicitly requires them.\n"
            f"Goal: {goal}\n\n"
            f"Your previous response was invalid: {error}\n"
            "Regenerate complete, valid JSON only."
        )

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
                    "input": {"value": "sample"},
                    "expected_keys": ["result"],
                }
            ],
            proposed_file_structure=["main.py"],
        )


def _extract_json(text: str) -> str:
    match = re.search(r"(\{.*\})", text.strip(), flags=re.DOTALL)
    return match.group(1) if match else text
