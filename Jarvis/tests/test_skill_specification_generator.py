from __future__ import annotations

from capabilities.specification import SkillSpecificationGenerator


class _FlakyBrain:
    """Emits malformed JSON on the first call(s), then a valid spec."""

    def __init__(self, bad_responses: list[str], good_response: str) -> None:
        self._responses = list(bad_responses) + [good_response]
        self.calls = 0

    def generate_structured(self, prompt, schema, *, max_tokens=700, temperature=0.2, top_p=0.9):
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


class _AlwaysBadBrain:
    def __init__(self) -> None:
        self.calls = 0

    def generate_structured(self, prompt, schema, *, max_tokens=700, temperature=0.2, top_p=0.9):
        self.calls += 1
        return "not json at all {{{"


GOOD_SPEC = """
{
  "capability_id": "local.double.an.integer.number",
  "objective": "Double an integer number.",
  "functional_requirements": ["Multiply the input integer by two."],
  "inputs": {"type": "object", "additionalProperties": true},
  "outputs": {"type": "object", "additionalProperties": true},
  "constraints": ["Use Python standard library only.", "Expose run(payload: dict) -> dict."],
  "allowed_dependencies": [],
  "permissions": [],
  "acceptance_criteria": ["Public examples pass."],
  "public_tests": [{"name": "doubles", "input": {"number": 3}, "expected": {"result": 6}}],
  "proposed_file_structure": ["main.py"]
}
"""


def test_spec_generator_retries_after_malformed_json_and_succeeds():
    brain = _FlakyBrain(bad_responses=["not json", "{\"files\": [}"], good_response=GOOD_SPEC)
    generator = SkillSpecificationGenerator(brain=brain, attempts=3)

    spec = generator.generate("Double an integer number.")

    assert brain.calls == 3
    assert spec.capability_id == "local.double.an.integer.number"
    assert spec.public_tests == [{"name": "doubles", "input": {"number": 3}, "expected": {"result": 6}}]


def test_spec_generator_falls_back_to_generic_spec_when_brain_never_recovers():
    brain = _AlwaysBadBrain()
    generator = SkillSpecificationGenerator(brain=brain, attempts=2)

    spec = generator.generate("Double an integer number.")

    assert brain.calls == 2
    # Falls back to the generic placeholder spec only after exhausting retries.
    assert spec.public_tests == [
        {"name": "returns_object", "input": {"value": "sample"}, "expected_keys": ["result"]}
    ]


def test_spec_generator_uses_first_valid_response_without_extra_calls():
    brain = _FlakyBrain(bad_responses=[], good_response=GOOD_SPEC)
    generator = SkillSpecificationGenerator(brain=brain, attempts=3)

    spec = generator.generate("Double an integer number.")

    assert brain.calls == 1
    assert spec.objective == "Double an integer number."


def test_spec_generator_rejects_public_tests_missing_input_object():
    # A public test with no usable "input" dict (e.g. {"value": "10"}) is
    # structurally invalid: the rendered test harness would call
    # main.run({}), which can never exercise the real behavior.
    bad_spec = (
        GOOD_SPEC.replace(
            '"public_tests": [{"name": "doubles", "input": {"number": 3}, "expected": {"result": 6}}]',
            '"public_tests": [{"value": "10"}]',
        )
    )
    brain = _FlakyBrain(bad_responses=[bad_spec], good_response=GOOD_SPEC)
    generator = SkillSpecificationGenerator(brain=brain, attempts=2)

    spec = generator.generate("Double an integer number.")

    assert brain.calls == 2
    assert spec.public_tests == [{"name": "doubles", "input": {"number": 3}, "expected": {"result": 6}}]
