from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any, Protocol

from environments.coding.actions import ActionCandidate, ActionType
from environments.coding.observation import CodingObservation


class BrainProvider(Protocol):
    def think(self, user_prompt: str, max_tokens: int = 700) -> str:
        ...


class ActionGenerator(Protocol):
    def generate(self, goal: str, observation: CodingObservation) -> list[ActionCandidate]:
        ...


class QwenActionGenerator:
    """Uses Qwen as structured high-level candidate generator only."""

    def __init__(self, brain: BrainProvider, num_candidates: int = 4) -> None:
        self.brain = brain
        self.num_candidates = num_candidates
        self._cache: dict[str, list[ActionCandidate]] = {}

    def generate(self, goal: str, observation: CodingObservation) -> list[ActionCandidate]:
        cache_key = sha256(f"{goal}\n{observation.to_text()}".encode("utf-8", errors="ignore")).hexdigest()
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        prompt = self._prompt(goal, observation)
        raw = self.brain.think(prompt, max_tokens=900)
        candidates = parse_action_candidates(raw)
        result = candidates[: self.num_candidates] or fallback_candidates(observation)
        self._cache[cache_key] = result
        return list(result)

    def _prompt(self, goal: str, observation: CodingObservation) -> str:
        allowed = ", ".join(action.name for action in ActionType)
        return (
            "Return JSON only. Generate plausible CodingWorld action candidates.\n"
            f"Allowed action_type values: {allowed}\n"
            "Schema: [{\"action_type\":\"READ_FILE\",\"arguments\":{\"path\":\"main.py\"},"
            "\"reasoning_summary\":\"short\",\"confidence\":0.5,\"estimated_cost\":1.0}]\n"
            f"Return at most {self.num_candidates} candidates.\n"
            f"Goal: {goal}\n"
            f"Observation:\n{observation.to_text()}"
        )


class HeuristicCodingActionGenerator:
    """Deterministic non-LLM generator for tests and local demos."""

    def __init__(self, num_candidates: int = 4) -> None:
        self.num_candidates = num_candidates

    def generate(self, goal: str, observation: CodingObservation) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []
        tree = observation.workspace_tree
        py_files = [path for path in tree if path.endswith(".py") and not path.startswith("test")]
        test_ran = bool(observation.test_state.get("ran", False))
        tests_passed = bool(observation.test_state.get("passed", 0) > 0 and observation.test_state.get("failed", 0) == 0)
        excerpts = observation.relevant_file_excerpts

        if tests_passed:
            candidates.append(ActionCandidate(ActionType.FINISH, reasoning_summary="Tests are passing.", confidence=0.9, estimated_cost=0.2))

        if not test_ran:
            candidates.append(ActionCandidate(ActionType.RUN_TESTS, reasoning_summary="Establish objective baseline.", confidence=0.7, estimated_cost=2.0))

        if py_files and not excerpts:
            candidates.append(ActionCandidate(ActionType.READ_FILE, {"path": py_files[0]}, "Inspect likely implementation file.", 0.65, 1.0))

        if test_ran and not tests_passed:
            candidates.append(ActionCandidate(ActionType.INSPECT_ERROR, reasoning_summary="Inspect latest failing output.", confidence=0.55, estimated_cost=0.5))
            candidates.append(ActionCandidate(ActionType.RUN_TESTS, reasoning_summary="Re-run tests after a change.", confidence=0.45, estimated_cost=2.0))

        fallback = fallback_candidates(observation)
        if excerpts:
            fallback = [
                candidate
                for candidate in fallback
                if candidate.action_type not in {ActionType.LIST_FILES, ActionType.READ_FILE}
            ]
        candidates.extend(fallback)
        deduped: list[ActionCandidate] = []
        seen = set()
        for candidate in candidates:
            key = (candidate.action_type, tuple(sorted(candidate.arguments.items())))
            if key not in seen:
                deduped.append(candidate)
                seen.add(key)
        return deduped[: self.num_candidates]


def parse_action_candidates(raw: str) -> list[ActionCandidate]:
    text = raw.strip()
    match = re.search(r"(\[.*\])", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    candidates = []
    for item in parsed:
        if isinstance(item, dict):
            try:
                candidates.append(ActionCandidate.from_dict(item))
            except (KeyError, ValueError):
                continue
    return candidates


def fallback_candidates(observation: CodingObservation) -> list[ActionCandidate]:
    first_file = next((path for path in observation.workspace_tree if path.endswith(".py")), None)
    candidates = [
        ActionCandidate(ActionType.LIST_FILES, reasoning_summary="Inspect workspace.", confidence=0.4, estimated_cost=0.5),
        ActionCandidate(ActionType.RUN_TESTS, reasoning_summary="Use objective tests.", confidence=0.4, estimated_cost=2.0),
    ]
    if first_file:
        candidates.append(ActionCandidate(ActionType.READ_FILE, {"path": first_file}, "Read a Python file.", 0.4, 1.0))
    candidates.append(ActionCandidate(ActionType.FINISH, reasoning_summary="Finish if solved.", confidence=0.1, estimated_cost=0.1))
    return candidates
