from __future__ import annotations

from environments.coding.actions import ActionCandidate, ActionType
from environments.coding.observation import CodingObservation
from runtime.action_generator import fallback_candidates


class GenericMutationActionGenerator:
    """Generic candidate generator for local demos; it does not inspect expected answers."""

    def __init__(self, num_candidates: int = 6) -> None:
        self.num_candidates = num_candidates

    def generate(self, goal: str, observation: CodingObservation) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []
        tests = observation.test_state
        if not tests.get("ran", False):
            candidates.append(ActionCandidate(ActionType.RUN_TESTS, reasoning_summary="Measure current behavior.", confidence=0.7, estimated_cost=2.0))
        first_py = next((path for path in observation.workspace_tree if path.endswith(".py") and not path.startswith("test_")), None)
        if first_py and first_py not in observation.relevant_file_excerpts:
            candidates.append(ActionCandidate(ActionType.READ_FILE, {"path": first_py}, "Inspect implementation.", 0.7, 1.0))
        for path, excerpt in observation.relevant_file_excerpts.items():
            candidates.extend(self._mutations(path, excerpt))
        if tests.get("ran", False) and tests.get("passed", 0) > 0 and tests.get("failed", 0) == 0:
            candidates.append(ActionCandidate(ActionType.FINISH, reasoning_summary="Public tests pass.", confidence=0.9, estimated_cost=0.2))
        candidates.extend(fallback_candidates(observation))
        deduped: list[ActionCandidate] = []
        seen = set()
        for candidate in candidates:
            key = (candidate.action_type, tuple(sorted(candidate.arguments.items())))
            if key not in seen:
                deduped.append(candidate)
                seen.add(key)
        return deduped[: self.num_candidates]

    def _mutations(self, path: str, excerpt: str) -> list[ActionCandidate]:
        candidates = []
        generic_replacements = [
            ("retun ", "return "),
            ("return a - b", "return a + b"),
            ("return x - y", "return x + y"),
            ("return left - right", "return left + right"),
            ("return a + b", "return a - b"),
            ("return x + y", "return x - y"),
            ("return value == None", "return value is None"),
            ("if value ==", "if value !="),
            ("if value !=", "if value =="),
        ]
        for old, new in generic_replacements:
            if old in excerpt:
                candidates.append(
                    ActionCandidate(
                        ActionType.PATCH_FILE,
                        {"path": path, "old": old, "new": new},
                        "Try a generic small code mutation.",
                        0.55,
                        1.0,
                    )
                )
        if excerpt and not candidates:
            candidates.append(ActionCandidate(ActionType.RUN_TESTS, reasoning_summary="Re-run objective tests.", confidence=0.45, estimated_cost=2.0))
        return candidates
