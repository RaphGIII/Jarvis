from __future__ import annotations

import json
import re
from hashlib import sha256
import time
from typing import Any, Protocol

from environments.coding.actions import ActionCandidate, ActionType
from environments.coding.observation import CodingObservation


class BrainProvider(Protocol):
    provider_name: str
    model_name: str

    def generate(self, prompt: str, *, max_tokens: int = 700, temperature: float = 0.2, top_p: float | None = None) -> str:
        ...

    def generate_coding(self, prompt: str, *, max_tokens: int = 450, temperature: float = 0.6, top_p: float = 0.9) -> str:
        ...

    def think(self, user_prompt: str, max_tokens: int = 700) -> str:
        ...


class ActionGenerator(Protocol):
    def generate(self, goal: str, observation: CodingObservation) -> list[ActionCandidate]:
        ...


class QwenActionGenerator:
    """Uses Qwen as structured high-level candidate generator only."""

    def __init__(
        self,
        brain: BrainProvider,
        num_candidates: int = 4,
        max_tokens: int = 450,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> None:
        self.brain = brain
        self.num_candidates = num_candidates
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._cache: dict[str, list[ActionCandidate]] = {}
        self.last_generation_metadata: dict[str, Any] = {}

    def generate(self, goal: str, observation: CodingObservation) -> list[ActionCandidate]:
        cache_key = sha256(f"{goal}\n{observation.to_text()}".encode("utf-8", errors="ignore")).hexdigest()
        if cache_key in self._cache:
            self.last_generation_metadata = {"cache_key": cache_key, "cache_hit": True, "latency_seconds": 0.0}
            return list(self._cache[cache_key])
        prompt = self._prompt(goal, observation)
        started = time.perf_counter()
        if hasattr(self.brain, "generate_coding"):
            raw = self.brain.generate_coding(prompt, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p)
        elif hasattr(self.brain, "think_coding"):
            raw = self.brain.think_coding(prompt, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p)
        else:
            raw = self.brain.think(prompt, max_tokens=self.max_tokens)
        latency_seconds = time.perf_counter() - started
        candidates, parse_metadata = parse_action_candidates(raw, return_metadata=True)
        result, diversity_metadata = dedupe_action_candidates(candidates, self.num_candidates)
        backfilled = 0
        if len(result) < self.num_candidates:
            result, backfilled = backfill_action_candidates(result, fallback_candidates(observation), self.num_candidates)
        self.last_generation_metadata = {
            "cache_key": cache_key,
            "cache_hit": False,
            "latency_seconds": latency_seconds,
            "provider": getattr(self.brain, "provider_name", self.brain.__class__.__name__),
            "model": getattr(self.brain, "model_name", ""),
            "generated_tokens": getattr(self.brain, "last_metadata", {}).get("generated_tokens"),
            "raw_response_hash": sha256(raw.encode("utf-8", errors="ignore")).hexdigest(),
            "valid_candidates": len(candidates),
            "zero_valid_qwen_candidates": 1 if not candidates else 0,
            "returned_candidates": len(result),
            "fallback_backfill_count": backfilled,
            **parse_metadata,
            **diversity_metadata,
        }
        self._cache[cache_key] = result
        return list(result)

    def _prompt(self, goal: str, observation: CodingObservation) -> str:
        allowed = ", ".join(action.name for action in ActionType)
        return (
            "Return JSON only. You are generating concrete CodingWorld action candidates, not final prose.\n"
            f"Allowed action_type values: {allowed}\n"
            f"Return exactly {self.num_candidates} candidates when possible; otherwise return as many structurally plausible candidates as you can.\n"
            "Use this schema:\n"
            "[{\"reason\":\"short private summary\","
            "\"action_type\":\"PATCH_FILE\","
            "\"path\":\"solution.py\","
            "\"arguments\":{\"old\":\"exact old text\",\"new\":\"exact replacement text\"},"
            "\"expected_effect\":\"what should improve\","
            "\"confidence\":0.5,"
            "\"estimated_cost\":1.0}]\n"
            "For PATCH_FILE, provide concrete old/new text that can be applied exactly.\n"
            "If source content is not visible, do not guess a PATCH_FILE old string. Generate READ_FILE instead.\n"
            "If public tests have not run, RUN_TESTS is reasonable.\n"
            "If relevant source is known, generate concrete patches using exact visible text only.\n"
            "For WRITE_FILE, provide path and content. For READ_FILE/RUN_PYTHON, provide path.\n"
            "Include different strategies: inspect/test/minimal patch/alternative patch when useful.\n"
            "Do not claim private evaluator knowledge and do not modify tests.\n"
            f"Return at most {self.num_candidates} candidates; fewer is acceptable when only fewer actions are structurally plausible.\n"
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


def parse_action_candidates(raw: str, return_metadata: bool = False):
    text = raw.strip()
    match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    metadata: dict[str, Any] = {"malformed_items": 0, "schema_invalid_candidates": 0, "parse_error": "", "parse_error_count": 0}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        metadata["parse_error"] = str(exc)
        metadata["parse_error_count"] = 1
        return ([], metadata) if return_metadata else []
    if isinstance(parsed, dict):
        parsed = parsed.get("candidates", parsed.get("actions", [parsed]))
    if not isinstance(parsed, list):
        metadata["malformed_items"] = 1
        return ([], metadata) if return_metadata else []
    candidates = []
    for item in parsed:
        if isinstance(item, dict):
            try:
                candidates.append(_candidate_from_qwen_item(item))
            except (KeyError, ValueError):
                metadata["malformed_items"] += 1
                metadata["schema_invalid_candidates"] += 1
        else:
            metadata["malformed_items"] += 1
            metadata["schema_invalid_candidates"] += 1
    return (candidates, metadata) if return_metadata else candidates


def _candidate_from_qwen_item(item: dict[str, Any]) -> ActionCandidate:
    raw_type = str(item.get("action_type", "")).strip().upper()
    if raw_type not in ActionType.__members__:
        raise ValueError(f"Unsupported action_type: {raw_type}")
    action_type = ActionType[raw_type]
    arguments = dict(item.get("arguments") or {})
    if "path" in item and "path" not in arguments:
        arguments["path"] = item["path"]
    if action_type == ActionType.PATCH_FILE:
        patch = item.get("patch")
        if isinstance(patch, dict):
            for key in ["old", "new", "content"]:
                if key in patch and key not in arguments:
                    arguments[key] = patch[key]
        for key in ["old", "new", "content"]:
            if key in item and key not in arguments:
                arguments[key] = item[key]
        if isinstance(patch, str) and "patch" not in arguments:
            arguments["patch"] = patch
    if action_type == ActionType.WRITE_FILE and "content" in item and "content" not in arguments:
        arguments["content"] = item["content"]
    if action_type in {ActionType.RUN_TESTS, ActionType.LIST_FILES, ActionType.INSPECT_ERROR, ActionType.FINISH}:
        arguments = {}
    reason = str(item.get("reasoning_summary") or item.get("reason") or "")
    expected = str(item.get("expected_effect") or "")
    summary = reason if not expected else f"{reason} Expected: {expected}".strip()
    return ActionCandidate(
        action_type=action_type,
        arguments=arguments,
        reasoning_summary=summary[:500],
        confidence=_clamp_float(item.get("confidence", 0.5), 0.0, 1.0),
        estimated_cost=_clamp_float(item.get("estimated_cost", _default_cost(action_type)), 0.1, 10.0),
    )


def dedupe_action_candidates(candidates: list[ActionCandidate], limit: int) -> tuple[list[ActionCandidate], dict[str, int]]:
    deduped: list[ActionCandidate] = []
    seen: set[str] = set()
    duplicates = 0
    for candidate in candidates:
        key = _candidate_diversity_key(candidate)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= limit:
            break
    return deduped, {"duplicate_candidates": duplicates}


def backfill_action_candidates(
    candidates: list[ActionCandidate],
    fallback: list[ActionCandidate],
    limit: int,
) -> tuple[list[ActionCandidate], int]:
    result = list(candidates)
    seen = {_candidate_diversity_key(candidate) for candidate in result}
    added = 0
    for candidate in fallback:
        key = _candidate_diversity_key(candidate)
        if key in seen:
            continue
        result.append(candidate)
        seen.add(key)
        added += 1
        if len(result) >= limit:
            break
    return result, added


def _candidate_diversity_key(candidate: ActionCandidate) -> str:
    payload = {
        "action_type": candidate.action_type.name,
        "arguments": candidate.arguments,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = low
    return max(low, min(high, numeric))


def _default_cost(action_type: ActionType) -> float:
    if action_type in {ActionType.READ_FILE, ActionType.LIST_FILES, ActionType.INSPECT_ERROR}:
        return 0.5
    if action_type in {ActionType.RUN_TESTS, ActionType.RUN_PYTHON}:
        return 2.0
    if action_type in {ActionType.PATCH_FILE, ActionType.WRITE_FILE}:
        return 3.0
    return 1.0


def fallback_candidates(observation: CodingObservation) -> list[ActionCandidate]:
    implementation_files = [
        path
        for path in observation.workspace_tree
        if path.endswith(".py") and not path.rsplit("/", 1)[-1].startswith("test")
    ]
    unread_implementation = next((path for path in implementation_files if path not in observation.relevant_file_excerpts), None)
    test_ran = bool(observation.test_state.get("ran", False))
    tests_passing = bool(observation.test_state.get("passed", 0) > 0 and observation.test_state.get("failed", 0) == 0)
    has_error = bool(observation.error_output)
    candidates: list[ActionCandidate] = []
    if unread_implementation:
        candidates.append(ActionCandidate(ActionType.READ_FILE, {"path": unread_implementation}, "Read implementation source before editing.", 0.65, 1.0))
        if not test_ran:
            candidates.append(ActionCandidate(ActionType.RUN_TESTS, reasoning_summary="Establish objective public-test baseline.", confidence=0.55, estimated_cost=2.0))
        if has_error:
            candidates.append(ActionCandidate(ActionType.INSPECT_ERROR, reasoning_summary="Inspect public error output.", confidence=0.45, estimated_cost=0.5))
        if not observation.workspace_tree:
            candidates.append(ActionCandidate(ActionType.LIST_FILES, reasoning_summary="Inspect workspace.", confidence=0.35, estimated_cost=0.5))
    else:
        if not test_ran or not tests_passing:
            candidates.append(ActionCandidate(ActionType.RUN_TESTS, reasoning_summary="Use objective public tests.", confidence=0.55, estimated_cost=2.0))
        if has_error:
            candidates.append(ActionCandidate(ActionType.INSPECT_ERROR, reasoning_summary="Inspect public error output.", confidence=0.5, estimated_cost=0.5))
        if not observation.workspace_tree:
            candidates.append(ActionCandidate(ActionType.LIST_FILES, reasoning_summary="Inspect workspace.", confidence=0.35, estimated_cost=0.5))
    if tests_passing:
        candidates.append(ActionCandidate(ActionType.FINISH, reasoning_summary="Public tests are passing.", confidence=0.5, estimated_cost=0.1))
    if observation.workspace_tree:
        candidates.append(ActionCandidate(ActionType.LIST_FILES, reasoning_summary="Refresh workspace listing if no better unique fallback is available.", confidence=0.2, estimated_cost=0.5))
    if not candidates:
        candidates.append(ActionCandidate(ActionType.LIST_FILES, reasoning_summary="Inspect workspace.", confidence=0.35, estimated_cost=0.5))
    return candidates
