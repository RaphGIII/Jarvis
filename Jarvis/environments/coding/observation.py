from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor

from learning.representations.semantic import SemanticObservationFeatures, SemanticTextEncoder


@dataclass
class CodingObservation:
    task_description: str
    workspace_tree: list[str]
    relevant_file_excerpts: dict[str, str] = field(default_factory=dict)
    latest_action: dict[str, Any] | None = None
    latest_action_result: str = ""
    test_state: dict[str, Any] = field(default_factory=dict)
    error_output: str = ""
    step_number: int = 0
    remaining_budget: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_text(self) -> str:
        excerpts = "\n".join(f"{path}:\n{text}" for path, text in self.relevant_file_excerpts.items())
        return "\n".join(
            [
                f"Task: {self.task_description}",
                f"Step: {self.step_number}",
                f"Remaining: {self.remaining_budget}",
                "Tree:",
                "\n".join(self.workspace_tree),
                f"Latest action: {self.latest_action}",
                f"Latest result: {self.latest_action_result}",
                f"Tests: {self.test_state}",
                f"Errors: {self.error_output[:1200]}",
                "Excerpts:",
                excerpts,
            ]
        )


class ObservationAdapter:
    """Converts CodingObservation to numeric features for ObservationEncoder."""

    def __init__(self, feature_dim: int = 24) -> None:
        if feature_dim < 16:
            raise ValueError("feature_dim must be at least 16")
        self.feature_dim = feature_dim

    def encode(self, observation: CodingObservation) -> Tensor:
        return self.numeric_features(observation)

    def numeric_features(self, observation: CodingObservation) -> Tensor:
        tree = observation.workspace_tree
        excerpts = observation.relevant_file_excerpts
        tests = observation.test_state
        latest = observation.latest_action or {}
        latest_type = str(latest.get("action_type", ""))
        latest_index = self._stable_bucket(latest_type, buckets=9)
        error = observation.error_output or ""

        features = torch.zeros(self.feature_dim, dtype=torch.float32)
        features[0] = min(1.0, len(tree) / 30.0)
        features[1] = min(1.0, len(excerpts) / 8.0)
        features[2] = min(1.0, observation.step_number / 20.0)
        features[3] = min(1.0, observation.remaining_budget / 20.0)
        features[4] = float(tests.get("passed", 0)) / max(1.0, float(tests.get("total", 1)))
        features[5] = float(tests.get("failed", 0)) / max(1.0, float(tests.get("total", 1)))
        features[6] = 1.0 if tests.get("last_return_code", 0) == 0 and tests.get("ran", False) else 0.0
        features[7] = 1.0 if error else 0.0
        features[8] = 1.0 if "syntaxerror" in error.lower() else 0.0
        features[9] = 1.0 if "assert" in error.lower() or "failed" in error.lower() else 0.0
        features[10] = min(1.0, len(observation.task_description) / 240.0)
        features[11] = min(1.0, len(observation.latest_action_result) / 800.0)
        features[12] = latest_index / 8.0
        features[13] = 1.0 if any(path.endswith(".py") for path in tree) else 0.0
        features[14] = 1.0 if excerpts else 0.0
        features[15] = 1.0 if tests.get("ran", False) else 0.0

        text_hash = self._stable_bucket(observation.to_text(), buckets=max(1, self.feature_dim - 16))
        if self.feature_dim > 16:
            features[16 + text_hash] = 1.0
        return features

    def semantic_text(self, observation: CodingObservation) -> str:
        return observation.to_text()

    def encode_semantic(
        self,
        observation: CodingObservation,
        text_encoder: SemanticTextEncoder,
    ) -> SemanticObservationFeatures:
        text = self.semantic_text(observation)
        return SemanticObservationFeatures(
            semantic=text_encoder.encode(text),
            numeric=self.numeric_features(observation),
            cache_key=text_encoder.cache_key(text),
        )

    @staticmethod
    def _stable_bucket(text: str, buckets: int) -> int:
        if buckets <= 1:
            return 0
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        return int(digest[:8], 16) % buckets
