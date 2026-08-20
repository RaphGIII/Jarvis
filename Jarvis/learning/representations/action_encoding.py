from __future__ import annotations

import json

import torch
from torch import Tensor, nn

from environments.coding.actions import ActionCandidate, ActionType
from learning.representations.semantic import SemanticTextEncoder


class SemanticActionEncoder(nn.Module):
    """Encodes concrete actions, including paths and patch text, into trainable embeddings."""

    def __init__(
        self,
        text_encoder: SemanticTextEncoder,
        action_embedding_dim: int,
        hidden_dim: int = 128,
        numeric_dim: int = 6,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.num_actions = len(ActionType)
        self.semantic_dim = text_encoder.embedding_dim
        self.numeric_dim = numeric_dim
        self.raw_dim = self.num_actions + 3 * self.semantic_dim + self.numeric_dim
        self.action_embedding_dim = action_embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(self.raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_embedding_dim),
        )
        self.training_step = 0

    def raw_features(self, action_candidate: ActionCandidate) -> Tensor:
        return self.raw_features_batch([action_candidate]).squeeze(0)

    def raw_features_batch(self, action_candidates: list[ActionCandidate]) -> Tensor:
        if not action_candidates:
            return torch.empty((0, self.raw_dim), dtype=torch.float32)
        vectors = torch.zeros((len(action_candidates), self.raw_dim), dtype=torch.float32)
        texts: list[str] = []
        numeric_rows: list[Tensor] = []
        for row, action_candidate in enumerate(action_candidates):
            arguments = action_candidate.arguments
            vectors[row, action_candidate.action_index] = 1.0
            path_text = str(arguments.get("path", ""))
            args_text = json.dumps(arguments, sort_keys=True)
            patch_text = "\n".join(
                [
                    str(arguments.get("old", "")),
                    "=>",
                    str(arguments.get("new", "")),
                    str(arguments.get("content", ""))[:2000],
                ]
            )
            texts.extend([path_text, args_text, patch_text])
            numeric_rows.append(
                torch.tensor(
                    [
                        max(0.0, min(1.0, action_candidate.confidence)),
                        min(1.0, max(0.0, action_candidate.estimated_cost / 10.0)),
                        min(1.0, len(path_text) / 160.0),
                        min(1.0, len(str(arguments.get("old", ""))) / 1000.0),
                        min(1.0, len(str(arguments.get("new", ""))) / 1000.0),
                        min(1.0, len(arguments) / 8.0),
                    ],
                    dtype=torch.float32,
                )
            )
        if hasattr(self.text_encoder, "encode_batch"):
            encoded_texts = self.text_encoder.encode_batch(texts)
        else:
            encoded_texts = torch.stack([self.text_encoder.encode(text) for text in texts])
        encoded_texts = encoded_texts.reshape(len(action_candidates), 3, self.semantic_dim)
        offset = self.num_actions
        vectors[:, offset : offset + self.semantic_dim] = encoded_texts[:, 0, :]
        offset += self.semantic_dim
        vectors[:, offset : offset + self.semantic_dim] = encoded_texts[:, 1, :]
        offset += self.semantic_dim
        vectors[:, offset : offset + self.semantic_dim] = encoded_texts[:, 2, :]
        offset += self.semantic_dim
        vectors[:, offset : offset + self.numeric_dim] = torch.stack(numeric_rows)
        return vectors

    def _legacy_raw_features(self, action_candidate: ActionCandidate) -> Tensor:
        arguments = action_candidate.arguments
        vector = torch.zeros(self.raw_dim, dtype=torch.float32)
        vector[action_candidate.action_index] = 1.0
        path_text = str(arguments.get("path", ""))
        args_text = json.dumps(arguments, sort_keys=True)
        patch_text = "\n".join(
            [
                str(arguments.get("old", "")),
                "=>",
                str(arguments.get("new", "")),
                str(arguments.get("content", ""))[:2000],
            ]
        )
        offset = self.num_actions
        vector[offset : offset + self.semantic_dim] = self.text_encoder.encode(path_text)
        offset += self.semantic_dim
        vector[offset : offset + self.semantic_dim] = self.text_encoder.encode(args_text)
        offset += self.semantic_dim
        vector[offset : offset + self.semantic_dim] = self.text_encoder.encode(patch_text)
        offset += self.semantic_dim
        numeric = torch.tensor(
            [
                max(0.0, min(1.0, action_candidate.confidence)),
                min(1.0, max(0.0, action_candidate.estimated_cost / 10.0)),
                min(1.0, len(path_text) / 160.0),
                min(1.0, len(str(arguments.get("old", ""))) / 1000.0),
                min(1.0, len(str(arguments.get("new", ""))) / 1000.0),
                min(1.0, len(arguments) / 8.0),
            ],
            dtype=torch.float32,
        )
        vector[offset : offset + self.numeric_dim] = numeric
        return vector

    def forward_from_raw(self, raw_features: Tensor) -> Tensor:
        if raw_features.dim() == 1:
            raw_features = raw_features.unsqueeze(0)
        return self.projection(raw_features.float())

    def encode(self, action_candidate: ActionCandidate) -> Tensor:
        return self.forward_from_raw(self.raw_features(action_candidate)).squeeze(0)
