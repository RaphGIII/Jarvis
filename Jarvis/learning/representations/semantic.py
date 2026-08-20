from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn


class SemanticTextEncoder(Protocol):
    embedding_dim: int

    def encode(self, text: str) -> Tensor:
        ...

    def cache_key(self, text: str) -> str:
        ...


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


class DeterministicTextEncoder:
    """Fast deterministic semantic stand-in for tests and non-Qwen demos."""

    def __init__(self, embedding_dim: int = 64) -> None:
        self.embedding_dim = embedding_dim
        self._cache: dict[str, Tensor] = {}

    def cache_key(self, text: str) -> str:
        return stable_text_hash(text)

    def encode(self, text: str) -> Tensor:
        key = self.cache_key(text)
        if key in self._cache:
            return self._cache[key].clone()
        vector = torch.zeros(self.embedding_dim, dtype=torch.float32)
        tokens = text.lower().replace("\n", " ").split()
        for token in tokens or [""]:
            digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.embedding_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = vector.norm().clamp_min(1e-6)
        vector = vector / norm
        self._cache[key] = vector
        return vector.clone()


class QwenHiddenStateTextEncoder:
    """Frozen local Qwen hidden-state encoder using an already loaded JarvisBrain."""

    def __init__(self, brain) -> None:
        self.brain = brain
        self.tokenizer = brain.tokenizer
        self.model = brain.model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.embedding_dim = int(getattr(self.model.config, "hidden_size", getattr(self.model.config, "d_model", 0)))
        if self.embedding_dim <= 0:
            raise ValueError("Could not infer Qwen hidden size from model.config")
        self._cache: dict[str, Tensor] = {}

    def cache_key(self, text: str) -> str:
        return stable_text_hash(text)

    def encode(self, text: str) -> Tensor:
        key = self.cache_key(text)
        if key in self._cache:
            return self._cache[key].clone()
        device = self.model.device
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.inference_mode():
            output = self.model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[-1]
            mask = inputs.get("attention_mask")
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                weights = mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        vector = pooled.squeeze(0).detach().float().cpu()
        self._cache[key] = vector
        return vector.clone()


@dataclass
class SemanticObservationFeatures:
    semantic: Tensor
    numeric: Tensor
    cache_key: str

    def to_metadata(self) -> dict[str, object]:
        return {
            "semantic": self.semantic.detach().cpu().tolist(),
            "numeric": self.numeric.detach().cpu().tolist(),
            "cache_key": self.cache_key,
        }


class ProjectionEncoder(nn.Module):
    """Trainable projection from frozen semantic features plus numeric features."""

    def __init__(self, semantic_dim: int, numeric_dim: int, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.semantic_dim = semantic_dim
        self.numeric_dim = numeric_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(semantic_dim + numeric_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.training_step = 0

    def forward(self, semantic: Tensor, numeric: Tensor) -> Tensor:
        if semantic.dim() == 1:
            semantic = semantic.unsqueeze(0)
        if numeric.dim() == 1:
            numeric = numeric.unsqueeze(0)
        features = torch.cat([semantic.float(), numeric.float()], dim=-1)
        return self.net(features)

    def encode_features(self, features: SemanticObservationFeatures) -> Tensor:
        return self.forward(features.semantic, features.numeric)
