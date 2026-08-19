from __future__ import annotations

import torch
from torch import Tensor, nn

from learning.config import DEFAULT_CONFIG


class ObservationEncoder(nn.Module):
    """Trainable observation encoder f_theta(o, memory) -> z.

    The first implementation consumes numeric features. Higher-level adapters can
    turn text, tool traces, and memory records into numeric features later.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = DEFAULT_CONFIG.latent_dim,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, observation_features: Tensor) -> Tensor:
        features = observation_features.float()
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.net(features)

    def encode(self, observation_features: Tensor) -> Tensor:
        return self.forward(observation_features)


class ObservationAutoencoder(nn.Module):
    """Small numeric autoencoder for representation learning experiments."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = DEFAULT_CONFIG.latent_dim,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = ObservationEncoder(input_dim, latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z

    def reconstruction_loss(self, x: Tensor) -> Tensor:
        reconstruction, _ = self.forward(x)
        return torch.mean((reconstruction - x.float()) ** 2)
