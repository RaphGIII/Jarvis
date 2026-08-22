from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from learning.representations.embeddings import one_hot
from learning.world_model.model import WorldModel, WorldModelOutput


@dataclass
class WorldModelPredictor:
    model: WorldModel

    def predict_action_index(self, latent_state: Tensor, action_index: int) -> WorldModelOutput:
        action = one_hot(action_index, self.model.config.action_dim).to(latent_state.device)
        return self.model(latent_state, action)
