from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class EWCAnchor:
    parameter_name: str
    parameter_value: Tensor
    fisher_information: Tensor


def ewc_penalty(current_parameters: dict[str, Tensor], anchors: list[EWCAnchor]) -> Tensor:
    if not anchors:
        return torch.tensor(0.0)
    penalties = []
    for anchor in anchors:
        if anchor.parameter_name in current_parameters:
            delta = current_parameters[anchor.parameter_name] - anchor.parameter_value
            penalties.append(torch.sum(anchor.fisher_information * delta.pow(2)))
    if not penalties:
        return torch.tensor(0.0)
    return torch.stack(penalties).sum()
