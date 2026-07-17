from __future__ import annotations

import torch
import torch.nn.functional as F


def critic_loss(prediction: torch.Tensor, target: torch.Tensor, name: str = "huber", delta: float = 1.0) -> torch.Tensor:
    if name == "huber":
        return F.huber_loss(prediction, target, delta=delta)
    if name == "mse":
        return F.mse_loss(prediction, target)
    if name == "mae":
        return F.l1_loss(prediction, target)
    raise ValueError(f"Unknown critic loss: {name}")
