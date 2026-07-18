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


def pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    margin: float,
    min_target_gap: float,
) -> torch.Tensor:
    target_gap = target.unsqueeze(2) - target.unsqueeze(1)
    prediction_gap = prediction.unsqueeze(2) - prediction.unsqueeze(1)
    valid = target_gap.abs() >= min_target_gap
    valid &= torch.triu(torch.ones_like(valid, dtype=torch.bool), diagonal=1)
    if not valid.any():
        return prediction.sum() * 0.0
    signed_gap = target_gap.sign() * prediction_gap
    return F.relu(margin - signed_gap)[valid].mean()


def failure_margin_loss(
    prediction: torch.Tensor,
    failure: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    safe_over_failure = prediction.unsqueeze(2) - prediction.unsqueeze(1)
    valid = torch.logical_and((~failure).unsqueeze(2), failure.unsqueeze(1))
    if not valid.any():
        return prediction.sum() * 0.0
    return F.relu(margin - safe_over_failure)[valid].mean()
