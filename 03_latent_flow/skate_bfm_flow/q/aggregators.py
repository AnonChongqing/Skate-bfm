from __future__ import annotations

import torch


def aggregate(q1: torch.Tensor, q2: torch.Tensor, name: str = "min", beta: float = 0.5) -> torch.Tensor:
    if name == "min":
        return torch.minimum(q1, q2)
    if name == "mean":
        return (q1 + q2) * 0.5
    if name == "mean_minus_std":
        mean = (q1 + q2) * 0.5
        std = torch.sqrt(((q1 - mean).square() + (q2 - mean).square()) * 0.5 + 1e-8)
        return mean - beta * std
    if name == "min_minus_disagreement":
        return torch.minimum(q1, q2) - beta * (q1 - q2).abs()
    raise ValueError(f"Unknown Q aggregation: {name}")
