from __future__ import annotations

import torch
from torch import nn


class RunningNormalizer(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-4, clip: float = 10.0) -> None:
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        batch_mean = value.mean(0)
        batch_var = value.var(0, unbiased=False)
        batch_count = torch.tensor(value.shape[0], device=value.device, dtype=value.dtype)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return ((value - self.mean) / torch.sqrt(self.var + 1e-6)).clamp(-self.clip, self.clip)
