from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from .blocks import finite, mlp


@dataclass
class FlowSample:
    action: torch.Tensor
    log_prob: torch.Tensor
    mean_action: torch.Tensor


class FlowPolicy(nn.Module):
    """Frame-stacked tanh Gaussian policy over low-dimensional latent flow."""

    def __init__(
        self,
        frame_dim: int,
        flow_dim: int,
        frame_stack: int = 5,
        hidden_dims: list[int] | None = None,
        activation_name: str = "elu",
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128]
        self.frame_dim = frame_dim
        self.frame_stack = frame_stack
        self.flow_dim = flow_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.backbone = mlp([frame_dim * frame_stack, *hidden_dims], activation_name, layer_norm=True, final_activation=True)
        self.mean_head = nn.Linear(hidden_dims[-1], flow_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], flow_dim)

    def forward(self, stacked_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if stacked_obs.shape[-1] != self.frame_dim * self.frame_stack:
            raise ValueError(f"Expected actor obs dim {self.frame_dim * self.frame_stack}, got {stacked_obs.shape[-1]}")
        hidden = self.backbone(finite("flow_actor_obs", stacked_obs))
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, stacked_obs: torch.Tensor, deterministic: bool = False) -> FlowSample:
        mean, log_std = self(stacked_obs)
        distribution = Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        log_prob = distribution.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)
        return FlowSample(action=action, log_prob=log_prob.sum(-1, keepdim=True), mean_action=torch.tanh(mean))
