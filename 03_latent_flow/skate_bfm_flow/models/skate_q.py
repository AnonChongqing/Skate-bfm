from __future__ import annotations

import copy

import torch
from torch import nn

from ..schemas import QInputBatch
from .blocks import mlp


ENCODER_OUTPUT = {
    "robot": 128,
    "board": 64,
    "contact": 64,
    "goal_mode": 64,
    "z_current": 128,
    "z_candidate": 128,
    "flow": 64,
    "preview": 64,
}


class SkateQNetwork(nn.Module):
    def __init__(self, branch_dims: dict[str, int], activation_name: str = "elu", final_hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        final_hidden_dims = final_hidden_dims or [512, 256, 128]
        unknown = set(branch_dims) - set(ENCODER_OUTPUT)
        if unknown:
            raise ValueError(f"Unknown Q branches: {sorted(unknown)}")
        self.branch_dims = dict(branch_dims)
        self.encoders = nn.ModuleDict()
        for name, input_dim in branch_dims.items():
            output_dim = ENCODER_OUTPUT[name]
            middle = 256 if name in {"robot", "z_current", "z_candidate"} else 128
            self.encoders[name] = mlp([input_dim, middle, output_dim], activation_name, layer_norm=True, final_activation=True)
        fusion_dim = sum(ENCODER_OUTPUT[name] for name in branch_dims)
        self.fusion = mlp([fusion_dim, *final_hidden_dims, 1], activation_name, layer_norm=True)

    def forward(self, batch: QInputBatch | dict[str, torch.Tensor]) -> torch.Tensor:
        branches = batch.branch_tensors if isinstance(batch, QInputBatch) else batch
        if set(branches) != set(self.branch_dims):
            raise ValueError(f"Q branch mismatch: expected {set(self.branch_dims)}, got {set(branches)}")
        encoded = [self.encoders[name](branches[name]) for name in self.branch_dims]
        return self.fusion(torch.cat(encoded, dim=-1))


class TwinSkateQ(nn.Module):
    def __init__(self, branch_dims: dict[str, int], activation_name: str = "elu", final_hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        self.q1 = SkateQNetwork(branch_dims, activation_name, final_hidden_dims)
        self.q2 = SkateQNetwork(branch_dims, activation_name, final_hidden_dims)
        if any(first is second for first, second in zip(self.q1.parameters(), self.q2.parameters())):
            raise AssertionError("Twin critics must not share parameters")

    def forward(self, batch: QInputBatch | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(batch), self.q2(batch)

    def make_targets(self) -> "TwinSkateQ":
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        return target
