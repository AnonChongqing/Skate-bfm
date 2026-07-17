from __future__ import annotations

import math

import torch
from torch import nn

from ..schemas import LatentMapOutput


class LatentMapper(nn.Module):
    def __init__(self, basis: torch.Tensor, step_size: float = 0.25, update_type: str = "tangent_residual") -> None:
        super().__init__()
        if basis.ndim != 3:
            raise ValueError("basis must have shape [modes,z_dim,flow_dim]")
        self.register_buffer("basis", basis.float().clone())
        self.step_size = float(step_size)
        self.update_type = update_type
        self.radius = math.sqrt(basis.shape[1])

    @staticmethod
    def project(z: torch.Tensor, radius: float) -> torch.Tensor:
        return radius * z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, z_current: torch.Tensor, mode_id: torch.Tensor, flow: torch.Tensor) -> LatentMapOutput:
        if z_current.ndim != 2 or z_current.shape[1] != self.basis.shape[1]:
            raise ValueError("z_current shape does not match basis")
        if flow.shape != (z_current.shape[0], self.basis.shape[2]):
            raise ValueError("flow shape does not match batch/flow_dim")
        chosen = self.basis.index_select(0, mode_id.long())
        raw = torch.bmm(chosen, flow.unsqueeze(-1)).squeeze(-1)
        radial = (raw * z_current).sum(-1, keepdim=True) / z_current.square().sum(-1, keepdim=True).clamp_min(1e-8)
        tangent = raw - radial * z_current
        if self.update_type == "tangent_residual":
            candidate_raw = z_current + self.step_size * tangent
        elif self.update_type == "euclidean_residual":
            candidate_raw = z_current + self.step_size * raw
        elif self.update_type == "prototype_residual":
            candidate_raw = z_current + self.step_size * raw
        elif self.update_type == "direct_z":
            candidate_raw = raw
        else:
            raise ValueError(f"Unknown latent update type: {self.update_type}")
        candidate = self.project(candidate_raw, self.radius)
        cosine = torch.nn.functional.cosine_similarity(z_current, candidate, dim=-1, eps=1e-8)
        return LatentMapOutput(
            z_candidate=candidate,
            raw_direction=raw,
            tangent_direction=tangent,
            delta_norm=(candidate - z_current).norm(dim=-1),
            cosine=cosine,
        )
