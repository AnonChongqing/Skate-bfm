from __future__ import annotations

import torch
from torch import nn

from .batch_action_adapter import BatchActionAdapter
from .frozen_policy import FrozenBfmPolicy


class FrozenBfmActionPreview(nn.Module):
    def __init__(self, policy: FrozenBfmPolicy, adapter: BatchActionAdapter, preview_type: str = "action_23d") -> None:
        super().__init__()
        self.policy = policy
        self.adapter = adapter
        self.preview_type = preview_type

    @torch.no_grad()
    def forward(self, bfm_obs: dict[str, torch.Tensor], z_candidate: torch.Tensor) -> torch.Tensor:
        action29 = self.policy.act(bfm_obs, z_candidate)
        if self.preview_type == "none":
            return action29.new_empty(action29.shape[0], 0)
        if self.preview_type == "action_29d":
            return action29
        action23 = self.adapter(action29)
        if self.preview_type in {"action_23d", "first_action_23d"}:
            return action23
        if self.preview_type == "lower_body_12d":
            lower_ids = [int((self.adapter.shared_ids == index).nonzero()[0]) for index in range(12)]
            return action23[:, lower_ids]
        raise ValueError(f"Unknown preview type: {self.preview_type}")
