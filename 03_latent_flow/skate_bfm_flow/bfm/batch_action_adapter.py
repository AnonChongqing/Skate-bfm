from __future__ import annotations

import torch
from torch import nn

from .constants import ACTION_RESCALE, ACTION_SCALES, DEFAULT_JOINT_POS, POLICY_JOINT_NAMES


class BatchActionAdapter(nn.Module):
    """Pure-tensor version of the live 29D BFM to 23D HUSKY adapter."""

    def __init__(
        self,
        shared_ids: torch.Tensor,
        husky_default: torch.Tensor,
        husky_scale: torch.Tensor,
        mode: str = "reference",
        reference_blend: float = 0.0,
        action_gain: float = 1.25,
        action_clip: float | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"reference", "nominal_aligned", "target_position", "raw_shared"}:
            raise ValueError(f"Unsupported adapter mode: {mode}")
        self.mode = mode
        self.reference_blend = float(reference_blend)
        self.action_gain = float(action_gain)
        self.action_clip = action_clip
        self.register_buffer("shared_ids", shared_ids.long().clone())
        self.register_buffer("bfm_scales", ACTION_SCALES.clone())
        self.register_buffer("bfm_default", DEFAULT_JOINT_POS.clone())
        self.register_buffer("husky_default", husky_default.reshape(-1).float().clone())
        self.register_buffer("husky_scale", husky_scale.reshape(-1).float().clone())

    @classmethod
    def from_env(cls, env, **kwargs) -> "BatchActionAdapter":
        term = env.action_manager.get_term("joint_pos")
        ids = torch.tensor([POLICY_JOINT_NAMES.index(name) for name in term.target_names], device=env.device)
        target_ids = term.target_ids
        default = env.robot.data.default_joint_pos[0, target_ids]
        scale = term.scale
        if not isinstance(scale, torch.Tensor):
            scale = torch.full_like(default, float(scale))
        elif scale.ndim > 1:
            scale = scale[0]
        return cls(ids, default, scale, **kwargs).to(env.device)

    def forward(self, action_29d: torch.Tensor) -> torch.Tensor:
        if action_29d.shape[-1] != 29:
            raise ValueError(f"Expected [...,29] BFM action, got {tuple(action_29d.shape)}")
        shared = action_29d.index_select(-1, self.shared_ids)
        if self.mode == "raw_shared":
            output = shared
        else:
            delta = shared * self.bfm_scales[self.shared_ids] * ACTION_RESCALE * self.action_gain
            if self.mode == "nominal_aligned":
                reference = self.husky_default
            else:
                bfm_reference = self.bfm_default[self.shared_ids]
                if self.mode == "reference":
                    reference = bfm_reference + self.reference_blend * (self.husky_default - bfm_reference)
                else:
                    reference = bfm_reference
            output = (reference + delta - self.husky_default) / self.husky_scale.clamp_min(1e-6)
        if self.action_clip is not None:
            output = output.clamp(-self.action_clip, self.action_clip)
        return output
