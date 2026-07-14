from __future__ import annotations

from dataclasses import dataclass

import torch

from skate_bfm01.constants import ACTION_RESCALE, ACTION_SCALES, DEFAULT_JOINT_POS, POLICY_JOINT_NAMES


@dataclass(frozen=True)
class ActionMappingReport:
    bfm0_joint_names: tuple[str, ...]
    husky_joint_names: tuple[str, ...]
    shared_joint_names: tuple[str, ...]
    dropped_joint_names: tuple[str, ...]
    mode: str


class Bfm0ToHusky23ActionAdapter:
    """Map BFM0 29D actions to current HUSKY 23D raw actions."""

    def __init__(
        self,
        env,
        action_term_name: str = "joint_pos",
        mode: str = "reference",
        action_clip: float | None = None,
        reference_blend: float = 0.0,
        action_gain: float = 1.25,
    ) -> None:
        if mode not in {"reference", "nominal_aligned", "target_position", "raw_shared"}:
            raise ValueError(f"Unsupported action mapping mode: {mode}")
        self.env = env
        self.mode = mode
        self.action_clip = action_clip
        self.action_term = env.action_manager.get_term(action_term_name)
        self.husky_joint_names = tuple(self.action_term.target_names)
        self.bfm0_joint_names = POLICY_JOINT_NAMES

        bfm_name_to_idx = {name: idx for idx, name in enumerate(self.bfm0_joint_names)}
        husky_name_to_idx = {name: idx for idx, name in enumerate(self.husky_joint_names)}
        shared = tuple(name for name in self.bfm0_joint_names if name in husky_name_to_idx)
        dropped = tuple(name for name in self.bfm0_joint_names if name not in husky_name_to_idx)
        unexpected = tuple(name for name in self.husky_joint_names if name not in bfm_name_to_idx)
        if unexpected:
            raise ValueError(f"HUSKY has joints not found in BFM0 policy order: {unexpected}")

        self.shared_joint_names = shared
        self.dropped_joint_names = dropped
        self._bfm_shared_ids = torch.tensor([bfm_name_to_idx[name] for name in self.husky_joint_names], device=env.device, dtype=torch.long)
        self._bfm_action_scales = torch.tensor(ACTION_SCALES, device=env.device, dtype=torch.float32)
        self._bfm_default_joint_pos = torch.tensor(DEFAULT_JOINT_POS, device=env.device, dtype=torch.float32)
        target_ids = self.action_term.target_ids
        self._husky_default_pos = env.robot.data.default_joint_pos[:, target_ids]
        self.reference_blend = reference_blend
        self.action_gain = action_gain
        scale = self.action_term.scale
        if isinstance(scale, torch.Tensor):
            self._husky_action_scale = scale
        else:
            self._husky_action_scale = torch.full_like(self._husky_default_pos, float(scale))

    @property
    def report(self) -> ActionMappingReport:
        return ActionMappingReport(
            bfm0_joint_names=self.bfm0_joint_names,
            husky_joint_names=self.husky_joint_names,
            shared_joint_names=self.shared_joint_names,
            dropped_joint_names=self.dropped_joint_names,
            mode=self.mode,
        )

    def map_action(self, bfm0_action: torch.Tensor) -> torch.Tensor:
        if bfm0_action.ndim == 1:
            bfm0_action = bfm0_action.unsqueeze(0)
        bfm0_action = bfm0_action.to(device=self.env.device, dtype=torch.float32)
        if bfm0_action.shape[-1] != len(self.bfm0_joint_names):
            raise ValueError(f"Expected BFM0 action dim 29, got {bfm0_action.shape[-1]}")
        if bfm0_action.shape[0] == 1 and self.env.num_envs > 1:
            bfm0_action = bfm0_action.repeat(self.env.num_envs, 1)

        shared_action = bfm0_action[:, self._bfm_shared_ids]
        if self.mode == "raw_shared":
            husky_action = shared_action
        else:
            delta = (
                shared_action
                * self._bfm_action_scales[self._bfm_shared_ids]
                * ACTION_RESCALE
                * self.action_gain
            )
            if self.mode == "nominal_aligned":
                target = self._husky_default_pos + delta
            elif self.mode == "reference":
                bfm_default = self._bfm_default_joint_pos[self._bfm_shared_ids]
                reference = bfm_default + self.reference_blend * (self._husky_default_pos - bfm_default)
                target = reference + delta
            else:
                target = self._bfm_default_joint_pos[self._bfm_shared_ids] + delta
            husky_action = (target - self._husky_default_pos) / torch.clamp(self._husky_action_scale, min=1e-6)

        if self.action_clip is not None:
            husky_action = torch.clamp(husky_action, -self.action_clip, self.action_clip)
        return husky_action

    def set_calibration(self, reference_blend: float, action_gain: float) -> None:
        self.reference_blend = max(0.0, min(float(reference_blend), 1.0))
        self.action_gain = max(0.0, float(action_gain))
