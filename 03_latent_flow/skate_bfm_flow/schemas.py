from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor
FEATURE_SCHEMA_VERSION = "skate-flow-v2"


@dataclass
class FeatureBatch:
    actor_frame: Tensor
    critic_robot: Tensor
    critic_board: Tensor
    critic_contact: Tensor
    critic_goal_mode: Tensor
    mode_id: Tensor


@dataclass
class LatentMapOutput:
    z_candidate: Tensor
    raw_direction: Tensor
    tangent_direction: Tensor
    delta_norm: Tensor
    cosine: Tensor


@dataclass
class QInputBatch:
    branch_tensors: dict[str, Tensor]
    batch_size: int


@dataclass
class TargetOutput:
    target: Tensor
    bootstrap: Tensor


@dataclass
class SkateRewardOutput:
    total_low_level: Tensor
    components: dict[str, Tensor]
    gates: dict[str, Tensor]


@dataclass
class MacroStepResult:
    actor_obs: Tensor
    features: FeatureBatch
    bfm_obs: dict[str, Tensor]
    z_current: Tensor
    previous_flow: Tensor
    reward_macro: Tensor
    reward_components: Tensor
    terminated: Tensor
    truncated: Tensor
    diagnostics: dict[str, Tensor]
