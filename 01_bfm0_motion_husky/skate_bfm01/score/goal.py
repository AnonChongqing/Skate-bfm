from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab_husky.tasks.skater import mdp


@dataclass(frozen=True)
class GoalCfg:
    pos_std: float = 0.05**0.5
    rot_std: float = 0.10**0.5


def transition_goal(env, cfg: GoalCfg = GoalCfg()) -> dict[str, torch.Tensor]:
    """Track HUSKY's planned key-body goal only during transition phases."""
    pos = mdp.transition_body_pos_tracking(env, std=cfg.pos_std)
    rot = mdp.transition_body_rot_tracking(env, std=cfg.rot_std)
    in_transition = env._get_transition_target_b()[2].float()
    return {
        "goal_pos": pos,
        "goal_rot": rot,
        "transition_goal": 0.5 * (pos + rot),
        "transition_active": in_transition,
    }
