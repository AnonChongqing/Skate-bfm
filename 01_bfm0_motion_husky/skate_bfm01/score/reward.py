from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_husky.tasks.skater import mdp

from .goal import GoalCfg, transition_goal


@dataclass(frozen=True)
class ScoreCfg:
    push_speed_std: float = 0.25**0.5
    push_yaw_std: float = 0.25**0.5
    steer_pose_std: float = 0.20**0.5
    steer_feet_std: float = 0.10**0.5
    steer_heading_std: float = 0.02**0.5
    steer_tilt_std: float = 0.02**0.5
    board_distance_std: float = 0.45
    goal: GoalCfg = field(default_factory=GoalCfg)


def compute_scores(env, cfg: ScoreCfg = ScoreCfg()) -> dict[str, torch.Tensor]:
    """Evaluate BFM0 motion with the HUSKY paper's phase-specific task terms."""
    board = SceneEntityCfg("skateboard")
    push_speed = mdp.push_skateboard_lin_vel(
        env,
        std=cfg.push_speed_std,
        command_name="skate",
        asset_cfg=board,
    )
    push_yaw = mdp.push_yaw_align(env, std=cfg.push_yaw_std)
    push_air = mdp.feet_air_time(
        env,
        sensor_name="left_feet_ground_contact",
        threshold_min=0.1,
        threshold_max=0.5,
        command_name="skate",
        command_threshold=0.1,
    )
    push_ankle = mdp.push_contact_ground_parallel(env).float()
    push_quality = (3.0 * push_speed + push_yaw + 3.0 * push_air + 0.5 * push_ankle) / 7.5

    board_offset = env.skateboard.data.root_link_pos_w[:, :2] - env.robot.data.root_link_pos_w[:, :2]
    board_distance = torch.linalg.vector_norm(board_offset, dim=-1)
    board_proximity = torch.exp(-torch.square(board_distance / cfg.board_distance_std))
    feet_board_contact = env._get_feet_contact_b()
    board_contact = torch.any(feet_board_contact, dim=-1).float()
    upright = torch.clamp((env.robot.data.root_link_pos_w[:, 2] - 0.3) / 0.4, 0.0, 1.0)
    board_retention = board_proximity * (0.25 + 0.75 * board_contact)
    push_task = push_quality * board_retention * upright

    steer_contact = mdp.steer_contact_num(env)
    steer_pose = mdp.steer_joint_pos(env, std=cfg.steer_pose_std)
    steer_feet = mdp.steer_feet_dis(env, std=cfg.steer_feet_std)
    steer_heading = mdp.steer_track_heading(env, command_name="skate", std=cfg.steer_heading_std)
    steer_tilt = mdp.steer_tilt_guide(env, command_name="skate", std=cfg.steer_tilt_std)
    steer_quality = (
        3.0 * steer_contact
        + 1.5 * steer_pose
        + steer_feet
        + 5.0 * steer_heading
        + 4.0 * steer_tilt
    ) / 17.5

    goal = transition_goal(env, cfg.goal)
    phase = env.contact_phase
    push_active = phase[:, 0]
    steer_active = phase[:, 1]
    transition_active = torch.clamp(phase[:, 2] + phase[:, 3], max=1.0)
    push = push_quality * push_active
    steer = steer_quality * steer_active
    active = push + steer + goal["transition_goal"] * transition_active

    return {
        "push": push,
        "push_quality": push_quality,
        "push_speed": push_speed,
        "push_yaw": push_yaw,
        "push_air": push_air,
        "push_ankle": push_ankle,
        "push_task": push_task,
        "board_distance": board_distance,
        "board_proximity": board_proximity,
        "board_contact": board_contact,
        "board_retention": board_retention,
        "upright": upright,
        "steer": steer,
        "steer_quality": steer_quality,
        "steer_contact": steer_contact,
        "steer_pose": steer_pose,
        "steer_feet": steer_feet,
        "steer_heading": steer_heading,
        "steer_tilt": steer_tilt,
        **goal,
        "phase_push": push_active,
        "phase_steer": steer_active,
        "phase_transition": transition_active,
        "active": active,
    }
