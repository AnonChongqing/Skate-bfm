from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul, wrap_to_pi

from ..schemas import FEATURE_SCHEMA_VERSION, FeatureBatch


ROBOT_NAMES = tuple(
    [f"joint_pos_rel/{i}" for i in range(23)]
    + [f"joint_vel/{i}" for i in range(23)]
    + [f"base_ang_vel/{axis}" for axis in "xyz"]
    + [f"gravity/{axis}" for axis in "xyz"]
    + ["root_height"]
    + [f"previous_action/{i}" for i in range(23)]
    + [f"base_lin_vel/{axis}" for axis in "xyz"]
)
BOARD_NAMES = tuple(
    [f"relative_pos/{axis}" for axis in "xyz"]
    + [f"orientation_6d/{i}" for i in range(6)]
    + [f"local_lin_vel/{axis}" for axis in "xyz"]
    + [f"local_ang_vel/{axis}" for axis in "xyz"]
    + ["heading_error_sin", "heading_error_cos", "distance", "deck_tilt"]
)
CONTACT_DEPLOY_NAMES = tuple(
    ["left_board", "right_board", "left_ground", "right_ground"]
    + [f"duration/{name}" for name in ("left_board", "right_board", "left_ground", "right_ground")]
    + [f"marker_error/{foot}/{axis}" for foot in ("left", "right") for axis in "xyz"]
)
CONTACT_FORCE_NAMES = tuple(
    f"force/{surface}/{foot}/{axis}"
    for surface in ("board", "ground")
    for foot in ("left", "right")
    for axis in "xyz"
)
CONTACT_NAMES = CONTACT_DEPLOY_NAMES + CONTACT_FORCE_NAMES
GOAL_MODE_NAMES = tuple(
    ["goal_local_x", "goal_local_y", "goal_distance", "heading_error_sin", "heading_error_cos"]
    + [f"mode/{name}" for name in ("push", "mount", "steer", "dismount", "recover")]
    + [f"phase/{name}" for name in ("push", "steer", "mount", "dismount")]
    + ["phase_sin", "phase_cos"]
)


class StateFeatureBuilder:
    schema_version = FEATURE_SCHEMA_VERSION

    def __init__(self, env, flow_dim: int, max_force: float = 500.0) -> None:
        self.env = env
        self.flow_dim = flow_dim
        self.max_force = max_force
        self.contact_duration = torch.zeros(env.husky_env.num_envs, 4, device=env.device)

    @property
    def dimensions(self) -> dict[str, int]:
        return {"robot": len(ROBOT_NAMES), "board": len(BOARD_NAMES), "contact": len(CONTACT_NAMES), "goal_mode": len(GOAL_MODE_NAMES)}

    def reset(self) -> None:
        self.contact_duration.zero_()

    def _contacts(self, info: dict | None) -> tuple[torch.Tensor, torch.Tensor]:
        if info is not None and info.get("feet_board_contact") is not None:
            board = torch.as_tensor(info["feet_board_contact"], device=self.env.device).reshape(1, 2).bool()
            ground = torch.as_tensor(info["feet_ground_contact"], device=self.env.device).reshape(1, 2).bool()
        else:
            board = self.env.husky_env._get_feet_contact_b()
            ground = self.env.husky_env._get_feet_contact_g()
        return board, ground

    def _force_vector(self, sensor_name: str) -> torch.Tensor:
        force = self.env.husky_env.scene.sensors[sensor_name].data.force
        force = force.reshape(force.shape[0], -1, 3).sum(dim=1)
        clipped = force.abs().clamp_max(self.max_force)
        return force.sign() * torch.log1p(clipped) / math.log1p(self.max_force)

    def build(self, z_current: torch.Tensor, previous_flow: torch.Tensor, mode_id: torch.Tensor, info: dict | None = None, update_duration: bool = True) -> FeatureBatch:
        env = self.env.husky_env
        robot = env.robot.data
        board = env.skateboard.data
        joint_rel = robot.joint_pos - robot.default_joint_pos
        previous_action = env.action_manager.action
        robot_features = torch.cat(
            (joint_rel, robot.joint_vel, robot.root_link_ang_vel_b, robot.projected_gravity_b,
             robot.root_link_pos_w[:, 2:3], previous_action, robot.root_link_lin_vel_b), dim=-1
        )

        relative_pos = quat_apply_inverse(robot.root_link_quat_w, board.root_link_pos_w - robot.root_link_pos_w)
        relative_quat = quat_mul(quat_inv(robot.root_link_quat_w), board.root_link_quat_w)
        rotation = matrix_from_quat(relative_quat)
        orientation_6d = rotation[..., :2].reshape(env.num_envs, 6)
        command = env.command_manager.get_command("skate")
        heading_error = wrap_to_pi(command[:, 1] - board.heading_w)
        distance = torch.linalg.vector_norm(relative_pos[:, :2], dim=-1, keepdim=True)
        tilt = board.joint_pos[:, :1]
        board_features = torch.cat(
            (relative_pos, orientation_6d, board.root_link_lin_vel_b, board.root_link_ang_vel_b,
             heading_error.sin().unsqueeze(-1), heading_error.cos().unsqueeze(-1), distance, tilt), dim=-1
        )

        board_contact, ground_contact = self._contacts(info)
        contacts = torch.cat((board_contact, ground_contact), dim=-1)
        if update_duration:
            self.contact_duration = torch.where(contacts, self.contact_duration + env.step_dt, torch.zeros_like(self.contact_duration))
        marker_error = quat_apply_inverse(
            robot.root_link_quat_w[:, None, :].expand(-1, 2, -1),
            board.site_pos_w[:, env.marker_body_ids, :] - robot.body_link_pos_w[:, env.feet_body_ids, :],
        ).reshape(env.num_envs, 6)
        forces = torch.cat(
            (
                self._force_vector("left_feet_board_contact"), self._force_vector("right_feet_board_contact"),
                self._force_vector("left_feet_ground_contact"), self._force_vector("right_feet_ground_contact"),
            ), dim=-1
        )
        deploy_contact = torch.cat((contacts.float(), self.contact_duration, marker_error), dim=-1)
        critic_contact = torch.cat((deploy_contact, forces), dim=-1)

        goal_xy = torch.stack((heading_error.cos(), heading_error.sin()), dim=-1)
        goal = torch.cat((goal_xy, torch.ones(env.num_envs, 1, device=env.device), heading_error.sin().unsqueeze(-1), heading_error.cos().unsqueeze(-1)), dim=-1)
        mode_one_hot = F.one_hot(mode_id.long(), num_classes=5).float()
        phase_scalar = env._get_phase()
        goal_mode = torch.cat((goal, mode_one_hot, env.contact_phase, torch.sin(2 * math.pi * phase_scalar).unsqueeze(-1), torch.cos(2 * math.pi * phase_scalar).unsqueeze(-1)), dim=-1)

        actor_frame = torch.cat((robot_features, board_features[:, :18], deploy_contact, goal, mode_one_hot, z_current, previous_flow), dim=-1)
        output = FeatureBatch(actor_frame, robot_features, board_features, critic_contact, goal_mode, mode_id)
        for name, tensor, expected in (
            ("robot", robot_features, 79), ("board", board_features, 19),
            ("contact", critic_contact, 26), ("goal_mode", goal_mode, 16),
        ):
            if tensor.shape != (env.num_envs, expected) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid {name} features: {tuple(tensor.shape)}")
        return output
