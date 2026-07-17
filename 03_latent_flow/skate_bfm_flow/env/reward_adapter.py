from __future__ import annotations

import torch

from ..schemas import SkateRewardOutput

REWARD_COMPONENTS = (
    "push_total", "steer_total", "transition_total", "regularization_total",
    "board_progress", "heading_progress", "foot_board_contact", "foot_ground_contact",
    "retention", "upright", "fall_penalty", "illegal_contact",
    "latent_magnitude_penalty", "latent_smoothness_penalty",
)


class RewardAdapter:
    def __init__(self, env, gate_progress_by_retention: bool = True, fall_height: float = 0.45) -> None:
        self.env = env
        self.gate_progress = gate_progress_by_retention
        self.fall_height = fall_height
        self.previous_board_x: torch.Tensor | None = None
        self.previous_heading_error: torch.Tensor | None = None

    def reset(self) -> None:
        husky = self.env.husky_env
        self.previous_board_x = husky.skateboard.data.root_link_pos_w[:, 0].clone()
        command = husky.command_manager.get_command("skate")
        self.previous_heading_error = torch.atan2(
            torch.sin(command[:, 1] - husky.skateboard.data.heading_w),
            torch.cos(command[:, 1] - husky.skateboard.data.heading_w),
        ).abs()

    @staticmethod
    def _manager_total(manager, dt: float) -> torch.Tensor:
        return manager._step_reward.sum(dim=-1) * dt

    def compute(self, info: dict, z_current: torch.Tensor, z_candidate: torch.Tensor, flow: torch.Tensor, previous_flow: torch.Tensor) -> SkateRewardOutput:
        env = self.env.husky_env
        device = env.device
        board_contact = torch.as_tensor(info["feet_board_contact"], device=device).reshape(1, 2).float()
        ground_contact = torch.as_tensor(info["feet_ground_contact"], device=device).reshape(1, 2).float()
        distance = torch.as_tensor(info["skateboard_xy_distance"], device=device).reshape(1)
        upright = ((env.robot.data.root_link_pos_w[:, 2] - 0.3) / 0.4).clamp(0.0, 1.0)
        retention = torch.exp(-torch.square(distance / 0.45)) * (0.25 + 0.75 * board_contact.amax(-1)) * upright
        board_x = env.skateboard.data.root_link_pos_w[:, 0]
        board_progress = board_x - self.previous_board_x
        command = env.command_manager.get_command("skate")
        heading_error = torch.atan2(torch.sin(command[:, 1] - env.skateboard.data.heading_w), torch.cos(command[:, 1] - env.skateboard.data.heading_w)).abs()
        heading_progress = self.previous_heading_error - heading_error
        self.previous_board_x = board_x.clone()
        self.previous_heading_error = heading_error.clone()
        if self.gate_progress:
            board_progress = board_progress * retention
            heading_progress = heading_progress * retention
        illegal_sensor = env.scene.sensors["illegal_contact"]
        illegal = torch.any(illegal_sensor.data.found, dim=-1).float()
        fall = (env.robot.data.root_link_pos_w[:, 2] < self.fall_height).float()
        components = {
            "push_total": self._manager_total(env.push_reward_manager, env.step_dt) * env.contact_phase[:, 0],
            "steer_total": self._manager_total(env.steer_reward_manager, env.step_dt) * env.contact_phase[:, 1],
            "transition_total": self._manager_total(env.transition_reward_manager, env.step_dt) * env.contact_phase[:, 2:].amax(-1),
            "regularization_total": self._manager_total(env.reg_reward_manager, env.step_dt),
            "board_progress": board_progress,
            "heading_progress": heading_progress,
            "foot_board_contact": board_contact.mean(-1),
            "foot_ground_contact": ground_contact.mean(-1),
            "retention": retention,
            "upright": upright,
            "fall_penalty": -fall,
            "illegal_contact": illegal,
            "latent_magnitude_penalty": -(z_candidate - z_current).square().mean(-1),
            "latent_smoothness_penalty": -(flow - previous_flow).square().mean(-1),
        }
        total = torch.as_tensor(info["husky_reward"], device=device).reshape(1)
        if not torch.isfinite(total).all() or any(not torch.isfinite(value).all() for value in components.values()):
            raise FloatingPointError("Non-finite Stage 03 reward")
        return SkateRewardOutput(total, components, {"retention": retention, "upright": upright})

    @staticmethod
    def vector(output: SkateRewardOutput) -> torch.Tensor:
        return torch.stack([output.components[name] for name in REWARD_COMPONENTS], dim=-1)
