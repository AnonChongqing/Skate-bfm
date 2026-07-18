from __future__ import annotations

import torch

from ..enums import SkateMode


class ModeScheduler:
    def __init__(self, recover_root_height: float = 0.55, recover_board_distance: float = 0.8, transition_lead_seconds: float = 0.3) -> None:
        self.recover_root_height = recover_root_height
        self.recover_board_distance = recover_board_distance
        self.transition_lead_seconds = transition_lead_seconds

    def mode(self, env, info: dict | None = None) -> torch.Tensor:
        root_height = env.robot.data.root_link_pos_w[:, 2]
        board_delta = env.skateboard.data.root_link_pos_w[:, :2] - env.robot.data.root_link_pos_w[:, :2]
        distance = torch.linalg.vector_norm(board_delta, dim=-1)
        recover = torch.logical_or(root_height < self.recover_root_height, distance > self.recover_board_distance)
        phase = env._get_phase()
        ratios = env.phase_ratios
        lead = self.transition_lead_seconds / env.cycle_time
        mount_start = (ratios[:, 1] - lead).clamp_min(ratios[:, 0])
        dismount_start = (ratios[:, 3] - lead).clamp_min(ratios[:, 2])
        mode = torch.full((env.num_envs,), int(SkateMode.PUSH), device=env.device, dtype=torch.long)
        mode[torch.logical_and(phase >= mount_start, phase < ratios[:, 2])] = int(SkateMode.MOUNT)
        mode[torch.logical_and(phase >= ratios[:, 2], phase < dismount_start)] = int(SkateMode.STEER)
        mode[phase >= dismount_start] = int(SkateMode.DISMOUNT)
        mode[recover] = int(SkateMode.RECOVER)
        return mode
