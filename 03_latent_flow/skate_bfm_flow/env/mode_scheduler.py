from __future__ import annotations

import torch

from ..enums import SkateMode


class ModeScheduler:
    def __init__(self, recover_root_height: float = 0.55, recover_board_distance: float = 0.8) -> None:
        self.recover_root_height = recover_root_height
        self.recover_board_distance = recover_board_distance

    def mode(self, env, info: dict | None = None) -> torch.Tensor:
        root_height = env.robot.data.root_link_pos_w[:, 2]
        board_delta = env.skateboard.data.root_link_pos_w[:, :2] - env.robot.data.root_link_pos_w[:, :2]
        distance = torch.linalg.vector_norm(board_delta, dim=-1)
        recover = torch.logical_or(root_height < self.recover_root_height, distance > self.recover_board_distance)
        phase = env.contact_phase
        mode = torch.full((env.num_envs,), int(SkateMode.PUSH), device=env.device, dtype=torch.long)
        mode[phase[:, 0] > 0.5] = int(SkateMode.PUSH)
        mode[phase[:, 2] > 0.5] = int(SkateMode.MOUNT)
        mode[phase[:, 1] > 0.5] = int(SkateMode.STEER)
        mode[phase[:, 3] > 0.5] = int(SkateMode.DISMOUNT)
        mode[recover] = int(SkateMode.RECOVER)
        return mode
