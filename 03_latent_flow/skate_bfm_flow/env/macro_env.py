from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..bfm.batch_action_adapter import BatchActionAdapter
from ..bfm.frozen_policy import FrozenBfmPolicy
from ..bfm.latent_basis import load_basis
from ..bfm.latent_mapper import LatentMapper
from ..config import Stage03Config
from ..enums import MODE_NAMES
from ..schemas import MacroStepResult
from .husky import HuskyEnv, HuskyEnvConfig
from .mode_scheduler import ModeScheduler
from .reward_adapter import REWARD_COMPONENTS, RewardAdapter
from .snapshot import HuskyEnvSnapshot
from .state_features import StateFeatureBuilder


class LatentFlowMacroEnv:
    def __init__(self, cfg: Stage03Config, render_mode: str | None = None) -> None:
        self.cfg = cfg
        env_cfg = HuskyEnvConfig(
            task_id=cfg.env.task_id, num_envs=cfg.env.num_envs, seed=cfg.experiment.seed,
            device=cfg.experiment.device, action_mapping=cfg.env.action_mapping,
            action_clip=cfg.env.action_clip, reference_blend=cfg.env.reference_blend,
            action_gain=cfg.env.action_gain, command_speed=cfg.env.command_speed,
            command_heading=cfg.env.command_heading, domain_randomization=cfg.env.domain_randomization,
            reset_noise=cfg.env.reset_noise, initial_mode=cfg.env.initial_mode,
            steer_initial_speed=cfg.env.steer_initial_speed, render_mode=render_mode or cfg.env.render_mode,
        )
        self.low_env = HuskyEnv(env_cfg)
        self.bfm = FrozenBfmPolicy(cfg.paths.bfm_model_dir, cfg.experiment.device, cfg.control.mean_bfm_action)
        self.adapter = BatchActionAdapter.from_env(
            self.low_env.husky_env, mode=cfg.env.action_mapping, reference_blend=cfg.env.reference_blend,
            action_gain=cfg.env.action_gain, action_clip=cfg.env.action_clip,
        )
        basis, self.basis_metadata = load_basis(cfg.paths.basis_path, cfg.experiment.device)
        self.mapper = LatentMapper(basis, cfg.latent.step_size, cfg.latent.update_type).to(cfg.experiment.device)
        self.scheduler = ModeScheduler(cfg.mode.recover_root_height, cfg.mode.recover_board_distance)
        self.features = StateFeatureBuilder(self.low_env, cfg.latent.flow_dim)
        self.reward_adapter = RewardAdapter(self.low_env, cfg.reward.gate_progress_by_retention, cfg.reward.fall_height)
        self.prototypes = torch.stack([
            torch.as_tensor(np.load(cfg.latent.prototype_paths[name]), device=cfg.experiment.device, dtype=torch.float32).reshape(-1, 256)[0]
            for name in MODE_NAMES
        ])
        self.prototypes = LatentMapper.project(self.prototypes, 16.0)
        self.previous_flow = torch.zeros(1, cfg.latent.flow_dim, device=cfg.experiment.device)
        self.z_current = self.prototypes[0:1].clone()
        self.frame_history: deque[torch.Tensor] = deque(maxlen=cfg.policy.frame_stack)
        self.latest_info: dict | None = None
        self.latest_features = None
        self.capture_low_frames = False
        self.captured_frames: list = []

    @property
    def actor_obs_dim(self) -> int:
        return self.latest_features.actor_frame.shape[-1] * self.cfg.policy.frame_stack

    def _stacked_actor_obs(self) -> torch.Tensor:
        return torch.cat(tuple(self.frame_history), dim=-1)

    def reset(self, seed: int | None = None) -> torch.Tensor:
        self.low_env.reset(seed)
        self.features.reset()
        self.reward_adapter.reset()
        self.previous_flow.zero_()
        mode_id = self.scheduler.mode(self.low_env.husky_env)
        self.z_current = self.prototypes.index_select(0, mode_id).clone()
        self.latest_info = None
        self.latest_features = self.features.build(self.z_current, self.previous_flow, mode_id)
        self.frame_history.clear()
        for _ in range(self.cfg.policy.frame_stack):
            self.frame_history.append(self.latest_features.actor_frame.clone())
        return self._stacked_actor_obs()

    def step(self, flow_action: torch.Tensor) -> MacroStepResult:
        flow = flow_action.to(self.cfg.experiment.device).reshape(1, self.cfg.latent.flow_dim).clamp(-1.0, 1.0)
        mode_id = self.scheduler.mode(self.low_env.husky_env, self.latest_info)
        mapped = self.mapper(self.z_current, mode_id, flow)
        reward_macro = torch.zeros(1, device=flow.device)
        component_macro = torch.zeros(1, len(REWARD_COMPONENTS), device=flow.device)
        terminated = torch.zeros(1, dtype=torch.bool, device=flow.device)
        truncated = torch.zeros_like(terminated)
        gamma_low = self.cfg.control.gamma_macro ** (1.0 / self.cfg.control.macro_steps)
        executed = 0
        for low_step in range(self.cfg.control.macro_steps):
            action29 = self.bfm.act(self.low_env.observation, mapped.z_candidate)
            _, _, _, info = self.low_env.step(action29)
            if self.capture_low_frames:
                self.captured_frames.append(self.render())
            reward = self.reward_adapter.compute(info, self.z_current, mapped.z_candidate, flow, self.previous_flow)
            weight = gamma_low ** low_step if self.cfg.reward.macro_aggregation == "discounted_sum" else 1.0
            reward_macro += weight * reward.total_low_level
            component_macro += weight * self.reward_adapter.vector(reward)
            terminated |= torch.tensor([info["terminated"]], device=flow.device)
            truncated |= torch.tensor([info["truncated"]], device=flow.device)
            self.latest_info = info
            executed += 1
            if bool(terminated.item() or truncated.item()):
                break
        if self.cfg.reward.macro_aggregation == "mean" and executed:
            reward_macro /= executed
            component_macro /= executed
        self.z_current = mapped.z_candidate.detach()
        self.previous_flow = flow.detach()
        next_mode = self.scheduler.mode(self.low_env.husky_env, self.latest_info)
        self.latest_features = self.features.build(self.z_current, self.previous_flow, next_mode, self.latest_info)
        self.frame_history.append(self.latest_features.actor_frame.clone())
        diagnostics = {
            "executed_low_steps": float(executed), "latent_delta_norm": float(mapped.delta_norm.item()),
            "latent_cosine": float(mapped.cosine.item()), "board_distance": float(self.latest_info["skateboard_xy_distance"]),
            "root_height": float(self.latest_info["root_height"]),
            "fell_over": float(self.latest_info["fell_over"]),
            "feet_off_board": float(self.latest_info["feet_off_board"]),
            "illegal_contact": float(self.latest_info["illegal_contact"]),
        }
        return MacroStepResult(
            actor_obs=self._stacked_actor_obs(), features=self.latest_features,
            bfm_obs={key: value.unsqueeze(0).to(flow.device) for key, value in self.low_env.observation.items()},
            z_current=self.z_current, previous_flow=self.previous_flow,
            reward_macro=reward_macro.unsqueeze(-1), reward_components=component_macro,
            terminated=terminated.unsqueeze(-1), truncated=truncated.unsqueeze(-1), diagnostics=diagnostics,
        )

    def snapshot(self) -> HuskyEnvSnapshot:
        extra = {
            "z_current": self.z_current.clone(), "previous_flow": self.previous_flow.clone(),
            "frame_history": [frame.clone() for frame in self.frame_history],
            "contact_duration": self.features.contact_duration.clone(),
            "previous_board_x": self.reward_adapter.previous_board_x.clone(),
            "previous_heading_error": self.reward_adapter.previous_heading_error.clone(),
            "latest_info": self.latest_info,
        }
        return HuskyEnvSnapshot.capture(self.low_env, extra)

    def restore(self, snapshot: HuskyEnvSnapshot) -> torch.Tensor:
        extra = snapshot.restore(self.low_env)
        self.z_current = extra["z_current"]
        self.previous_flow = extra["previous_flow"]
        self.frame_history = deque(extra["frame_history"], maxlen=self.cfg.policy.frame_stack)
        self.features.contact_duration.copy_(extra["contact_duration"])
        self.reward_adapter.previous_board_x = extra["previous_board_x"]
        self.reward_adapter.previous_heading_error = extra["previous_heading_error"]
        self.latest_info = extra["latest_info"]
        mode = self.scheduler.mode(self.low_env.husky_env, self.latest_info)
        self.latest_features = self.features.build(self.z_current, self.previous_flow, mode, self.latest_info, update_duration=False)
        return self._stacked_actor_obs()

    def close(self) -> None:
        self.low_env.close()

    def render(self):
        return self.low_env.render()
