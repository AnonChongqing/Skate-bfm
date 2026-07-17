from __future__ import annotations

from collections import deque

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
            command_heading=cfg.env.command_heading, command_speed_range=cfg.env.command_speed_range,
            command_heading_range=cfg.env.command_heading_range, domain_randomization=cfg.env.domain_randomization,
            interval_push=cfg.env.interval_push,
            reset_noise=cfg.env.reset_noise, observation_noise=cfg.env.observation_noise,
            initial_mode=cfg.env.initial_mode, steer_reset_fraction=cfg.env.steer_reset_fraction,
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
        self.features = StateFeatureBuilder(self.low_env, cfg.latent.flow_dim, observation_noise=cfg.env.observation_noise)
        self.reward_adapter = RewardAdapter(self.low_env, cfg.reward)
        self.prototypes = torch.stack([
            torch.as_tensor(np.load(cfg.latent.prototype_paths[name]), device=cfg.experiment.device, dtype=torch.float32).reshape(-1, 256)[0]
            for name in MODE_NAMES
        ])
        self.prototypes = LatentMapper.project(self.prototypes, 16.0)
        n = self.low_env.husky_env.num_envs
        self.previous_flow = torch.zeros(n, cfg.latent.flow_dim, device=cfg.experiment.device)
        self.z_current = self.prototypes[0:1].repeat(n, 1)
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

    def reset(
        self,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
        phase: float | None = None,
    ) -> torch.Tensor:
        full_reset = env_ids is None
        if env_ids is None:
            env_ids = torch.arange(self.low_env.husky_env.num_envs, device=self.z_current.device)
        else:
            env_ids = env_ids.to(device=self.z_current.device, dtype=torch.long).reshape(-1)
        self.low_env.reset(seed, env_ids)
        if phase is not None:
            self.low_env.set_phase(phase, env_ids)
        self.features.reset(env_ids)
        self.reward_adapter.reset(env_ids)
        self.previous_flow[env_ids] = 0
        mode_id = self.scheduler.mode(self.low_env.husky_env)
        self.z_current[env_ids] = self.prototypes.index_select(0, mode_id[env_ids])
        self.latest_info = None
        self.latest_features = self.features.build(self.z_current, self.previous_flow, mode_id)
        if full_reset or not self.frame_history:
            self.frame_history.clear()
            for _ in range(self.cfg.policy.frame_stack):
                self.frame_history.append(self.latest_features.actor_frame.clone())
        else:
            for frame in self.frame_history:
                frame[env_ids] = self.latest_features.actor_frame[env_ids]
        return self._stacked_actor_obs()

    def step(self, flow_action: torch.Tensor) -> MacroStepResult:
        n = self.low_env.husky_env.num_envs
        flow = flow_action.to(self.cfg.experiment.device).reshape(n, self.cfg.latent.flow_dim).clamp(-1.0, 1.0)
        mode_id = self.scheduler.mode(self.low_env.husky_env, self.latest_info)
        mapped = self.mapper(self.z_current, mode_id, flow)
        reward_macro = torch.zeros(n, device=flow.device)
        component_macro = torch.zeros(n, len(REWARD_COMPONENTS), device=flow.device)
        terminated = torch.zeros(n, dtype=torch.bool, device=flow.device)
        truncated = torch.zeros_like(terminated)
        active = torch.ones_like(terminated)
        gamma_low = self.cfg.control.gamma_macro ** (1.0 / self.cfg.control.macro_steps)
        executed = torch.zeros(n, device=flow.device)
        forward_progress = torch.zeros(n, device=flow.device)
        speed_error = torch.zeros(n, device=flow.device)
        heading_error = torch.zeros(n, device=flow.device)
        board_tilt = torch.zeros(n, device=flow.device)
        phase_steps = torch.zeros(n, 4, device=flow.device)
        husky_terms: dict[str, torch.Tensor] = {}
        for low_step in range(self.cfg.control.macro_steps):
            action29 = self.bfm.act(self.low_env.observation, mapped.z_candidate)
            _, _, _, info = self.low_env.step(action29)
            if self.capture_low_frames:
                self.captured_frames.append(self.render())
            reward = self.reward_adapter.compute(info, self.z_current, mapped.z_candidate, flow, self.previous_flow)
            weight = gamma_low ** low_step if self.cfg.reward.macro_aggregation == "discounted_sum" else 1.0
            reward_macro += weight * reward.total_low_level * active
            component_macro += weight * self.reward_adapter.vector(reward) * active.unsqueeze(-1)
            env = self.low_env.husky_env
            command = env.command_manager.get_command("skate")
            target_heading = self.low_env.target_heading()
            forward_speed = env.skateboard.data.root_link_lin_vel_b[:, 0]
            forward_progress += env.step_dt * forward_speed * active
            speed_error += (command[:, 0] - forward_speed).abs() * active
            heading_error += torch.atan2(
                torch.sin(target_heading - env.skateboard.data.heading_w),
                torch.cos(target_heading - env.skateboard.data.heading_w),
            ).abs() * active
            board_tilt += env.skateboard.data.joint_pos[:, 0].abs() * active
            phase_steps += env.contact_phase * active.unsqueeze(-1)
            for manager_name, manager, gate in (
                ("push", env.push_reward_manager, env.contact_phase[:, 0]),
                ("steer", env.steer_reward_manager, env.contact_phase[:, 1]),
                ("transition", env.transition_reward_manager, env.contact_phase[:, 2:].amax(-1)),
                ("regularization", env.reg_reward_manager, torch.ones_like(active)),
            ):
                for term_index, term_name in enumerate(manager.active_terms):
                    name = f"husky/{manager_name}/{term_name}"
                    value = manager._step_reward[:, term_index] * env.step_dt * gate * active
                    husky_terms[name] = husky_terms.get(name, torch.zeros_like(value)) + weight * value
            terminated |= info["terminated"].to(flow.device)
            truncated |= info["truncated"].to(flow.device)
            self.latest_info = info
            executed += active.float()
            active &= ~(terminated | truncated)
            if not active.any():
                break
        if self.cfg.reward.macro_aggregation == "mean":
            reward_macro /= executed.clamp_min(1.0)
            component_macro /= executed.clamp_min(1.0).unsqueeze(-1)
        self.z_current = mapped.z_candidate.detach()
        self.previous_flow = flow.detach()
        next_mode = self.scheduler.mode(self.low_env.husky_env, self.latest_info)
        self.latest_features = self.features.build(self.z_current, self.previous_flow, next_mode, self.latest_info)
        self.frame_history.append(self.latest_features.actor_frame.clone())
        diagnostics = {
            "executed_low_steps": executed.detach(), "latent_delta_norm": mapped.delta_norm.detach(),
            "latent_cosine": mapped.cosine.detach(),
            "board_distance": self.latest_info["skateboard_xy_distance"].detach(),
            "root_height": self.latest_info["root_height"].detach(),
            "fell_over": self.latest_info["fell_over"].float().detach(),
            "feet_off_board": self.latest_info["feet_off_board"].float().detach(),
            "illegal_contact": self.latest_info["illegal_contact"].float().detach(),
            "board_forward_progress": forward_progress.detach(),
            "board_forward_speed": (
                forward_progress / (executed * self.low_env.husky_env.step_dt).clamp_min(1e-6)
            ).detach(),
            "speed_error": (speed_error / executed.clamp_min(1.0)).detach(),
            "heading_error": (heading_error / executed.clamp_min(1.0)).detach(),
            "board_tilt_abs": (board_tilt / executed.clamp_min(1.0)).detach(),
        }
        for phase_index, phase_name in enumerate(("push", "steer", "push2steer", "steer2push")):
            diagnostics[f"phase/{phase_name}"] = (phase_steps[:, phase_index] / executed.clamp_min(1.0)).detach()
        diagnostics.update({name: value.detach() for name, value in husky_terms.items()})
        return MacroStepResult(
            actor_obs=self._stacked_actor_obs(), features=self.latest_features,
            bfm_obs={key: value.to(flow.device) for key, value in self.low_env.observation.items()},
            z_current=self.z_current, previous_flow=self.previous_flow,
            reward_macro=reward_macro.unsqueeze(-1), reward_components=component_macro,
            terminated=terminated.unsqueeze(-1), truncated=truncated.unsqueeze(-1), diagnostics=diagnostics,
        )

    def snapshot(self) -> HuskyEnvSnapshot:
        extra = {
            "z_current": self.z_current.clone(), "previous_flow": self.previous_flow.clone(),
            "frame_history": [frame.clone() for frame in self.frame_history],
            "contact_duration": self.features.contact_duration.clone(),
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
        self.reward_adapter.previous_heading_error = extra["previous_heading_error"]
        self.latest_info = extra["latest_info"]
        mode = self.scheduler.mode(self.low_env.husky_env, self.latest_info)
        self.latest_features = self.features.build(self.z_current, self.previous_flow, mode, self.latest_info, update_duration=False)
        return self._stacked_actor_obs()

    def close(self) -> None:
        self.low_env.close()

    def render(self):
        return self.low_env.render()
