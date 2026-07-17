from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from mjlab.utils.lab_api.math import combine_frame_transforms, quat_apply, yaw_quat

from ..bfm.batch_action_adapter import BatchActionAdapter
from ..bfm.constants import DEFAULT_JOINT_POS, POLICY_JOINT_NAMES


@dataclass
class HuskyEnvConfig:
    task_id: str = "Mjlab-Skater-Flat-Unitree-G1"
    num_envs: int = 1
    seed: int = 42
    device: str | None = None
    action_mapping: str = "reference"
    action_clip: float | None = None
    observation_mapping: str = "reference"
    reference_blend: float = 0.0
    action_gain: float = 1.25
    command_speed: float | None = 0.7
    command_heading: float | None = 0.4
    command_speed_range: tuple[float, float] | None = None
    command_heading_range: tuple[float, float] | None = None
    domain_randomization: bool = False
    interval_push: bool = True
    reset_noise: float = 0.0
    observation_noise: bool = False
    history_len: int = 4
    disable_debug_vis: bool = True
    play: bool = False
    render_mode: str | None = None
    width: int = 640
    height: int = 480
    initial_mode: str = "push"
    steer_reset_fraction: float = 0.35
    steer_initial_speed: float = 0.7
    preserve_terminal_state: bool = True


class HuskyEnv:
    """Stage 03 low-level wrapper around the HUSKY-23DoF skateboard env."""

    def __init__(self, cfg: HuskyEnvConfig | None = None) -> None:
        self.cfg = cfg or HuskyEnvConfig()
        if self.cfg.observation_mapping not in {"reference", "nominal_aligned", "bfm_absolute"}:
            raise ValueError(f"Unknown observation mapping: {self.cfg.observation_mapping}")
        if self.cfg.initial_mode not in {"push", "steer", "mixed"}:
            raise ValueError(f"Unknown initial mode: {self.cfg.initial_mode}")
        skate_bfm_root = Path(__file__).resolve().parents[3]
        self.husky_root = skate_bfm_root / "husky_sim"
        os.chdir(self.husky_root)

        import mjlab_husky
        import mjlab_husky.tasks  # noqa: F401
        from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
        from mjlab_husky.tasks.registry import load_env_cfg

        husky_source = Path(mjlab_husky.__file__).resolve()
        if not husky_source.is_relative_to(self.husky_root):
            raise RuntimeError(f"HUSKY was imported outside Skate-bfm: {husky_source}")

        device = self.cfg.device
        if device is None:
            device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
        self.device = device

        env_cfg = load_env_cfg(self.cfg.task_id, play=self.cfg.play)
        env_cfg.seed = self.cfg.seed
        env_cfg.scene.num_envs = self.cfg.num_envs
        env_cfg.viewer.width = self.cfg.width
        env_cfg.viewer.height = self.cfg.height
        if not self.cfg.domain_randomization:
            env_cfg.events = {
                name: term
                for name, term in env_cfg.events.items()
                if not term.domain_randomization
            }
        if not self.cfg.interval_push:
            env_cfg.events.pop("push_robot", None)
        reset_event = env_cfg.events.get("reset_robot_joints")
        if reset_event is not None:
            reset_event.params["position_range"] = (-self.cfg.reset_noise, self.cfg.reset_noise)
        if self.cfg.disable_debug_vis and "skate" in env_cfg.commands:
            env_cfg.commands["skate"].debug_vis = False
        skate_command = env_cfg.commands.get("skate")
        if skate_command is not None:
            if self.cfg.command_speed_range is not None:
                skate_command.ranges.lin_vel_x = self.cfg.command_speed_range
            elif self.cfg.command_speed is not None:
                skate_command.ranges.lin_vel_x = (self.cfg.command_speed, self.cfg.command_speed)
            if self.cfg.command_heading_range is not None:
                skate_command.ranges.heading = self.cfg.command_heading_range
            elif self.cfg.command_heading is not None:
                skate_command.ranges.heading = (self.cfg.command_heading, self.cfg.command_heading)
        if self.cfg.preserve_terminal_state:
            env_cfg.terminations = {}
        self.husky_env = G1SkaterManagerBasedRlEnv(
            cfg=env_cfg,
            device=device,
            render_mode=self.cfg.render_mode,
        )
        self.action_adapter = BatchActionAdapter.from_env(
            self.husky_env,
            mode=self.cfg.action_mapping,
            action_clip=self.cfg.action_clip,
            reference_blend=self.cfg.reference_blend,
            action_gain=self.cfg.action_gain,
        )

        self._bfm_name_to_idx = {name: idx for idx, name in enumerate(POLICY_JOINT_NAMES)}
        self._husky_joint_ids_by_bfm_order = [
            self.husky_env.robot.joint_names.index(name) if name in self.husky_env.robot.joint_names else None
            for name in POLICY_JOINT_NAMES
        ]
        n = self.husky_env.num_envs
        self._last_bfm0_action = torch.zeros(n, 29, device=device)
        self._bfm_default_joint_pos = DEFAULT_JOINT_POS.to(device).unsqueeze(0)
        self._history_action = torch.zeros(n, self.cfg.history_len, 29, device=device)
        self._history_ang_vel = torch.zeros(n, self.cfg.history_len, 3, device=device)
        self._history_dof_pos = torch.zeros(n, self.cfg.history_len, 29, device=device)
        self._history_dof_vel = torch.zeros(n, self.cfg.history_len, 29, device=device)
        self._history_projected_gravity = torch.zeros(n, self.cfg.history_len, 3, device=device)

        pelvis_id = self.husky_env.robot.body_names.index("pelvis")
        steer_pose = np.load(self.husky_root / "dataset/ref_pose/steer_start_pose_b.npy")
        self._steer_root_pose_b = torch.as_tensor(
            steer_pose[pelvis_id], device=device, dtype=torch.float32
        ).unsqueeze(0).repeat(n, 1)

    def reset(self, seed: int | None = None, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.husky_env.num_envs, device=self.device)
            self.husky_env.reset(seed=self.cfg.seed if seed is None else seed, env_ids=env_ids)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
            if not len(env_ids):
                return self._obs
            self.husky_env.reset(seed=seed, env_ids=env_ids)
        self.husky_env.contact_phase[env_ids] = 0
        self.husky_env.contact_phase[env_ids, 0] = 1
        if not hasattr(self.husky_env, "last_contact_phase"):
            self.husky_env.last_contact_phase = self.husky_env.contact_phase.clone()
        else:
            self.husky_env.last_contact_phase[env_ids] = self.husky_env.contact_phase[env_ids]
        steer_ids = self._steer_reset_ids(env_ids)
        if len(steer_ids):
            self._reset_steer_state(steer_ids)
        self._last_bfm0_action[env_ids] = 0
        self._history_action[env_ids] = 0
        self._history_ang_vel[env_ids] = 0
        self._history_dof_pos[env_ids] = 0
        self._history_dof_vel[env_ids] = 0
        self._history_projected_gravity[env_ids] = 0
        self._obs = self._create_observation(env_ids)
        return self._obs

    def _steer_reset_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.cfg.initial_mode == "push":
            return env_ids[:0]
        if self.cfg.initial_mode == "steer":
            return env_ids
        mask = torch.rand(len(env_ids), device=self.device) < self.cfg.steer_reset_fraction
        return env_ids[mask]

    def _reset_steer_state(self, env_ids: torch.Tensor) -> None:
        """Place the robot in HUSKY's board-relative steer reference state."""
        env = self.husky_env
        board_pos = env.skateboard.data.root_link_pos_w[env_ids]
        board_quat = env.skateboard.data.root_link_quat_w[env_ids]
        root_pos, root_quat = combine_frame_transforms(
            board_pos,
            board_quat,
            self._steer_root_pose_b[env_ids, :3],
            yaw_quat(self._steer_root_pose_b[env_ids, 3:]),
        )
        env.robot.write_root_link_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids=env_ids)
        env.robot.write_joint_state_to_sim(
            env.steer_init_pos[env_ids].clone(),
            torch.zeros_like(env.steer_init_pos[env_ids]),
            env_ids=env_ids,
        )
        env.scene.write_data_to_sim()
        env.sim.forward()

        feet_pos = env.robot.data.body_link_pos_w[env_ids][:, env.feet_body_ids, :]
        marker_pos = env.skateboard.data.site_pos_w[env_ids][:, env.marker_body_ids, :]
        correction = torch.zeros_like(root_pos)
        correction[:, :2] = torch.mean(marker_pos[:, :, :2] - feet_pos[:, :, :2], dim=1)
        deck_ankle_height = board_pos[:, 2] + 0.045
        correction[:, 2] = deck_ankle_height - torch.mean(feet_pos[:, :, 2], dim=1)
        root_pos = root_pos + correction
        env.robot.write_root_link_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids=env_ids)

        local_velocity = torch.zeros(len(env_ids), 3, device=self.device)
        local_velocity[:, 0] = self.cfg.steer_initial_speed
        world_velocity = quat_apply(board_quat, local_velocity)
        root_velocity = torch.cat((world_velocity, torch.zeros_like(world_velocity)), dim=-1)
        env.robot.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
        env.skateboard.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)

        steer_phase = env.phase_ratios[env_ids, 2] * env.cycle_time / env.step_dt
        env.phase_length_buf[env_ids] = steer_phase.to(dtype=torch.long)
        env.contact_phase[env_ids] = 0
        env.contact_phase[env_ids, 1] = 1
        env.last_contact_phase[env_ids] = env.contact_phase[env_ids]
        env.last_contacts[env_ids] = False
        env.last_contacts_b[env_ids] = False
        env.last_contacts_g[env_ids] = False
        env.scene.write_data_to_sim()
        env.sim.forward()

    def step(self, bfm0_action):
        bfm0_action_t = self._as_action_tensor(bfm0_action)
        current_obs = self._obs
        husky_action = self.action_adapter(bfm0_action_t)
        _, reward, terminated, truncated, extras = self.husky_env.step(husky_action)
        feet_board = self.husky_env._get_feet_contact_b()
        feet_ground = self.husky_env._get_feet_contact_g()
        fell_over = torch.zeros_like(terminated)
        feet_off_board = torch.zeros_like(terminated)
        illegal = torch.zeros_like(terminated)
        if self.cfg.preserve_terminal_state:
            tilt = torch.acos((-self.husky_env.robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)).abs()
            fell_over = tilt > math.radians(70.0)
            feet_off_board = ~torch.any(feet_board, dim=-1)
            illegal = torch.any(self.husky_env.scene.sensors["illegal_contact"].data.found, dim=-1)
            terminated = fell_over | feet_off_board | illegal
            truncated = self.husky_env.episode_length_buf >= self.husky_env.max_episode_length
        self._last_bfm0_action.copy_(bfm0_action_t * 5.0)
        next_obs = self._create_observation()
        self._obs = next_obs
        info = self._make_info(reward, terminated, truncated, extras, husky_action, feet_board, feet_ground)
        info.update({"fell_over": fell_over, "feet_off_board": feet_off_board, "illegal_contact": illegal})
        return current_obs, next_obs, reward, info

    def close(self) -> None:
        self.husky_env.close()

    def render(self):
        return self.husky_env.render()

    @property
    def observation(self) -> dict[str, torch.Tensor]:
        return self._obs

    def goal_observation(self, joint_pos: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build an approximate BFM0 goal observation from a HUSKY joint pose."""
        batch = self.tracking_observation(joint_pos)
        return {name: value[0] for name, value in batch.items()}

    def tracking_observation(self, joint_pos: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map a HUSKY joint trajectory to batched BFM0 tracking observations."""
        if joint_pos.ndim == 1:
            joint_pos = joint_pos.unsqueeze(0)
        joint_pos = joint_pos.to(self.device)
        target = self._joint_tensor_bfm_order(joint_pos)
        dof_pos = target - self._joint_reference()[:1]

        count = dof_pos.shape[0]
        dof_vel = torch.zeros_like(dof_pos)
        if count > 1:
            dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / self.husky_env.step_dt
            dof_vel[0] = dof_vel[1]
        gravity = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(count, 1)
        ang_vel = torch.zeros(count, 3, device=self.device)
        state = torch.cat((dof_pos, dof_vel, gravity, ang_vel), dim=1)

        history_pos = torch.stack(
            [dof_pos[torch.clamp(torch.arange(count, device=self.device) - lag, min=0)] for lag in range(self.cfg.history_len)],
            dim=1,
        )
        history_vel = torch.stack(
            [dof_vel[torch.clamp(torch.arange(count, device=self.device) - lag, min=0)] for lag in range(self.cfg.history_len)],
            dim=1,
        )
        history = torch.cat(
            (
                torch.zeros(count, self.cfg.history_len * 29, device=self.device),
                torch.zeros(count, self.cfg.history_len * 3, device=self.device),
                history_pos.reshape(count, -1),
                history_vel.reshape(count, -1),
                gravity[:, None, :].repeat(1, self.cfg.history_len, 1).reshape(count, -1),
            ),
            dim=1,
        )
        return {
            "state": state.detach().cpu(),
            "history_actor": history.detach().cpu(),
            "last_action": torch.zeros(count, 29),
            "privileged_state": torch.zeros(count, 463),
        }

    @property
    def mapping_report(self):
        return {
            "bfm_joints": POLICY_JOINT_NAMES,
            "husky_joints": tuple(self.action_adapter.shared_ids.tolist()),
            "mode": self.action_adapter.mode,
        }

    def set_calibration(self, reference_blend: float, action_gain: float) -> None:
        self.cfg.reference_blend = max(0.0, min(float(reference_blend), 1.0))
        self.cfg.action_gain = max(0.0, float(action_gain))
        self.action_adapter.reference_blend = self.cfg.reference_blend
        self.action_adapter.action_gain = self.cfg.action_gain

    def _as_action_tensor(self, action) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            out = action.to(device=self.device, dtype=torch.float32)
        else:
            out = torch.tensor(action, device=self.device, dtype=torch.float32)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        expected = (self.husky_env.num_envs, 29)
        if out.shape != expected:
            raise ValueError(f"Expected action shape {expected}, got {tuple(out.shape)}")
        return out

    def _joint_tensor_bfm_order(self, value: torch.Tensor, default: torch.Tensor | None = None) -> torch.Tensor:
        out = torch.zeros(value.shape[0], 29, device=self.device, dtype=torch.float32)
        for bfm_idx, husky_idx in enumerate(self._husky_joint_ids_by_bfm_order):
            if husky_idx is not None:
                out[:, bfm_idx] = value[:, husky_idx]
            elif default is not None:
                out[:, bfm_idx] = default[:, bfm_idx]
        return out

    def _joint_reference(self) -> torch.Tensor:
        if self.cfg.observation_mapping == "bfm_absolute":
            return self._bfm_default_joint_pos
        husky_default = self._joint_tensor_bfm_order(self.husky_env.robot.data.default_joint_pos)
        if self.cfg.observation_mapping == "nominal_aligned":
            return husky_default
        return self._bfm_default_joint_pos + self.cfg.reference_blend * (
            husky_default - self._bfm_default_joint_pos
        )

    def _roll_history(self, name: str, value: torch.Tensor, env_ids: torch.Tensor) -> None:
        hist = getattr(self, name)
        hist[env_ids, 1:] = hist[env_ids, :-1].clone()
        hist[env_ids, 0] = value[env_ids]

    def _create_observation(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        robot_data = self.husky_env.robot.data
        if env_ids is None:
            env_ids = torch.arange(self.husky_env.num_envs, device=self.device)
        joint_pos = self._joint_tensor_bfm_order(robot_data.joint_pos)
        dof_pos = joint_pos - self._joint_reference()
        dof_vel = self._joint_tensor_bfm_order(robot_data.joint_vel)
        projected_gravity = robot_data.projected_gravity_b.to(dtype=torch.float32)
        ang_vel = robot_data.root_link_ang_vel_b.to(dtype=torch.float32) * 0.25

        self._roll_history("_history_action", self._last_bfm0_action, env_ids)
        self._roll_history("_history_ang_vel", ang_vel, env_ids)
        self._roll_history("_history_dof_pos", dof_pos, env_ids)
        self._roll_history("_history_dof_vel", dof_vel, env_ids)
        self._roll_history("_history_projected_gravity", projected_gravity, env_ids)

        state = torch.cat((dof_pos, dof_vel, projected_gravity, ang_vel), dim=1)
        history = torch.cat(
            (
                self._history_action.reshape(self.husky_env.num_envs, -1),
                self._history_ang_vel.reshape(self.husky_env.num_envs, -1),
                self._history_dof_pos.reshape(self.husky_env.num_envs, -1),
                self._history_dof_vel.reshape(self.husky_env.num_envs, -1),
                self._history_projected_gravity.reshape(self.husky_env.num_envs, -1),
            ),
            dim=1,
        )
        return {
            "state": state.detach(),
            "history_actor": history.detach(),
            "last_action": self._last_bfm0_action.detach().clone(),
            "privileged_state": torch.zeros(self.husky_env.num_envs, 463, device=self.device),
        }

    def _make_info(self, reward, terminated, truncated, extras, husky_action: torch.Tensor, feet_board: torch.Tensor, feet_ground: torch.Tensor) -> dict:
        env = self.husky_env
        command = env.command_manager.get_command("skate")
        root_position = env.robot.data.root_link_pos_w
        board_position = env.skateboard.data.root_link_pos_w
        board_relative = board_position - root_position
        return {
            "step": int(env.common_step_counter),
            "root_position": root_position.detach().clone(),
            "root_height": root_position[:, 2].detach().clone(),
            "skateboard_position": board_position.detach().clone(),
            "skateboard_relative_position": board_relative.detach().clone(),
            "skateboard_xy_distance": torch.linalg.vector_norm(board_relative[:, :2], dim=-1).detach(),
            "husky_reward": reward.detach().clone(),
            "terminated": terminated.detach().clone(),
            "truncated": truncated.detach().clone(),
            "dt": float(env.step_dt),
            "command": command.detach().clone() if command is not None else None,
            "skateboard_lin_vel_b": env.skateboard.data.root_link_lin_vel_b.detach().clone(),
            "skateboard_heading_w": env.skateboard.data.heading_w.detach().clone(),
            "contact_phase": env.contact_phase.detach().clone(),
            "feet_board_contact": feet_board.detach().clone(),
            "feet_ground_contact": feet_ground.detach().clone(),
            "husky_action": husky_action.detach().clone(),
            "extras": extras,
        }
