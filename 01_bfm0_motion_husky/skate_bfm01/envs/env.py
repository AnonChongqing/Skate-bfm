from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from mjlab.utils.lab_api.math import combine_frame_transforms, quat_apply, yaw_quat

from skate_bfm01.adapters import Bfm0ToHusky23ActionAdapter
from skate_bfm01.constants import DEFAULT_JOINT_POS, POLICY_JOINT_NAMES


@dataclass
class Bfm0Husky23EnvCfg:
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
    domain_randomization: bool = False
    reset_noise: float = 0.0
    history_len: int = 4
    disable_debug_vis: bool = True
    play: bool = False
    render_mode: str | None = None
    width: int = 640
    height: int = 480
    initial_mode: str = "push"
    steer_initial_speed: float = 0.7


class Bfm0Husky23Env:
    """BFM0-style single-env wrapper around the official HUSKY-23DoF skateboard env."""

    def __init__(self, cfg: Bfm0Husky23EnvCfg | None = None) -> None:
        self.cfg = cfg or Bfm0Husky23EnvCfg()
        if self.cfg.num_envs != 1:
            raise ValueError("First-layer BFM0 adapter currently supports num_envs=1 only.")
        if self.cfg.observation_mapping not in {"reference", "nominal_aligned", "bfm_absolute"}:
            raise ValueError(f"Unknown observation mapping: {self.cfg.observation_mapping}")
        if self.cfg.initial_mode not in {"push", "steer"}:
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
        reset_event = env_cfg.events.get("reset_robot_joints")
        if reset_event is not None:
            reset_event.params["position_range"] = (-self.cfg.reset_noise, self.cfg.reset_noise)
        if self.cfg.disable_debug_vis and "skate" in env_cfg.commands:
            env_cfg.commands["skate"].debug_vis = False
        skate_command = env_cfg.commands.get("skate")
        if skate_command is not None:
            if self.cfg.command_speed is not None:
                skate_command.ranges.lin_vel_x = (self.cfg.command_speed, self.cfg.command_speed)
            if self.cfg.command_heading is not None:
                skate_command.ranges.heading = (self.cfg.command_heading, self.cfg.command_heading)
        self.husky_env = G1SkaterManagerBasedRlEnv(
            cfg=env_cfg,
            device=device,
            render_mode=self.cfg.render_mode,
        )
        self.action_adapter = Bfm0ToHusky23ActionAdapter(
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
        self._last_bfm0_action = torch.zeros(1, 29, device=device)
        self._bfm_default_joint_pos = torch.tensor(DEFAULT_JOINT_POS, device=device).unsqueeze(0)
        self._history_action = torch.zeros(self.cfg.history_len, 29, device=device)
        self._history_ang_vel = torch.zeros(self.cfg.history_len, 3, device=device)
        self._history_dof_pos = torch.zeros(self.cfg.history_len, 29, device=device)
        self._history_dof_vel = torch.zeros(self.cfg.history_len, 29, device=device)
        self._history_projected_gravity = torch.zeros(self.cfg.history_len, 3, device=device)

        pelvis_id = self.husky_env.robot.body_names.index("pelvis")
        steer_pose = np.load(self.husky_root / "dataset/ref_pose/steer_start_pose_b.npy")
        self._steer_root_pose_b = torch.as_tensor(
            steer_pose[pelvis_id], device=device, dtype=torch.float32
        ).unsqueeze(0)

    def reset(self, seed: int | None = None):
        self.husky_env.reset(seed=self.cfg.seed if seed is None else seed)
        if self.cfg.initial_mode == "steer":
            self._reset_steer_state()
        self._last_bfm0_action.zero_()
        self._history_action.zero_()
        self._history_ang_vel.zero_()
        self._history_dof_pos.zero_()
        self._history_dof_vel.zero_()
        self._history_projected_gravity.zero_()
        self._obs = self._create_observation()
        return self._obs

    def _reset_steer_state(self) -> None:
        """Place the robot in HUSKY's board-relative steer reference state."""
        env = self.husky_env
        board_pos = env.skateboard.data.root_link_pos_w.squeeze(1)
        board_quat = env.skateboard.data.root_link_quat_w.squeeze(1)
        root_pos, root_quat = combine_frame_transforms(
            board_pos,
            board_quat,
            self._steer_root_pose_b[:, :3],
            yaw_quat(self._steer_root_pose_b[:, 3:]),
        )
        env.robot.write_root_link_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1))
        env.robot.write_joint_state_to_sim(
            env.steer_init_pos.clone(),
            torch.zeros_like(env.steer_init_pos),
        )
        env.scene.write_data_to_sim()
        env.sim.forward()

        feet_pos = env.robot.data.body_link_pos_w[:, env.feet_body_ids, :]
        marker_pos = env.skateboard.data.site_pos_w[:, env.marker_body_ids, :]
        correction = torch.zeros_like(root_pos)
        correction[:, :2] = torch.mean(marker_pos[:, :, :2] - feet_pos[:, :, :2], dim=1)
        deck_ankle_height = board_pos[:, 2] + 0.045
        correction[:, 2] = deck_ankle_height - torch.mean(feet_pos[:, :, 2], dim=1)
        root_pos = root_pos + correction
        env.robot.write_root_link_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1))

        local_velocity = torch.zeros(env.num_envs, 3, device=self.device)
        local_velocity[:, 0] = self.cfg.steer_initial_speed
        world_velocity = quat_apply(board_quat, local_velocity)
        root_velocity = torch.cat((world_velocity, torch.zeros_like(world_velocity)), dim=-1)
        env.robot.write_root_link_velocity_to_sim(root_velocity)
        env.skateboard.write_root_link_velocity_to_sim(root_velocity)

        steer_phase = env.phase_ratios[:, 2] * env.cycle_time / env.step_dt
        env.phase_length_buf[:] = steer_phase.to(dtype=torch.long)
        env.last_contacts.zero_()
        env.last_contacts_b.zero_()
        env.last_contacts_g.zero_()
        env.scene.write_data_to_sim()
        env.sim.forward()

    def step(self, bfm0_action):
        bfm0_action_t = self._as_action_tensor(bfm0_action)
        current_obs = self._obs
        husky_action = self.action_adapter.map_action(bfm0_action_t)
        _, reward, terminated, truncated, extras = self.husky_env.step(husky_action)
        self._last_bfm0_action = bfm0_action_t * 5.0
        next_obs = self._create_observation()
        self._obs = next_obs
        info = self._make_info(reward, terminated, truncated, extras, husky_action)
        return current_obs, next_obs, float(reward[0].detach().cpu()), info

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
        dof_pos = target - self._joint_reference()

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
        return self.action_adapter.report

    def set_calibration(self, reference_blend: float, action_gain: float) -> None:
        self.cfg.reference_blend = max(0.0, min(float(reference_blend), 1.0))
        self.cfg.action_gain = max(0.0, float(action_gain))
        self.action_adapter.set_calibration(self.cfg.reference_blend, self.cfg.action_gain)

    def _as_action_tensor(self, action) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            out = action.to(device=self.device, dtype=torch.float32)
        else:
            out = torch.tensor(action, device=self.device, dtype=torch.float32)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.shape != (1, 29):
            raise ValueError(f"Expected action shape (29,) or (1, 29), got {tuple(out.shape)}")
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

    def _roll_history(self, name: str, value: torch.Tensor) -> None:
        hist = getattr(self, name)
        hist[1:] = hist[:-1].clone()
        hist[0] = value[0]

    def _create_observation(self) -> dict[str, torch.Tensor]:
        robot_data = self.husky_env.robot.data
        joint_pos = self._joint_tensor_bfm_order(robot_data.joint_pos)
        dof_pos = joint_pos - self._joint_reference()
        dof_vel = self._joint_tensor_bfm_order(robot_data.joint_vel)
        projected_gravity = robot_data.projected_gravity_b.to(dtype=torch.float32)
        ang_vel = robot_data.root_link_ang_vel_b.to(dtype=torch.float32) * 0.25

        self._roll_history("_history_action", self._last_bfm0_action)
        self._roll_history("_history_ang_vel", ang_vel)
        self._roll_history("_history_dof_pos", dof_pos)
        self._roll_history("_history_dof_vel", dof_vel)
        self._roll_history("_history_projected_gravity", projected_gravity)

        state = torch.cat((dof_pos[0], dof_vel[0], projected_gravity[0], ang_vel[0]), dim=0)
        history = torch.cat(
            (
                self._history_action.reshape(-1),
                self._history_ang_vel.reshape(-1),
                self._history_dof_pos.reshape(-1),
                self._history_dof_vel.reshape(-1),
                self._history_projected_gravity.reshape(-1),
            ),
            dim=0,
        )
        return {
            "state": state.detach().cpu(),
            "history_actor": history.detach().cpu(),
            "last_action": self._last_bfm0_action[0].detach().cpu(),
            "privileged_state": torch.zeros(463, dtype=torch.float32),
        }

    def _make_info(self, reward, terminated, truncated, extras, husky_action: torch.Tensor) -> dict:
        env = self.husky_env
        command = env.command_manager.get_command("skate")
        feet_board = env._get_feet_contact_b() if hasattr(env, "_get_feet_contact_b") else None
        feet_ground = env._get_feet_contact_g() if hasattr(env, "_get_feet_contact_g") else None
        root_position = env.robot.data.root_link_pos_w[0]
        board_position = env.skateboard.data.root_link_pos_w[0]
        board_relative = board_position - root_position
        return {
            "step": int(env.common_step_counter),
            "root_position": root_position.detach().cpu().numpy().copy(),
            "root_height": float(root_position[2].detach().cpu()),
            "skateboard_position": board_position.detach().cpu().numpy().copy(),
            "skateboard_relative_position": board_relative.detach().cpu().numpy().copy(),
            "skateboard_xy_distance": float(torch.linalg.vector_norm(board_relative[:2]).detach().cpu()),
            "husky_reward": float(reward[0].detach().cpu()),
            "terminated": bool(terminated[0].detach().cpu()),
            "truncated": bool(truncated[0].detach().cpu()),
            "dt": float(env.step_dt),
            "command": command[0].detach().cpu().numpy().copy() if command is not None else None,
            "skateboard_lin_vel_b": env.skateboard.data.root_link_lin_vel_b[0].detach().cpu().numpy().copy(),
            "skateboard_heading_w": float(env.skateboard.data.heading_w[0].detach().cpu()),
            "contact_phase": env.contact_phase[0].detach().cpu().numpy().copy(),
            "feet_board_contact": feet_board[0].detach().cpu().numpy().copy() if feet_board is not None else None,
            "feet_ground_contact": feet_ground[0].detach().cpu().numpy().copy() if feet_ground is not None else None,
            "husky_action": husky_action[0].detach().cpu().numpy().copy(),
            "extras": extras,
        }
