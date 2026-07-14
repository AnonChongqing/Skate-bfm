from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .policy import BfmPolicy, load_z


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class FixedControl:
    def __init__(self, policy: BfmPolicy) -> None:
        self.policy = policy
        self.status = {"state": "fixed", "phase": 0.0, "blend": 0.0, "heading_error": 0.0}

    def __call__(self, obs):
        return self.policy(obs)


class SteerControl:
    """Run signed BFM0 rotate prompts from an on-board steer reset."""

    def __init__(
        self,
        policy: BfmPolicy,
        env,
        reward_path: Path,
        steer_path: Path | None = None,
        steer_key: str = "move-ego-0-0",
        steer_index: int = 0,
        turn_mix: float = 0.2,
        turn_index: int = 0,
        follow_mix: float = 0.0,
        follow_index: int = 0,
        follow_start: float = 0.08,
        follow_pos_gain: float = 1.0,
        follow_vel_gain: float = 0.0,
        follow_tilt_gain: float = 0.0,
        follow_foot_gain: float = 0.0,
    ) -> None:
        self.policy = policy
        self.env = env
        self.base_z = (
            torch.as_tensor(np.load(steer_path), device=policy.device, dtype=torch.float32)
            if steer_path is not None and steer_path.exists()
            else load_z(reward_path, steer_key, steer_index).to(policy.device)
        )
        self.turn_pos_z = load_z(reward_path, "rotate-z-5-0.5", turn_index).to(policy.device)
        self.turn_neg_z = load_z(reward_path, "rotate-z--5-0.5", turn_index).to(policy.device)
        self.follow_z = {
            0.0: load_z(reward_path, "move-ego-0-0.3", follow_index).to(policy.device),
            math.pi / 2: load_z(reward_path, "move-ego-90-0.3", follow_index).to(policy.device),
            -math.pi / 2: load_z(reward_path, "move-ego--90-0.3", follow_index).to(policy.device),
            math.pi: load_z(reward_path, "move-ego-180-0.3", follow_index).to(policy.device),
        }
        self.turn_mix = max(0.0, min(turn_mix, 1.0))
        self.follow_mix = max(0.0, min(follow_mix, 1.0))
        self.follow_start = max(0.0, follow_start)
        self.follow_pos_gain = follow_pos_gain
        self.follow_vel_gain = follow_vel_gain
        self.follow_tilt_gain = follow_tilt_gain
        self.follow_foot_gain = follow_foot_gain
        self.status: dict[str, float | str | bool] = {}

    def __call__(self, obs):
        husky = self.env.husky_env
        target = husky.get_heading_target_w("skate")
        if target is None:
            target_heading = float(husky.command_manager.get_command("skate")[0, 1].detach().cpu())
        else:
            target_heading = float(target[0].detach().cpu())
        board_heading = float(husky.skateboard.data.heading_w[0].detach().cpu())
        heading_error = _wrap(target_heading - board_heading)
        root_height = float(husky.robot.data.root_link_pos_w[0, 2].detach().cpu())
        stability = max(0.0, min((root_height - 0.3) / 0.35, 1.0))
        board_contacts = husky._get_feet_contact_b()[0]
        board_contact = bool(torch.any(board_contacts).detach().cpu())
        contact_scale = 1.0 if board_contact else 0.2
        blend = min(abs(heading_error) / 0.5, 1.0) * self.turn_mix * stability * contact_scale
        board_offset = husky.skateboard.data.root_link_pos_w[0, :2] - husky.robot.data.root_link_pos_w[0, :2]
        board_distance = float(torch.linalg.vector_norm(board_offset).detach().cpu())
        robot_heading = float(husky.robot.data.heading_w[0].detach().cpu())
        cosine = math.cos(robot_heading)
        sine = math.sin(robot_heading)

        def world_to_body(vector: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                (
                    cosine * vector[0] + sine * vector[1],
                    -sine * vector[0] + cosine * vector[1],
                )
            )

        local_offset = world_to_body(board_offset)
        relative_velocity = (
            husky.skateboard.data.root_link_lin_vel_w[0, :2]
            - husky.robot.data.root_link_lin_vel_w[0, :2]
        )
        local_velocity = world_to_body(relative_velocity)
        projected_gravity = husky.robot.data.projected_gravity_b[0, :2]
        feet_position = husky.robot.data.body_link_pos_w[0, husky.feet_body_ids, :2]
        marker_position = husky.skateboard.data.site_pos_w[0, husky.marker_body_ids, :2]
        missing_contacts = torch.logical_not(board_contacts)
        if torch.any(missing_contacts):
            foot_error_w = torch.mean(
                marker_position[missing_contacts] - feet_position[missing_contacts], dim=0
            )
        else:
            foot_error_w = torch.mean(marker_position - feet_position, dim=0)
        local_foot_error = world_to_body(foot_error_w)
        balance_error = (
            self.follow_pos_gain * local_offset
            + self.follow_vel_gain * local_velocity
            + self.follow_tilt_gain * projected_gravity
            + self.follow_foot_gain * local_foot_error
        )
        balance_magnitude = float(torch.linalg.vector_norm(balance_error).detach().cpu())
        balance_heading = math.atan2(float(balance_error[1]), float(balance_error[0]))
        follow_heading = min(self.follow_z, key=lambda angle: abs(_wrap(balance_heading - angle)))
        follow_scale = min(max((balance_magnitude - self.follow_start) / 0.25, 0.0), 1.0)
        follow_blend = self.follow_mix * follow_scale
        base_z = self.policy.project(
            (1.0 - follow_blend) * self.base_z + follow_blend * self.follow_z[follow_heading]
        )
        turn_z = self.turn_pos_z if heading_error >= 0.0 else self.turn_neg_z
        z = self.policy.project((1.0 - blend) * base_z + blend * turn_z)

        self.status = {
            "state": "steer_only",
            "phase": float(husky._get_phase()[0].detach().cpu()),
            "blend": blend,
            "heading_error": heading_error,
            "stability": stability,
            "board_distance": board_distance,
            "board_speed": float(husky.skateboard.data.root_link_lin_vel_b[0, 0].detach().cpu()),
            "board_contact": board_contact,
            "follow_blend": follow_blend,
            "follow_heading": follow_heading,
            "balance_magnitude": balance_magnitude,
            "balance_x": float(balance_error[0].detach().cpu()),
            "balance_y": float(balance_error[1].detach().cpu()),
            "foot_error_x": float(local_foot_error[0].detach().cpu()),
            "foot_error_y": float(local_foot_error[1].detach().cpu()),
        }
        return self.policy.act(obs, z)


class PhaseControl:
    """Reward/tracking/reward prompt switching around the frozen BFM0 actor."""

    def __init__(
        self,
        policy: BfmPolicy,
        env,
        reward_path: Path,
        push_path: Path | None = None,
        steer_path: Path | None = None,
        start_steps: int = 0,
        start_key: str = "move-ego-0-0",
        start_index: int = 0,
        transition_start: float = 0.35,
        transition_enter: float = 0.4,
        transition_scale: float = 1.0,
        transition_mix: float = 0.5,
        transition_blend: float = 0.7,
        transition_steps: int = 18,
        trigger_after: float = 0.12,
        trigger_speed: float = 0.2,
        trigger_distance: float = 0.25,
        turn_mix: float = 0.05,
        turn_index: int = 0,
        recover_height: float = 0.0,
        stable_height: float = 0.01,
        push_feedback: bool = True,
        push_hold_key: str = "move-ego-0-0",
        push_hold_index: int = 0,
        push_drive_mix: float = 0.3,
        push_hold_steps: int = 15,
        push_drive_steps: int = 8,
    ) -> None:
        self.policy = policy
        self.env = env
        self.push_z = (
            torch.as_tensor(np.load(push_path), device=policy.device, dtype=torch.float32)
            if push_path is not None and push_path.exists()
            else policy.z
        )
        self.start_z = load_z(reward_path, start_key, start_index).to(policy.device)
        self.push_hold_z = load_z(reward_path, push_hold_key, push_hold_index).to(policy.device)
        self.push_drive_z = policy.project(
            (1.0 - push_drive_mix) * self.push_hold_z + push_drive_mix * self.push_z
        )
        self.push_feedback = push_feedback
        self.push_hold_steps = max(push_hold_steps, 1)
        self.push_drive_steps = max(push_drive_steps, 1)
        self.steer_z = (
            torch.as_tensor(np.load(steer_path), device=policy.device, dtype=torch.float32)
            if steer_path is not None and steer_path.exists()
            else self.start_z
        )
        self.start_phase = start_steps * env.husky_env.step_dt / env.husky_env.cycle_time
        self.turn_pos_z = load_z(reward_path, "rotate-z-5-0.5", turn_index).to(policy.device)
        self.turn_neg_z = load_z(reward_path, "rotate-z--5-0.5", turn_index).to(policy.device)
        ratios = env.husky_env.phase_ratios[0].detach().cpu().tolist()
        self.transition_start = max(ratios[0], min(transition_start, ratios[1]))
        self.transition_enter = max(0.0, min(transition_enter, 1.0))
        self.transition_scale = max(0.0, min(transition_scale, 1.0))
        self.transition_mix = max(0.0, min(transition_mix, 1.0))
        self.transition_blend = max(0.0, min(transition_blend, 1.0))
        self.transition_steps = max(2, transition_steps)
        self.trigger_after = max(0.0, min(trigger_after, 1.0))
        self.trigger_speed = trigger_speed
        self.trigger_distance = trigger_distance
        self.turn_mix = max(0.0, min(turn_mix, 1.0))
        self.recover_height = recover_height
        self.stable_height = max(stable_height, recover_height + 1e-6)
        self._stability = 1.0
        self._push_track: torch.Tensor | None = None
        self._return_track: torch.Tensor | None = None
        self._last_phase = 0.0
        self._mode = "push"
        self._mode_step = 0
        self._push_mode = "hold"
        self._push_mode_step = 0
        self._push_level = 0.0
        self._current_push_z = self.push_hold_z
        self.status: dict[str, float | str | bool] = {}
        self._last_state = ""

    def _steer_prompt(self, heading_error: float, stability: float = 1.0) -> torch.Tensor:
        turn = self.turn_pos_z if heading_error >= 0.0 else self.turn_neg_z
        blend = min(abs(heading_error) / 0.5, 1.0) * self.turn_mix * stability
        return self.policy.project((1.0 - blend) * self.steer_z + blend * turn)

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = max(0.0, min(value, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _make_track(self, target: torch.Tensor, steps: int) -> torch.Tensor:
        start = self.env.husky_env.robot.data.joint_pos[0].detach()
        alpha = torch.linspace(0.0, 1.0, max(steps, 2), device=self.env.device)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        trajectory = start[None, :] + alpha[:, None] * self.transition_scale * (
            target.to(self.env.device)[None, :] - start[None, :]
        )
        return self.policy.infer_tracking(self.env.tracking_observation(trajectory))

    def _feedback_push(
        self,
        board_distance: float,
        board_speed: float,
        board_contacts: list[bool],
        ground_contacts: list[bool],
    ) -> torch.Tensor:
        if not self.push_feedback:
            self._current_push_z = self.push_z
            return self._current_push_z

        stance_ready = bool(
            board_contacts[1]
            and ground_contacts[0]
            and board_distance < 0.35
        )
        self._push_mode_step += 1
        if self._push_mode == "hold":
            if self._push_mode_step >= self.push_hold_steps and stance_ready and board_speed < 0.4:
                self._push_mode = "drive"
                self._push_mode_step = 0
        elif (
            self._push_mode_step >= self.push_drive_steps
            or board_distance > 0.45
            or board_speed > 0.55
        ):
            self._push_mode = "hold"
            self._push_mode_step = 0

        target = 1.0 if self._push_mode == "drive" else 0.0
        rate = 0.25 if target > self._push_level else 0.4
        self._push_level += max(-rate, min(target - self._push_level, rate))
        self._current_push_z = self.policy.project(
            (1.0 - self._push_level) * self.push_hold_z
            + self._push_level * self.push_drive_z
        )
        self._stance_ready = stance_ready
        return self._current_push_z

    def _track_prompt(
        self,
        phase: float,
        begin: float,
        end: float,
        target: torch.Tensor,
        start_z: torch.Tensor,
        reward_z: torch.Tensor,
        cache_name: str,
    ) -> tuple[torch.Tensor, float]:
        alpha = self._smoothstep((phase - begin) / max(end - begin, 1e-6))
        track = getattr(self, cache_name)
        if track is None:
            steps = round((end - begin) * self.env.husky_env.cycle_time / self.env.husky_env.step_dt) + 1
            track = self._make_track(target, steps)
            setattr(self, cache_name, track)
        index = min(round(alpha * (len(track) - 1)), len(track) - 1)
        base = self.policy.project((1.0 - alpha) * start_z + alpha * reward_z)
        enter_weight = (
            self._smoothstep(alpha / self.transition_enter)
            if self.transition_enter > 0.0
            else 1.0
        )
        exit_weight = (
            self._smoothstep((1.0 - alpha) / (1.0 - self.transition_blend))
            if self.transition_blend < 1.0
            else 1.0
        )
        track_weight = self.transition_mix * min(enter_weight, exit_weight)
        z = self.policy.project((1.0 - track_weight) * base + track_weight * track[index])
        return z, alpha

    def _select(self) -> tuple[torch.Tensor, str, float, float, float]:
        husky = self.env.husky_env
        phase = float(husky._get_phase()[0].detach().cpu())
        ratios = husky.phase_ratios[0].detach().cpu().tolist()
        target = husky.get_heading_target_w("skate")
        if target is None:
            command = husky.command_manager.get_command("skate")
            target_heading = float(command[0, 1].detach().cpu())
        else:
            target_heading = float(target[0].detach().cpu())
        board_heading = float(husky.skateboard.data.heading_w[0].detach().cpu())
        heading_error = _wrap(target_heading - board_heading)
        root_height = float(husky.robot.data.root_link_pos_w[0, 2].detach().cpu())
        board_offset = husky.skateboard.data.root_link_pos_w[0, :2] - husky.robot.data.root_link_pos_w[0, :2]
        board_distance = float(torch.linalg.vector_norm(board_offset).detach().cpu())
        board_speed = float(husky.skateboard.data.root_link_lin_vel_b[0, 0].detach().cpu())
        board_contacts = husky._get_feet_contact_b()[0].detach().cpu().tolist()
        ground_contacts = husky._get_feet_contact_g()[0].detach().cpu().tolist()
        board_contact = bool(any(board_contacts))
        self._board_distance = board_distance
        self._board_speed = board_speed
        self._board_contact = board_contact
        self._stability = self._smoothstep(
            (root_height - self.recover_height) / (self.stable_height - self.recover_height)
        )
        transition_steer_prompt = self._steer_prompt(heading_error)
        steer_prompt = self._steer_prompt(heading_error, self._stability)

        if phase < self._last_phase:
            self._push_track = None
            self._return_track = None
            self._mode = "push"
            self._mode_step = 0
            self._push_mode = "hold"
            self._push_mode_step = 0
            self._push_level = 0.0
        self._last_phase = phase

        if self._mode == "push":
            push_prompt = self._feedback_push(
                board_distance,
                board_speed,
                board_contacts,
                ground_contacts,
            )
            ready = (
                phase >= self.trigger_after
                and board_speed >= self.trigger_speed
                and board_distance <= self.trigger_distance
                and board_contact
                and root_height >= 0.5
            )
            timed_ready = (
                phase >= self.transition_start
                and board_speed >= self.trigger_speed
                and board_distance <= self.trigger_distance
                and board_contact
                and root_height >= 0.5
            )
            if ready or timed_ready:
                self._mode = "push2steer"
                self._mode_step = 0
                self._push_track = self._make_track(husky.steer_init_pos[0], self.transition_steps)
            else:
                if self.start_phase > 0.0 and phase < min(self.start_phase, self.transition_start):
                    alpha = max(0.0, min(phase / self.start_phase, 1.0))
                    alpha = self._smoothstep(alpha)
                    z = self.policy.project((1.0 - alpha) * self.start_z + alpha * push_prompt)
                    return z, "push_start", phase, alpha, heading_error
                return push_prompt, f"push_{self._push_mode}", phase, self._push_level, heading_error

        if self._mode == "push2steer":
            alpha = self._smoothstep(self._mode_step / (self.transition_steps - 1))
            assert self._push_track is not None
            index = min(self._mode_step, len(self._push_track) - 1)
            base = self.policy.project((1.0 - alpha) * self._current_push_z + alpha * transition_steer_prompt)
            envelope = math.sin(math.pi * alpha)
            track_weight = self.transition_mix * envelope
            z = self.policy.project((1.0 - track_weight) * base + track_weight * self._push_track[index])
            self._mode_step += 1
            if self._mode_step >= self.transition_steps:
                self._mode = "steer"
                self._mode_step = 0
            return z, "push2steer", phase, alpha, heading_error

        if self._mode == "steer":
            if ratios[3] <= phase < ratios[4]:
                self._mode = "steer2push"
                self._mode_step = 0
                self._return_track = self._make_track(
                    husky.robot.data.default_joint_pos[0], self.transition_steps
                )
            else:
                return steer_prompt, "steer", phase, 0.0, heading_error

        if self._mode == "steer2push":
            alpha = self._smoothstep(self._mode_step / (self.transition_steps - 1))
            assert self._return_track is not None
            index = min(self._mode_step, len(self._return_track) - 1)
            base = self.policy.project((1.0 - alpha) * steer_prompt + alpha * self._current_push_z)
            envelope = math.sin(math.pi * alpha)
            track_weight = self.transition_mix * envelope
            z = self.policy.project((1.0 - track_weight) * base + track_weight * self._return_track[index])
            self._mode_step += 1
            return z, "steer2push", phase, alpha, heading_error
        raise RuntimeError(f"Unknown control mode: {self._mode}")

    def __call__(self, obs):
        z, state, phase, blend, heading_error = self._select()
        self.status = {
            "state": state,
            "phase": phase,
            "blend": blend,
            "heading_error": heading_error,
            "stability": self._stability,
            "board_distance": self._board_distance,
            "board_speed": self._board_speed,
            "board_contact": self._board_contact,
            "push_mode": self._push_mode,
            "push_level": self._push_level,
            "stance_ready": getattr(self, "_stance_ready", False),
        }
        if state != self._last_state:
            print(f"[Control] state={state} phase={phase:.3f}")
            self._last_state = state
        return self.policy.act(obs, z)


def make_control(
    name: str,
    policy: BfmPolicy,
    env,
    reward_path: Path,
    push_path: Path | None = None,
    steer_path: Path | None = None,
    start_steps: int = 0,
    start_key: str = "move-ego-0-0",
    start_index: int = 0,
    transition_start: float = 0.35,
    transition_enter: float = 0.4,
    transition_scale: float = 1.0,
    transition_mix: float = 0.5,
    transition_blend: float = 0.7,
    transition_steps: int = 18,
    trigger_after: float = 0.12,
    trigger_speed: float = 0.2,
    trigger_distance: float = 0.25,
    turn_mix: float = 0.05,
    turn_index: int = 0,
    recover_height: float = 0.0,
    stable_height: float = 0.01,
    push_feedback: bool = True,
    push_hold_key: str = "move-ego-0-0",
    push_hold_index: int = 0,
    push_drive_mix: float = 0.3,
    push_hold_steps: int = 15,
    push_drive_steps: int = 8,
    steer_key: str = "move-ego-0-0",
    steer_index: int = 0,
    follow_mix: float = 0.0,
    follow_index: int = 0,
    follow_start: float = 0.08,
    follow_pos_gain: float = 1.0,
    follow_vel_gain: float = 0.0,
    follow_tilt_gain: float = 0.0,
    follow_foot_gain: float = 0.0,
):
    if name == "fixed":
        return FixedControl(policy)
    if name == "steer":
        return SteerControl(
            policy,
            env,
            reward_path,
            steer_path,
            steer_key,
            steer_index,
            turn_mix,
            turn_index,
            follow_mix,
            follow_index,
            follow_start,
            follow_pos_gain,
            follow_vel_gain,
            follow_tilt_gain,
            follow_foot_gain,
        )
    return PhaseControl(
        policy,
        env,
        reward_path,
        push_path,
        steer_path,
        start_steps,
        start_key,
        start_index,
        transition_start,
        transition_enter,
        transition_scale,
        transition_mix,
        transition_blend,
        transition_steps,
        trigger_after,
        trigger_speed,
        trigger_distance,
        turn_mix,
        turn_index,
        recover_height,
        stable_height,
        push_feedback,
        push_hold_key,
        push_hold_index,
        push_drive_mix,
        push_hold_steps,
        push_drive_steps,
    )
