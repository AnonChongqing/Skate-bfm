from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from ..data.branch_dataset import BRANCH_ACTION_SEMANTICS, PHASE_SAMPLING_VERSION, BranchDataset
from ..env.macro_env import LatentFlowMacroEnv
from ..env.reward_adapter import REWARD_COMPONENTS
from ..enums import MODE_NAMES, SkateMode


@dataclass
class BranchCollector:
    env: LatentFlowMacroEnv
    seed: int = 42

    def _tracking_flow(self, mode_ids: torch.Tensor) -> torch.Tensor:
        device = self.env.z_current.device
        husky = self.env.low_env.husky_env
        current = husky.robot.data.joint_pos
        target = current.clone()
        mount = mode_ids == int(SkateMode.MOUNT)
        dismount = mode_ids == int(SkateMode.DISMOUNT)
        transition = torch.logical_or(mount, dismount)
        if not transition.any():
            return torch.zeros(len(mode_ids), self.env.cfg.latent.flow_dim, device=device)
        if mount.any():
            target[mount] = husky.steer_init_pos[mount]
        if dismount.any():
            target[dismount] = husky.robot.data.default_joint_pos[dismount]
        next_joint = current + 0.25 * (target - current)
        tracking_obs = self.env.low_env.tracking_observation(next_joint)
        tracking_z = self.env.bfm.tracking(tracking_obs).to(device)
        basis = self.env.mapper.basis.index_select(0, mode_ids)
        delta = tracking_z - self.env.z_current
        flow = torch.bmm(basis.transpose(1, 2), delta.unsqueeze(-1)).squeeze(-1)
        return torch.where(
            transition.unsqueeze(-1),
            flow / self.env.cfg.latent.step_size,
            torch.zeros_like(flow),
        )

    def _candidates(self, count: int, mode_ids: torch.Tensor) -> torch.Tensor:
        device = self.env.z_current.device
        flow_dim = self.env.cfg.latent.flow_dim
        num_envs = len(mode_ids)
        generator = torch.Generator(device=device).manual_seed(self.seed + int(self.env.low_env.husky_env.common_step_counter))
        candidates = [torch.zeros(num_envs, flow_dim, device=device)]
        transition = torch.logical_or(mode_ids == int(SkateMode.MOUNT), mode_ids == int(SkateMode.DISMOUNT))
        if transition.any():
            candidates.append(self._tracking_flow(mode_ids))
        local_count = max(1, round((count - 1) * 0.5))
        for _ in range(local_count):
            candidates.append(torch.randn(num_envs, flow_dim, generator=generator, device=device) * self.env.cfg.branch.local_std)
        basis = self.env.mapper.basis.index_select(0, mode_ids)
        target = self.env.prototypes.index_select(0, mode_ids) - self.env.z_current
        prototype_flow = torch.bmm(basis.transpose(1, 2), target.unsqueeze(-1)).squeeze(-1) / self.env.cfg.latent.step_size
        candidates.append(prototype_flow)
        if transition.any():
            next_modes = torch.where(
                mode_ids == int(SkateMode.MOUNT),
                torch.full_like(mode_ids, int(SkateMode.STEER)),
                torch.full_like(mode_ids, int(SkateMode.PUSH)),
            )
            next_target = self.env.prototypes.index_select(0, next_modes) - self.env.z_current
            next_flow = torch.bmm(basis.transpose(1, 2), next_target.unsqueeze(-1)).squeeze(-1)
            candidates.append(next_flow / self.env.cfg.latent.step_size)
        while len(candidates) < count:
            candidates.append(torch.empty(num_envs, flow_dim, device=device).uniform_(-1.0, 1.0, generator=generator))
        return torch.stack(candidates[:count], dim=1).clamp(-1.0, 1.0)

    def _phase_plan(self, num_anchors: int) -> torch.Tensor | None:
        weights = self.env.cfg.branch.phase_weights
        if not weights:
            return None
        modes = [int(SkateMode[name.upper()]) for name in weights]
        normalized = torch.tensor(list(weights.values()), dtype=torch.float64)
        normalized /= normalized.sum()
        exact = normalized * num_anchors
        counts = exact.floor().long()
        remainder = num_anchors - int(counts.sum())
        if remainder:
            order = torch.argsort(exact - counts, descending=True)
            counts[order[:remainder]] += 1
        return torch.cat([
            torch.full((int(count),), mode, dtype=torch.long)
            for mode, count in zip(modes, counts, strict=True)
        ])

    def _prepare_phase(self, mode_id: int, seed: int, zero: torch.Tensor) -> None:
        env = self.env.low_env.husky_env
        ratios = env.phase_ratios[0]
        lead = self.env.cfg.mode.transition_lead_seconds / env.cycle_time
        interval = {
            int(SkateMode.PUSH): (float(ratios[0]), float(ratios[1]) - lead),
            int(SkateMode.MOUNT): (float(ratios[1]) - lead, float(ratios[2])),
            int(SkateMode.STEER): (float(ratios[2]), float(ratios[3]) - lead),
            int(SkateMode.DISMOUNT): (float(ratios[3]) - lead, float(ratios[4])),
        }[mode_id]
        lower, upper = interval
        transition = mode_id in {int(SkateMode.MOUNT), int(SkateMode.DISMOUNT)}
        initial_mode = "steer" if mode_id in {int(SkateMode.STEER), int(SkateMode.DISMOUNT)} else "push"
        phase = lower if transition else (lower + upper) * 0.5
        preroll_seconds = self.env.cfg.branch.transition_preroll_seconds if transition else 0.0
        reset_phase = max(0.0, phase - preroll_seconds / env.cycle_time)
        self.env.reset(seed, phase=reset_phase, initial_mode=initial_mode)

        # Reach transition anchors through BFM dynamics; changing only the phase clock creates invalid states.
        preroll_macro_steps = math.ceil(
            preroll_seconds / (env.step_dt * self.env.cfg.control.macro_steps)
        )
        for _ in range(preroll_macro_steps):
            result = self.env.step(zero)
            done_ids = (result.terminated | result.truncated).reshape(-1).nonzero().reshape(-1)
            if len(done_ids):
                self.env.reset(seed, done_ids, phase=reset_phase, initial_mode=initial_mode)

        if transition:
            duration_low_steps = max(1, round((upper - lower) * env.cycle_time / env.step_dt))
            max_offset = max(0, math.ceil(duration_low_steps / self.env.cfg.control.macro_steps) - 1)
            generator = torch.Generator().manual_seed(seed + 29009)
            warmup_macro_steps = int(torch.randint(0, max_offset + 1, (1,), generator=generator).item())
        else:
            warmup_macro_steps = math.ceil(
                self.env.cfg.branch.warmup_low_steps / self.env.cfg.control.macro_steps
            )
        for _ in range(warmup_macro_steps):
            mode_ids = torch.full(
                (len(zero),), mode_id, device=zero.device, dtype=torch.long,
            )
            flow = self._tracking_flow(mode_ids).clamp(-1.0, 1.0) if transition else zero
            result = self.env.step(flow)
            done_ids = (result.terminated | result.truncated).reshape(-1).nonzero().reshape(-1)
            if len(done_ids):
                self.env.reset(seed, done_ids, phase=phase, initial_mode=initial_mode)

        actual_modes = self.env.latest_features.mode_id
        bad_ids = (actual_modes != mode_id).nonzero().reshape(-1)
        if len(bad_ids):
            self.env.reset(seed, bad_ids, phase=phase, initial_mode=initial_mode)

    def collect(
        self,
        num_anchors: int,
        candidates_per_anchor: int,
        horizon_low_steps: int,
        horizon_low_steps_range: tuple[int, int] | None = None,
        anchor_offset: int = 0,
        log_interval: int = 1000,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> BranchDataset:
        started = time.perf_counter()
        num_envs = self.env.low_env.husky_env.num_envs
        zero = torch.zeros(num_envs, self.env.cfg.latent.flow_dim, device=self.env.z_current.device)
        phase_plan = self._phase_plan(num_anchors)
        if phase_plan is None:
            self.env.reset(self.seed)
            for _ in range(math.ceil(self.env.cfg.branch.warmup_low_steps / self.env.cfg.control.macro_steps)):
                result = self.env.step(zero)
                done_ids = (result.terminated | result.truncated).reshape(-1).nonzero().reshape(-1)
                if len(done_ids):
                    self.env.reset(self.seed, done_ids)
        records: dict[str, list[torch.Tensor]] = {}
        low_steps = self.env.cfg.control.macro_steps
        horizon_range = horizon_low_steps_range or (horizon_low_steps, horizon_low_steps)
        min_macro_horizon = math.ceil(horizon_range[0] / low_steps)
        max_macro_horizon = math.ceil(horizon_range[1] / low_steps)
        horizon_generator = torch.Generator().manual_seed(self.seed + 17011)

        def append(name: str, value: torch.Tensor) -> None:
            records.setdefault(name, []).append(value.detach().cpu())

        next_anchor_id = anchor_offset
        end_anchor_id = anchor_offset + num_anchors
        last_report = anchor_offset
        report_return = torch.zeros((), device=zero.device)
        report_components = torch.zeros(len(REWARD_COMPONENTS), device=zero.device)
        report_retention = torch.zeros((), device=zero.device)
        report_contact_loss = torch.zeros((), device=zero.device)
        report_count = 0
        component_index = {name: index for index, name in enumerate(REWARD_COMPONENTS)}
        while next_anchor_id < end_anchor_id:
            local_anchor = next_anchor_id - anchor_offset
            if phase_plan is not None:
                planned_mode = int(phase_plan[local_anchor])
                next_mode = (phase_plan[local_anchor:] != planned_mode).nonzero().reshape(-1)
                phase_remaining = int(next_mode[0]) if len(next_mode) else len(phase_plan) - local_anchor
                batch_size = min(num_envs, end_anchor_id - next_anchor_id, phase_remaining)
                self._prepare_phase(planned_mode, self.seed + next_anchor_id, zero)
            else:
                planned_mode = None
                batch_size = min(num_envs, end_anchor_id - next_anchor_id)
            keep = slice(0, batch_size)
            snapshot = self.env.snapshot()
            anchor_features = self.env.latest_features
            anchor_obs = {name: value.to(self.env.z_current.device) for name, value in self.env.low_env.observation.items()}
            anchor_actor = self.env._stacked_actor_obs().clone()
            anchor_z = self.env.z_current.clone()
            anchor_previous_flow = self.env.previous_flow.clone()
            mode_ids = anchor_features.mode_id.long()
            candidates = self._candidates(candidates_per_anchor, mode_ids)
            macro_horizon = int(torch.randint(
                min_macro_horizon, max_macro_horizon + 1, (1,), generator=horizon_generator,
            ).item())
            if planned_mode in {int(SkateMode.MOUNT), int(SkateMode.DISMOUNT)}:
                macro_horizon = max_macro_horizon
            sampled_horizon_low_steps = macro_horizon * low_steps
            start_heading = self.env.low_env.husky_env.skateboard.data.heading_w.clone()
            start_target_heading = self.env.low_env.target_heading().clone()
            anchor_ids = torch.arange(next_anchor_id, next_anchor_id + batch_size, dtype=torch.long).unsqueeze(-1)
            for candidate_id in range(candidates_per_anchor):
                flow = candidates[:, candidate_id]
                self.env.restore(snapshot)
                total_return = torch.zeros(num_envs, 1, device=flow.device)
                components = None
                terminated = torch.zeros(num_envs, 1, dtype=torch.bool, device=flow.device)
                truncated = torch.zeros_like(terminated)
                fell_over = torch.zeros_like(terminated)
                illegal_contact = torch.zeros_like(terminated)
                active = torch.ones_like(terminated)
                board_progress = torch.zeros(num_envs, device=flow.device)
                for step in range(macro_horizon):
                    if step == 0:
                        step_flow = flow
                    elif planned_mode in {int(SkateMode.MOUNT), int(SkateMode.DISMOUNT)}:
                        current_modes = self.env.scheduler.mode(self.env.low_env.husky_env, self.env.latest_info)
                        step_flow = self._tracking_flow(current_modes).clamp(-1.0, 1.0)
                    else:
                        step_flow = zero
                    result = self.env.step(step_flow)
                    total_return += (self.env.cfg.control.gamma_macro ** step) * result.reward_macro * active
                    weighted_components = result.reward_components * active
                    components = weighted_components if components is None else components + weighted_components
                    terminated |= result.terminated
                    truncated |= result.truncated
                    fell_over |= result.diagnostics["fell_over"].unsqueeze(-1).bool()
                    illegal_contact |= result.diagnostics["illegal_contact"].unsqueeze(-1).bool()
                    board_progress += result.diagnostics["board_forward_progress"] * active.reshape(-1)
                    active &= ~(result.terminated | result.truncated)
                    if not active.any():
                        break
                mapped = self.env.mapper(anchor_z, mode_ids, flow)
                append("anchor_id", anchor_ids)
                append("candidate_id", torch.full((batch_size, 1), candidate_id, dtype=torch.long))
                append("mode_id", mode_ids[keep].unsqueeze(-1))
                append("flow_actor_obs", anchor_actor[keep])
                append("critic_robot", anchor_features.critic_robot[keep])
                append("critic_board", anchor_features.critic_board[keep])
                append("critic_contact", anchor_features.critic_contact[keep])
                append("critic_goal_mode", anchor_features.critic_goal_mode[keep])
                for name, value in anchor_obs.items():
                    append(f"bfm_{name}", value[keep])
                append("z_current", anchor_z[keep])
                append("flow", flow[keep])
                append("previous_flow", anchor_previous_flow[keep])
                append("z_candidate", mapped.z_candidate[keep])
                append("latent_stats", torch.stack((mapped.delta_norm, mapped.cosine), dim=-1)[keep])
                append("finite_horizon_return", total_return[keep])
                append("horizon_low_steps", torch.full(
                    (batch_size, 1), sampled_horizon_low_steps, dtype=torch.long,
                ))
                append("reward_components", components[keep])
                append("fall", fell_over[keep].float())
                board_contact = self.env.latest_info["feet_board_contact"].any(dim=-1, keepdim=True)
                contact_loss = (~board_contact[keep]).float()
                append("contact_loss", contact_loss)
                append("illegal_contact", illegal_contact[keep].float())
                append("board_progress", board_progress[keep].unsqueeze(-1))
                final_heading = self.env.low_env.husky_env.skateboard.data.heading_w
                final_target_heading = self.env.low_env.target_heading()
                start_error = torch.atan2(
                    torch.sin(start_target_heading - start_heading),
                    torch.cos(start_target_heading - start_heading),
                ).abs()
                final_error = torch.atan2(
                    torch.sin(final_target_heading - final_heading),
                    torch.cos(final_target_heading - final_heading),
                ).abs()
                heading_progress = start_error - final_error
                append("heading_progress", heading_progress[keep].unsqueeze(-1))
                horizon_executed_low_steps = max(1, macro_horizon * low_steps)
                append("retention", components[keep, 8:9] / horizon_executed_low_steps)
                append("terminated", terminated[keep])
                append("truncated", truncated[keep])
                report_return += total_return[keep].sum()
                report_components += components[keep].sum(dim=0)
                report_retention += (
                    components[keep, component_index["retention"]] / horizon_executed_low_steps
                ).sum()
                report_contact_loss += contact_loss.sum()
                report_count += batch_size
            if phase_plan is None:
                self.env.restore(snapshot)
                advance = self.env.step(zero)
                done_ids = (advance.terminated | advance.truncated).reshape(-1).nonzero().reshape(-1)
                if len(done_ids):
                    self.env.reset(self.seed + next_anchor_id + 1, done_ids)
            next_anchor_id += batch_size
            if last_report == anchor_offset or next_anchor_id - last_report >= log_interval or next_anchor_id == end_anchor_id:
                completed = next_anchor_id - anchor_offset
                elapsed = max(time.perf_counter() - started, 1e-6)
                rate = completed * candidates_per_anchor / elapsed
                remaining = max(0, num_anchors - completed) * candidates_per_anchor
                eta_seconds = remaining / rate if rate > 0 else 0.0
                eta_minutes, eta_secs = divmod(int(eta_seconds), 60)
                eta_hours, eta_minutes = divmod(eta_minutes, 60)
                progress = completed / max(1, num_anchors)
                filled = min(30, round(progress * 30))
                bar = "#" * filled + "-" * (30 - filled)
                means = report_components / max(1, report_count)
                print(
                    f"[branch {shard_index + 1}/{num_shards}] [{bar}] {progress * 100:5.1f}% "
                    f"anchors={completed}/{num_anchors} global={next_anchor_id - 1}/{end_anchor_id - 1} "
                    f"candidates={completed * candidates_per_anchor} rate={rate:.1f}/s "
                    f"phase={MODE_NAMES[planned_mode] if planned_mode is not None else 'reactive'} "
                    f"horizon={sampled_horizon_low_steps / self.env.cfg.control.bfm_hz:.1f}s "
                    f"ETA={eta_hours:02d}:{eta_minutes:02d}:{eta_secs:02d} | "
                    f"reward={float(report_return / max(1, report_count)):.3f} "
                    f"push={float(means[component_index['push_total']]):.3f} "
                    f"steer={float(means[component_index['steer_total']]):.3f} "
                    f"transition={float(means[component_index['transition_total']]):.3f} "
                    f"regularization={float(means[component_index['regularization_total']]):.3f} "
                    f"retention={float(report_retention / max(1, report_count)):.3f} "
                    f"contact_loss={float(report_contact_loss / max(1, report_count)):.3f}",
                    flush=True,
                )
                last_report = next_anchor_id
                report_return.zero_()
                report_components.zero_()
                report_retention.zero_()
                report_contact_loss.zero_()
                report_count = 0
        tensors = {name: torch.cat(values, dim=0) for name, values in records.items()}
        anchor_modes = tensors["mode_id"][tensors["candidate_id"].reshape(-1) == 0].reshape(-1)
        phase_counts = {
            MODE_NAMES[mode]: int((anchor_modes == mode).sum())
            for mode in range(len(MODE_NAMES))
            if (anchor_modes == mode).any()
        }
        return BranchDataset(tensors, {
            "num_anchors": num_anchors, "candidates_per_anchor": candidates_per_anchor,
            "horizon_low_steps": [min_macro_horizon * low_steps, max_macro_horizon * low_steps],
            "candidate_hold_low_steps": low_steps,
            "branch_action_semantics": BRANCH_ACTION_SEMANTICS,
            "parallel_envs": num_envs,
            "anchor_offset": anchor_offset,
            "phase_sampling": PHASE_SAMPLING_VERSION if phase_plan is not None else "reactive-v1",
            "phase_anchor_counts": phase_counts,
            "basis_path": self.env.cfg.paths.basis_path,
            "basis_sha256": self.env.basis_metadata["sha256"],
        })
