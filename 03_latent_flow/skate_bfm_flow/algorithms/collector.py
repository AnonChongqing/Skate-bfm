from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from ..data.branch_dataset import BranchDataset
from ..env.macro_env import LatentFlowMacroEnv
from ..env.reward_adapter import REWARD_COMPONENTS


@dataclass
class BranchCollector:
    env: LatentFlowMacroEnv
    seed: int = 42

    def _candidates(self, count: int, mode_ids: torch.Tensor) -> torch.Tensor:
        device = self.env.z_current.device
        flow_dim = self.env.cfg.latent.flow_dim
        num_envs = len(mode_ids)
        generator = torch.Generator(device=device).manual_seed(self.seed + int(self.env.low_env.husky_env.common_step_counter))
        candidates = [torch.zeros(num_envs, flow_dim, device=device)]
        local_count = max(1, round((count - 1) * 0.5))
        for _ in range(local_count):
            candidates.append(torch.randn(num_envs, flow_dim, generator=generator, device=device) * self.env.cfg.branch.local_std)
        basis = self.env.mapper.basis.index_select(0, mode_ids)
        target = self.env.prototypes.index_select(0, mode_ids) - self.env.z_current
        prototype_flow = torch.bmm(basis.transpose(1, 2), target.unsqueeze(-1)).squeeze(-1) / self.env.cfg.latent.step_size
        candidates.append(prototype_flow)
        while len(candidates) < count:
            candidates.append(torch.empty(num_envs, flow_dim, device=device).uniform_(-1.0, 1.0, generator=generator))
        return torch.stack(candidates[:count], dim=1).clamp(-1.0, 1.0)

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
        self.env.reset(self.seed)
        num_envs = self.env.low_env.husky_env.num_envs
        zero = torch.zeros(num_envs, self.env.cfg.latent.flow_dim, device=self.env.z_current.device)
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
                active = torch.ones_like(terminated)
                board_progress = torch.zeros(num_envs, device=flow.device)
                for step in range(macro_horizon):
                    result = self.env.step(flow if step == 0 else zero)
                    total_return += (self.env.cfg.control.gamma_macro ** step) * result.reward_macro * active
                    weighted_components = result.reward_components * active
                    components = weighted_components if components is None else components + weighted_components
                    terminated |= result.terminated
                    truncated |= result.truncated
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
                append("fall", terminated[keep].float())
                board_contact = self.env.latest_info["feet_board_contact"].any(dim=-1, keepdim=True)
                contact_loss = (~board_contact[keep]).float()
                append("contact_loss", contact_loss)
                append("illegal_contact", components[keep, 11:12])
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
        return BranchDataset(tensors, {
            "num_anchors": num_anchors, "candidates_per_anchor": candidates_per_anchor,
            "horizon_low_steps": [min_macro_horizon * low_steps, max_macro_horizon * low_steps],
            "candidate_hold_low_steps": low_steps,
            "branch_action_semantics": "single_macro_then_zero",
            "parallel_envs": num_envs,
            "anchor_offset": anchor_offset,
            "basis_path": self.env.cfg.paths.basis_path,
            "basis_sha256": self.env.basis_metadata["sha256"],
        })
