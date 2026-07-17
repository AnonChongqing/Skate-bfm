from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..data.branch_dataset import BranchDataset
from ..env.macro_env import LatentFlowMacroEnv


@dataclass
class BranchCollector:
    env: LatentFlowMacroEnv
    seed: int = 42

    def _candidates(self, count: int, mode_id: int) -> torch.Tensor:
        device = self.env.z_current.device
        flow_dim = self.env.cfg.latent.flow_dim
        generator = torch.Generator(device=device).manual_seed(self.seed + int(self.env.low_env.husky_env.common_step_counter))
        candidates = [torch.zeros(flow_dim, device=device)]
        local_count = max(1, round((count - 1) * 0.5))
        for _ in range(local_count):
            candidates.append(torch.randn(flow_dim, generator=generator, device=device) * self.env.cfg.branch.local_std)
        basis = self.env.mapper.basis[mode_id]
        target = self.env.prototypes[mode_id] - self.env.z_current[0]
        prototype_flow = basis.T @ target / self.env.cfg.latent.step_size
        candidates.append(prototype_flow)
        while len(candidates) < count:
            candidates.append(torch.empty(flow_dim, device=device).uniform_(-1.0, 1.0, generator=generator))
        return torch.stack(candidates[:count]).clamp(-1.0, 1.0)

    def collect(self, num_anchors: int, candidates_per_anchor: int, horizon_low_steps: int) -> BranchDataset:
        self.env.reset(self.seed)
        zero = torch.zeros(1, self.env.cfg.latent.flow_dim, device=self.env.z_current.device)
        for _ in range(math.ceil(self.env.cfg.branch.warmup_low_steps / self.env.cfg.control.macro_steps)):
            result = self.env.step(zero)
            if result.terminated.any() or result.truncated.any():
                self.env.reset(self.seed)
        records: dict[str, list[torch.Tensor]] = {}
        macro_horizon = math.ceil(horizon_low_steps / self.env.cfg.control.macro_steps)

        def append(name: str, value: torch.Tensor) -> None:
            records.setdefault(name, []).append(value.detach().cpu())

        for anchor_id in range(num_anchors):
            snapshot = self.env.snapshot()
            anchor_features = self.env.latest_features
            anchor_obs = {name: value.unsqueeze(0).to(self.env.z_current.device) for name, value in self.env.low_env.observation.items()}
            anchor_actor = self.env._stacked_actor_obs().clone()
            anchor_z = self.env.z_current.clone()
            anchor_previous_flow = self.env.previous_flow.clone()
            mode_id = int(anchor_features.mode_id.item())
            candidates = self._candidates(candidates_per_anchor, mode_id)
            start_board_x = self.env.low_env.husky_env.skateboard.data.root_link_pos_w[0, 0].clone()
            start_heading = float(self.env.latest_info["skateboard_heading_w"]) if self.env.latest_info else 0.0
            for candidate_id, flow in enumerate(candidates):
                self.env.restore(snapshot)
                total_return = torch.zeros(1, 1, device=flow.device)
                components = None
                terminated = torch.zeros(1, 1, dtype=torch.bool, device=flow.device)
                truncated = torch.zeros_like(terminated)
                for step in range(macro_horizon):
                    result = self.env.step(flow.unsqueeze(0))
                    total_return += (self.env.cfg.control.gamma_macro ** step) * result.reward_macro
                    components = result.reward_components if components is None else components + result.reward_components
                    terminated |= result.terminated
                    truncated |= result.truncated
                    if terminated.any() or truncated.any():
                        break
                mapped = self.env.mapper(anchor_z, torch.tensor([mode_id], device=flow.device), flow.unsqueeze(0))
                append("anchor_id", torch.tensor([[anchor_id]], dtype=torch.long))
                append("candidate_id", torch.tensor([[candidate_id]], dtype=torch.long))
                append("mode_id", torch.tensor([[mode_id]], dtype=torch.long))
                append("flow_actor_obs", anchor_actor)
                append("critic_robot", anchor_features.critic_robot)
                append("critic_board", anchor_features.critic_board)
                append("critic_contact", anchor_features.critic_contact)
                append("critic_goal_mode", anchor_features.critic_goal_mode)
                for name, value in anchor_obs.items():
                    append(f"bfm_{name}", value)
                append("z_current", anchor_z)
                append("flow", flow.unsqueeze(0))
                append("previous_flow", anchor_previous_flow)
                append("z_candidate", mapped.z_candidate)
                append("latent_stats", torch.stack((mapped.delta_norm, mapped.cosine), dim=-1))
                append("finite_horizon_return", total_return)
                append("reward_components", components)
                append("fall", terminated.float())
                append("contact_loss", torch.tensor([[float(not any(self.env.latest_info["feet_board_contact"]))]]))
                append("illegal_contact", components[:, 11:12])
                final_x = self.env.low_env.husky_env.skateboard.data.root_link_pos_w[0, 0]
                append("board_progress", (final_x - start_board_x).reshape(1, 1))
                final_heading = float(self.env.latest_info["skateboard_heading_w"])
                append("heading_progress", torch.tensor([[abs(start_heading - self.env.cfg.env.command_heading) - abs(final_heading - self.env.cfg.env.command_heading)]]))
                append("retention", components[:, 8:9] / max(1, macro_horizon))
                append("terminated", terminated)
                append("truncated", truncated)
            self.env.restore(snapshot)
            advance = self.env.step(zero)
            if advance.terminated.any() or advance.truncated.any():
                self.env.reset(self.seed + anchor_id + 1)
        tensors = {name: torch.cat(values, dim=0) for name, values in records.items()}
        return BranchDataset(tensors, {
            "num_anchors": num_anchors, "candidates_per_anchor": candidates_per_anchor,
            "horizon_low_steps": horizon_low_steps, "basis_path": self.env.cfg.paths.basis_path,
        })
