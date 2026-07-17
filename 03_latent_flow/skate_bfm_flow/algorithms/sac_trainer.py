from __future__ import annotations

import torch

from ..bfm.action_preview import FrozenBfmActionPreview
from ..bfm.latent_mapper import LatentMapper
from ..data.batch import TensorBatch
from ..models.flow_policy import FlowPolicy
from ..models.skate_q import TwinSkateQ
from ..q.aggregators import aggregate
from ..q.input_builder import QInputBuilder
from ..q.losses import critic_loss
from ..q.targets import td_target
from .offline_q_trainer import bfm_obs_from_batch, features_from_batch
from .updates import grad_norm, soft_update


class SacUpdater:
    def __init__(self, policy: FlowPolicy, q: TwinSkateQ, target_q: TwinSkateQ, mapper: LatentMapper, preview: FrozenBfmActionPreview, input_builder: QInputBuilder, q_optimizer: torch.optim.Optimizer, policy_optimizer: torch.optim.Optimizer, alpha: torch.Tensor, alpha_optimizer: torch.optim.Optimizer | None, gamma: float = 0.99, aggregation: str = "min", beta: float = 0.5, loss_name: str = "huber", huber_delta: float = 1.0, tau: float = 0.005, flow_magnitude: float = 0.001, flow_smoothness: float = 0.01, grad_clip: float = 10.0) -> None:
        self.policy, self.q, self.target_q = policy, q, target_q
        self.mapper, self.preview, self.input_builder = mapper, preview, input_builder
        self.q_optimizer, self.policy_optimizer = q_optimizer, policy_optimizer
        self.log_alpha = torch.nn.Parameter(alpha.log())
        if alpha_optimizer is None:
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        else:
            self.alpha_optimizer = alpha_optimizer
        self.gamma, self.aggregation, self.beta = gamma, aggregation, beta
        self.loss_name, self.huber_delta, self.tau = loss_name, huber_delta, tau
        self.flow_magnitude, self.flow_smoothness, self.grad_clip = flow_magnitude, flow_smoothness, grad_clip

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _input(self, batch: TensorBatch, prefix: str, flow: torch.Tensor, z_candidate: torch.Tensor, preview: torch.Tensor, latent_stats: torch.Tensor) -> object:
        features = features_from_batch(batch, prefix)
        return self.input_builder.build(
            features, batch[f"{prefix}z_current"], flow, batch[f"{prefix}previous_flow"], z_candidate, latent_stats, preview,
        )

    def update(self, batch: TensorBatch) -> dict[str, float]:
        mode = batch["mode_id"].reshape(-1).long()
        mapped = self.mapper(batch["z_current"], mode, batch["flow"])
        preview = self.preview(bfm_obs_from_batch(batch), mapped.z_candidate)
        stats = torch.stack((mapped.delta_norm, mapped.cosine), dim=-1)
        current_input = self._input(batch, "", batch["flow"], mapped.z_candidate, preview, stats)
        with torch.no_grad():
            next_flow_sample = self.policy.sample(batch["next_flow_actor_obs"])
            next_mode = batch["next_mode_id"].reshape(-1).long()
            next_mapped = self.mapper(batch["next_z_current"], next_mode, next_flow_sample.action)
            next_preview = self.preview(bfm_obs_from_batch(batch, "next_"), next_mapped.z_candidate)
            next_stats = torch.stack((next_mapped.delta_norm, next_mapped.cosine), dim=-1)
            next_input = self._input(batch, "next_", next_flow_sample.action, next_mapped.z_candidate, next_preview, next_stats)
            next_q1, next_q2 = self.target_q(next_input)
            target = td_target(
                batch["reward_macro"], batch["terminated"], batch["truncated"], next_q1, next_q2,
                self.gamma, self.aggregation, self.beta, True, next_flow_sample.log_prob, self.alpha,
            ).target
        q1, q2 = self.q(current_input)
        q1_loss = critic_loss(q1, target, self.loss_name, self.huber_delta)
        q2_loss = critic_loss(q2, target, self.loss_name, self.huber_delta)
        q_loss = q1_loss + q2_loss
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.q_optimizer.step()

        sample = self.policy.sample(batch["flow_actor_obs"])
        policy_mapped = self.mapper(batch["z_current"], mode, sample.action)
        policy_preview = self.preview(bfm_obs_from_batch(batch), policy_mapped.z_candidate)
        policy_stats = torch.stack((policy_mapped.delta_norm, policy_mapped.cosine), dim=-1)
        policy_input = self._input(batch, "", sample.action, policy_mapped.z_candidate, policy_preview, policy_stats)
        policy_q1, policy_q2 = self.q(policy_input)
        policy_q = aggregate(policy_q1, policy_q2, self.aggregation, self.beta)
        policy_loss = (self.alpha.detach() * sample.log_prob - policy_q).mean()
        policy_loss = policy_loss + self.flow_magnitude * sample.action.square().mean() + self.flow_smoothness * (sample.action - batch["previous_flow"]).square().mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (sample.log_prob.detach() + self.policy.flow_dim)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        soft_update(self.q, self.target_q, self.tau)
        self.preview.policy.assert_frozen()
        return {
            "q1_loss": float(q1_loss.detach()), "q2_loss": float(q2_loss.detach()), "target_mean": float(target.mean().detach()),
            "q1_mean": float(q1.mean().detach()), "q2_mean": float(q2.mean().detach()), "q_disagreement_mean": float((q1 - q2).abs().mean().detach()),
            "actor_loss": float(policy_loss.detach()), "entropy": float(-sample.log_prob.mean().detach()), "alpha": float(self.alpha.detach()),
            "alpha_loss": float(alpha_loss.detach()), "actor_flow_norm": float(sample.action.norm(dim=-1).mean().detach()),
            "gradient_norm": grad_norm(self.policy),
        }
