from __future__ import annotations

import torch

from ..bfm.action_preview import FrozenBfmActionPreview
from ..bfm.latent_mapper import LatentMapper
from ..data.batch import TensorBatch
from ..models.skate_q import TwinSkateQ
from ..q.input_builder import QInputBuilder
from ..q.losses import critic_loss
from ..schemas import FeatureBatch
from .updates import grad_norm


def features_from_batch(batch: TensorBatch, prefix: str = "") -> FeatureBatch:
    return FeatureBatch(
        actor_frame=batch[f"{prefix}flow_actor_obs"], critic_robot=batch[f"{prefix}critic_robot"],
        critic_board=batch[f"{prefix}critic_board"], critic_contact=batch[f"{prefix}critic_contact"],
        critic_goal_mode=batch[f"{prefix}critic_goal_mode"], mode_id=batch[f"{prefix}mode_id"].reshape(-1),
    )


def bfm_obs_from_batch(batch: TensorBatch, prefix: str = "") -> dict[str, torch.Tensor]:
    return {name: batch[f"{prefix}bfm_{name}"] for name in ("state", "history_actor", "last_action", "privileged_state")}


class OfflineQTrainer:
    def __init__(self, q: TwinSkateQ, mapper: LatentMapper, preview: FrozenBfmActionPreview, input_builder: QInputBuilder, optimizer: torch.optim.Optimizer, loss_name: str = "huber", huber_delta: float = 1.0, grad_clip: float = 10.0) -> None:
        self.q = q
        self.mapper = mapper
        self.preview = preview
        self.input_builder = input_builder
        self.optimizer = optimizer
        self.loss_name = loss_name
        self.huber_delta = huber_delta
        self.grad_clip = grad_clip

    def q_input(self, batch: TensorBatch):
        mode = batch["mode_id"].reshape(-1).long()
        mapped = self.mapper(batch["z_current"], mode, batch["flow"])
        action_preview = self.preview(bfm_obs_from_batch(batch), mapped.z_candidate)
        latent_stats = torch.stack((mapped.delta_norm, mapped.cosine), dim=-1)
        return self.input_builder.build(
            features_from_batch(batch), batch["z_current"], batch["flow"], batch["previous_flow"],
            mapped.z_candidate, latent_stats, action_preview,
        )

    def update(self, batch: TensorBatch) -> dict[str, float]:
        inputs = self.q_input(batch)
        target = batch["finite_horizon_return"].detach()
        q1, q2 = self.q(inputs)
        q1_loss = critic_loss(q1, target, self.loss_name, self.huber_delta)
        q2_loss = critic_loss(q2, target, self.loss_name, self.huber_delta)
        loss = q1_loss + q2_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optimizer.step()
        self.preview.policy.assert_frozen()
        return {
            "q1_loss": float(q1_loss.detach()), "q2_loss": float(q2_loss.detach()), "q1_mean": float(q1.mean().detach()),
            "q2_mean": float(q2.mean().detach()), "target_mean": float(target.mean().detach()),
            "td_error_abs_mean": float(((q1 + q2) * 0.5 - target).abs().mean().detach()),
            "q_disagreement_mean": float((q1 - q2).abs().mean().detach()), "gradient_norm": grad_norm(self.q),
        }
