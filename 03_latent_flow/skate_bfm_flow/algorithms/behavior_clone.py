from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.branch_dataset import BranchDataset
from ..models.flow_policy import FlowPolicy


def best_flow_targets(dataset: BranchDataset, target_type: str = "hard_best", temperature: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    actor_obs, targets = [], []
    for anchor in torch.unique(dataset.tensors["anchor_id"]):
        indices = (dataset.tensors["anchor_id"] == anchor).reshape(-1).nonzero().reshape(-1)
        returns = dataset.tensors["finite_horizon_return"][indices].reshape(-1)
        flows = dataset.tensors["flow"][indices]
        if target_type == "hard_best":
            target = flows[returns.argmax()]
        elif target_type == "soft_weighted":
            target = (torch.softmax(returns / temperature, dim=0).unsqueeze(-1) * flows).sum(0)
        else:
            raise ValueError(f"Unknown BC target: {target_type}")
        actor_obs.append(dataset.tensors["flow_actor_obs"][indices[0]])
        targets.append(target)
    return torch.stack(actor_obs), torch.stack(targets)


def bc_update(policy: FlowPolicy, optimizer: torch.optim.Optimizer, actor_obs: torch.Tensor, target_flow: torch.Tensor) -> float:
    prediction = policy.sample(actor_obs, deterministic=True).mean_action
    loss = F.mse_loss(prediction, target_flow)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
