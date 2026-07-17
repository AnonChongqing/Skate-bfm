from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.branch_dataset import BranchDataset
from ..models.flow_policy import FlowPolicy


def best_flow_targets(dataset: BranchDataset, target_type: str = "hard_best", temperature: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    groups, _ = dataset.grouped_indices()
    returns = dataset.tensors["finite_horizon_return"][groups].squeeze(-1)
    flows = dataset.tensors["flow"][groups]
    if target_type == "hard_best":
        rows = torch.arange(len(groups), device=groups.device)
        targets = flows[rows, returns.argmax(dim=1)]
    elif target_type == "soft_weighted":
        targets = (torch.softmax(returns / temperature, dim=1).unsqueeze(-1) * flows).sum(dim=1)
    else:
        raise ValueError(f"Unknown BC target: {target_type}")
    return dataset.tensors["flow_actor_obs"][groups[:, 0]], targets


def bc_update(policy: FlowPolicy, optimizer: torch.optim.Optimizer, actor_obs: torch.Tensor, target_flow: torch.Tensor) -> float:
    prediction = policy.sample(actor_obs, deterministic=True).mean_action
    loss = F.mse_loss(prediction, target_flow)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
