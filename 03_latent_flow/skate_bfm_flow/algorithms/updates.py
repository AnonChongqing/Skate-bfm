from __future__ import annotations

import torch


@torch.no_grad()
def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
        target_parameter.lerp_(source_parameter, tau)


def grad_norm(module: torch.nn.Module) -> float:
    norms = [parameter.grad.detach().norm() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0
