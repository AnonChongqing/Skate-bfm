from __future__ import annotations

import torch

from ..schemas import TargetOutput
from .aggregators import aggregate


def terminal_mask(terminated: torch.Tensor, truncated: torch.Tensor, bootstrap_on_timeout: bool) -> torch.Tensor:
    done = terminated.bool()
    if not bootstrap_on_timeout:
        done = torch.logical_or(done, truncated.bool())
    return done.float()


def finite_horizon_target(return_value: torch.Tensor) -> TargetOutput:
    detached = return_value.detach()
    return TargetOutput(target=detached, bootstrap=torch.zeros_like(detached))


def td_target(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    next_q1: torch.Tensor,
    next_q2: torch.Tensor,
    gamma: float,
    aggregation: str = "min",
    beta: float = 0.5,
    bootstrap_on_timeout: bool = True,
    log_prob: torch.Tensor | None = None,
    alpha: torch.Tensor | float = 0.0,
) -> TargetOutput:
    with torch.no_grad():
        bootstrap = aggregate(next_q1, next_q2, aggregation, beta)
        if log_prob is not None:
            bootstrap = bootstrap - alpha * log_prob
        alive = 1.0 - terminal_mask(terminated, truncated, bootstrap_on_timeout)
        target = reward + gamma * alive * bootstrap
    return TargetOutput(target=target, bootstrap=bootstrap)
