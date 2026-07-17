from __future__ import annotations

import torch


def true_termination(terminated: torch.Tensor, truncated: torch.Tensor, bootstrap_on_timeout: bool = True) -> torch.Tensor:
    if bootstrap_on_timeout:
        return terminated.bool()
    return torch.logical_or(terminated.bool(), truncated.bool())
