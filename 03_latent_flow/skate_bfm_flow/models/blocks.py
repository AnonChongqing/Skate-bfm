from __future__ import annotations

import torch
from torch import nn


def activation(name: str) -> nn.Module:
    return {"elu": nn.ELU, "relu": nn.ReLU, "silu": nn.SiLU}[name]()


def mlp(dims: list[int], activation_name: str = "elu", layer_norm: bool = False, final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
        final = index == len(dims) - 2
        layers.append(nn.Linear(input_dim, output_dim))
        if not final or final_activation:
            if layer_norm:
                layers.append(nn.LayerNorm(output_dim))
            layers.append(activation(activation_name))
    return nn.Sequential(*layers)


def finite(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values in {name}")
    return tensor
