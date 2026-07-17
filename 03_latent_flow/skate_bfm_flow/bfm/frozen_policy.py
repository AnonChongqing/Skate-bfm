from __future__ import annotations

import inspect
from pathlib import Path

import torch

from bfm_zero_inference_code.fb_cpr_aux.model import FBcprAuxModel


class FrozenBfmPolicy(torch.nn.Module):
    def __init__(self, model_dir: str | Path, device: str, mean: bool = True) -> None:
        super().__init__()
        expected = Path(__file__).resolve().parents[2] / "vendor"
        source = Path(inspect.getfile(FBcprAuxModel)).resolve()
        if not source.is_relative_to(expected):
            raise RuntimeError(f"BFM source must come from Stage 03 vendor, got {source}")
        bfm_device = "cuda" if device.startswith("cuda") else "cpu"
        self.model = FBcprAuxModel.load(str(Path(model_dir) / "checkpoint" / "model"), device=bfm_device)
        self.model.eval().requires_grad_(False)
        self.device_name = device
        self.mean = mean

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        batch = {key: value.to(self.device_name) for key, value in obs.items()}
        if next(iter(batch.values())).ndim == 1:
            batch = {key: value.unsqueeze(0) for key, value in batch.items()}
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return self.model.act(batch, z.to(self.device_name), mean=self.mean)

    @torch.no_grad()
    def tracking(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model.tracking_inference({key: value.to(self.device_name) for key, value in obs.items()})

    def assert_frozen(self) -> None:
        if self.model.training or any(parameter.requires_grad for parameter in self.model.parameters()):
            raise AssertionError("BFM0 must remain eval-only and frozen")
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise AssertionError("BFM0 received gradients")
