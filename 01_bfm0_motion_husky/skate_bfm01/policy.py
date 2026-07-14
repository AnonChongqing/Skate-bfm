from __future__ import annotations

import inspect
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from bfm_zero_inference_code.fb_cpr_aux.model import FBcprAuxModel


def load_data(path: Path):
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except Exception:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(f"Could not read {path}; joblib is not installed.") from exc
        return joblib.load(path)


def load_zs(path: Path, key: str | None = None) -> torch.Tensor:
    if path.suffix == ".npy":
        value = np.load(path)
    else:
        value = load_data(path)
        if isinstance(value, dict):
            key = key or next(iter(value))
            value = value[key]
    if isinstance(value, (list, tuple)):
        value = torch.cat([torch.as_tensor(item).reshape(-1, torch.as_tensor(item).shape[-1]) for item in value])
    z = torch.as_tensor(value, dtype=torch.float32)
    return z.reshape(-1, z.shape[-1])


def load_z(path: Path, key: str | None = None, index: int = 0) -> torch.Tensor:
    zs = load_zs(path, key)
    if not 0 <= index < len(zs):
        raise IndexError(f"Latent index {index} is outside [0, {len(zs) - 1}]")
    return zs[index]


class BfmPolicy:
    def __init__(
        self,
        model_path: Path,
        z_path: Path,
        z_key: str | None,
        device: str,
        mean: bool,
        z_index: int = 0,
    ) -> None:
        local_bfm = Path(__file__).resolve().parents[1] / "bfm0"
        model_source = Path(inspect.getfile(FBcprAuxModel)).resolve()
        if not model_source.is_relative_to(local_bfm):
            raise RuntimeError(f"BFM0 was imported outside Skate-bfm: {model_source}")
        self.device = device
        self.mean = mean
        self.model = FBcprAuxModel.load(str(model_path / "checkpoint" / "model"), device=device)
        self.z = load_z(z_path, z_key, z_index).to(device)

    def _batch(self, obs: dict[str, torch.Tensor]):
        return tree_map(
            lambda value: value.to(self.device).unsqueeze(0)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=self.device).unsqueeze(0),
            obs,
        )

    def act(self, obs: dict[str, torch.Tensor], z: torch.Tensor | None = None) -> torch.Tensor:
        z = self.z if z is None else z.to(self.device)
        z = z.unsqueeze(0) if z.ndim == 1 else z
        with torch.no_grad():
            return self.model.act(self._batch(obs), z, mean=self.mean)

    def infer_goal(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self.model.goal_inference(self._batch(obs))[0]

    def infer_tracking(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Infer BFM0's per-frame prompts for a batched target trajectory."""
        batch = tree_map(
            lambda value: value.to(self.device)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=self.device),
            obs,
        )
        with torch.no_grad():
            return self.model.tracking_inference(batch)

    def infer_reward(
        self,
        obs: dict[str, torch.Tensor],
        reward: torch.Tensor,
        weighted: bool = True,
    ) -> torch.Tensor:
        """Run BFM0 reward inference on a batch of HUSKY-adapted observations."""
        batch = tree_map(
            lambda value: value.to(self.device)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=self.device),
            obs,
        )
        reward = reward.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            if weighted:
                return self.model.reward_wr_inference(batch, reward)
            return self.model.reward_inference(batch, reward)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.project_z(z.unsqueeze(0))[0]

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.act(obs)
