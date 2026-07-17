from __future__ import annotations

from pathlib import Path

import torch

from ..schemas import FEATURE_SCHEMA_VERSION
from .batch import TensorBatch
from .storage import atomic_torch_save


class BranchDataset:
    def __init__(self, tensors: dict[str, torch.Tensor], metadata: dict | None = None) -> None:
        lengths = {value.shape[0] for value in tensors.values()}
        if len(lengths) != 1:
            raise ValueError(f"Branch tensor lengths differ: {lengths}")
        self.tensors = tensors
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return next(iter(self.tensors.values())).shape[0]

    def batch(self, indices: torch.Tensor) -> TensorBatch:
        return TensorBatch({name: value[indices] for name, value in self.tensors.items()})

    def anchor_split(self, validation_fraction: float, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
        anchors = torch.unique(self.tensors["anchor_id"]).cpu()
        generator = torch.Generator().manual_seed(seed)
        anchors = anchors[torch.randperm(len(anchors), generator=generator)]
        validation_count = max(1, round(len(anchors) * validation_fraction))
        validation_anchors = anchors[:validation_count]
        validation_mask = torch.isin(self.tensors["anchor_id"].cpu(), validation_anchors)
        return (~validation_mask).nonzero().reshape(-1), validation_mask.nonzero().reshape(-1)

    def save(self, path: str | Path) -> None:
        atomic_torch_save({
            "tensors": {name: value.cpu() for name, value in self.tensors.items()},
            "metadata": {**self.metadata, "feature_schema_version": FEATURE_SCHEMA_VERSION},
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "BranchDataset":
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload["metadata"].get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("Branch dataset feature schema mismatch")
        return cls(payload["tensors"], payload["metadata"])
