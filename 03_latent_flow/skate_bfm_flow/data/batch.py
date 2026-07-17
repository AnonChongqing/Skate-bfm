from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TensorBatch:
    tensors: dict[str, torch.Tensor]

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def to(self, device: str | torch.device) -> "TensorBatch":
        return TensorBatch({name: value.to(device) for name, value in self.tensors.items()})

    def select(self, indices: torch.Tensor) -> "TensorBatch":
        return TensorBatch({name: value[indices] for name, value in self.tensors.items()})

    def __len__(self) -> int:
        return next(iter(self.tensors.values())).shape[0] if self.tensors else 0
