from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch

from ..schemas import FEATURE_SCHEMA_VERSION
from .batch import TensorBatch


class TensorReplayBuffer:
    """Preallocated tensor replay with validated named fields."""

    def __init__(self, capacity: int, field_shapes: dict[str, tuple[int, ...]], dtypes: dict[str, torch.dtype] | None = None, device: str = "cpu") -> None:
        self.capacity = int(capacity)
        self.field_shapes = dict(field_shapes)
        self.dtypes = dtypes or {}
        self.device = torch.device(device)
        self.storage = {
            name: torch.empty((capacity, *shape), dtype=self.dtypes.get(name, torch.float32), device=self.device)
            for name, shape in field_shapes.items()
        }
        self.position = 0
        self.size = 0

    @classmethod
    def from_example(cls, capacity: int, example: dict[str, torch.Tensor], device: str = "cpu") -> "TensorReplayBuffer":
        shapes = {name: tuple(value.shape[1:]) for name, value in example.items()}
        dtypes = {name: value.dtype for name, value in example.items()}
        return cls(capacity, shapes, dtypes, device)

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        if set(transition) != set(self.storage):
            raise ValueError(f"Replay fields differ: expected {set(self.storage)}, got {set(transition)}")
        batch_size = next(iter(transition.values())).shape[0]
        for name, value in transition.items():
            if tuple(value.shape) != (batch_size, *self.field_shapes[name]):
                raise ValueError(f"Replay field {name} expected [B,{self.field_shapes[name]}], got {tuple(value.shape)}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite replay field {name}")
        indices = torch.arange(self.position, self.position + batch_size, device=self.device) % self.capacity
        for name, value in transition.items():
            self.storage[name][indices] = value.to(self.device)
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, batch_size: int, mode_balanced: bool = False) -> TensorBatch:
        if self.size < batch_size:
            raise ValueError(f"Replay contains {self.size}, cannot sample {batch_size}")
        if not mode_balanced or "mode_id" not in self.storage:
            indices = torch.randint(self.size, (batch_size,), device=self.device)
        else:
            chunks = []
            per_mode = max(1, batch_size // 5)
            modes = self.storage["mode_id"][:self.size].reshape(-1)
            for mode in range(5):
                available = (modes == mode).nonzero().reshape(-1)
                if not len(available):
                    warnings.warn(f"Replay has no samples for mode {mode}")
                    continue
                chunks.append(available[torch.randint(len(available), (per_mode,), device=self.device)])
            indices = torch.cat(chunks) if chunks else torch.randint(self.size, (batch_size,), device=self.device)
            if len(indices) < batch_size:
                indices = torch.cat((indices, torch.randint(self.size, (batch_size - len(indices),), device=self.device)))
            indices = indices[:batch_size]
        return TensorBatch({name: value[indices] for name, value in self.storage.items()})

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity, "position": self.position, "size": self.size,
            "field_shapes": self.field_shapes, "dtypes": {name: str(dtype) for name, dtype in self.dtypes.items()},
            "storage": {name: value[:self.size].cpu() for name, value in self.storage.items()},
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"replay": self.state_dict(), "metadata": metadata or {}}, path)
        path.with_suffix(".json").write_text(json.dumps({"size": self.size, "capacity": self.capacity, **(metadata or {})}, indent=2))
