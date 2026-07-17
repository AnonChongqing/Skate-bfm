from __future__ import annotations

import os
from pathlib import Path

import torch


def atomic_torch_save(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_tensor_payload(path: str | Path, device: str = "cpu"):
    return torch.load(path, map_location=device, weights_only=False)
