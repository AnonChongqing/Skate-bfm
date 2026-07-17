from __future__ import annotations

from pathlib import Path

import torch

from ..schemas import FEATURE_SCHEMA_VERSION
from .git_info import git_commit


def make_checkpoint(**objects) -> dict:
    return {**objects, "feature_schema_version": FEATURE_SCHEMA_VERSION, "git_commit": git_commit()}


def save_checkpoint(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def validate_checkpoint(payload: dict, expected: dict) -> None:
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema mismatch")
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"checkpoint mismatch for {name}: expected {value}, got {payload.get(name)}")
