from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import torch

from ..schemas import FEATURE_SCHEMA_VERSION
from .git_info import git_commit


def dated_checkpoint_dir(root: str | Path, experiment: str) -> Path:
    run_date = os.environ.get("SKATE_BFM_RUN_DATE") or datetime.now().strftime("%Y-%m-%d")
    directory = Path(root) / run_date / experiment
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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
