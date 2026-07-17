from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MOTION_DIM = 36
ROOT_POS = slice(0, 3)
ROOT_QUAT_WXYZ = slice(3, 7)
ROOT_LINEAR_VEL = slice(7, 10)
ROOT_ANGULAR_VEL = slice(10, 13)
JOINT_POS = slice(13, 36)


@dataclass(frozen=True)
class HuskyMotion:
    name: str
    frames: np.ndarray

    @property
    def joint_pos(self) -> np.ndarray:
        return self.frames[:, JOINT_POS]

    @property
    def phase(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, len(self.frames), dtype=np.float32)


def load_motion(path: str | Path) -> HuskyMotion:
    path = Path(path)
    frames = np.load(path, allow_pickle=False)
    if frames.ndim != 2 or frames.shape[1] != MOTION_DIM:
        raise ValueError(f"Expected HUSKY motion [T,{MOTION_DIM}], got {frames.shape} from {path}")
    if len(frames) < 2 or not np.isfinite(frames).all():
        raise ValueError(f"HUSKY motion must contain at least two finite frames: {path}")
    quaternion_norm = np.linalg.norm(frames[:, ROOT_QUAT_WXYZ], axis=1)
    if not np.allclose(quaternion_norm, 1.0, atol=5e-3):
        raise ValueError(f"Invalid wxyz root quaternion in {path}")
    return HuskyMotion(path.name, frames.astype(np.float32, copy=False))


def load_motion_directory(directory: str | Path) -> list[HuskyMotion]:
    paths = sorted(Path(directory).glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No HUSKY motion files found under {directory}")
    return [load_motion(path) for path in paths]
