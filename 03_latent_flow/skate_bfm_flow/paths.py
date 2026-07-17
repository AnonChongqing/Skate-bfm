from __future__ import annotations

from pathlib import Path


STAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE_ROOT.parent
DEFAULT_DATA_ROOT = Path("/63data1/hwh_data/Skate-bfm")


def require_data_path(path: str | Path, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Large artifact path must be under {root}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
