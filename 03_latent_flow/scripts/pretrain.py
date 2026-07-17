from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from skate_bfm_flow.config import Stage03Config, load_config


def validate_configs(q_cfg: Stage03Config, bc_cfg: Stage03Config) -> None:
    matching = {
        "experiment.name": (q_cfg.experiment.name, bc_cfg.experiment.name),
        "train.dataset_path": (q_cfg.train.dataset_path, bc_cfg.train.dataset_path),
        "paths.basis_path": (q_cfg.paths.basis_path, bc_cfg.paths.basis_path),
        "latent.flow_dim": (q_cfg.latent.flow_dim, bc_cfg.latent.flow_dim),
    }
    mismatched = [name for name, values in matching.items() if values[0] != values[1]]
    if mismatched:
        raise ValueError(f"Q/BC configs disagree on: {', '.join(mismatched)}")
    if q_cfg.q.target.type != "finite_horizon_return":
        raise ValueError("Pretraining Q config must use q.target.type=finite_horizon_return")


def run_stage(label: str, command: list[str], project_root: Path) -> None:
    print("=" * 96, flush=True)
    print(f"[PRETRAIN] START {label}", flush=True)
    print(f"[COMMAND] {shlex.join(command)}", flush=True)
    print("=" * 96, flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=project_root, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        print(f"[PRETRAIN] FAILED {label} after {elapsed / 60:.1f} min", flush=True)
        raise SystemExit(completed.returncode)
    print(f"[PRETRAIN] DONE {label} in {elapsed / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Offline-Q and Flow-BC warm starts sequentially")
    parser.add_argument("--q-config", required=True)
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    q_config = Path(args.q_config).expanduser().resolve()
    bc_config = Path(args.bc_config).expanduser().resolve()
    q_cfg = load_config(q_config)
    bc_cfg = load_config(bc_config)
    validate_configs(q_cfg, bc_cfg)

    run_date = os.environ.get("SKATE_BFM_RUN_DATE") or date.today().isoformat()
    checkpoint_dir = Path(q_cfg.paths.checkpoint_dir) / run_date / q_cfg.experiment.name
    dataset_path = Path(q_cfg.train.dataset_path)
    project_root = Path(q_cfg.paths.project_root)
    scripts = Path(__file__).resolve().parent
    commands = (
        ("Offline Twin-Q", [sys.executable, str(scripts / "train_offline_q.py"), "--config", str(q_config)]),
        ("Flow Behavior Cloning", [sys.executable, str(scripts / "train_flow_bc.py"), "--config", str(bc_config)]),
    )

    print(f"[PRETRAIN] dataset: {dataset_path}", flush=True)
    print(f"[PRETRAIN] checkpoints: {checkpoint_dir}", flush=True)
    print(f"[PRETRAIN] Q steps: {q_cfg.train.steps:,}; BC steps: {bc_cfg.train.steps:,}", flush=True)
    if args.dry_run:
        for label, command in commands:
            print(f"[DRY RUN] {label}: {shlex.join(command)}", flush=True)
        return
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Merged branch dataset is missing: {dataset_path}")

    for label, command in commands:
        run_stage(label, command, project_root)

    q_checkpoint = checkpoint_dir / "offline_q.pt"
    bc_checkpoint = checkpoint_dir / "flow_bc.pt"
    if not q_checkpoint.is_file() or not bc_checkpoint.is_file():
        raise FileNotFoundError(
            f"Pretraining completed without expected checkpoints: {q_checkpoint}, {bc_checkpoint}"
        )
    print(f"[PRETRAIN] COMPLETE: {q_checkpoint} + {bc_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
