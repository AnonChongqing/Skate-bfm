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


def validate_configs(
    q_cfg: Stage03Config,
    bc_cfg: Stage03Config,
    sac_cfg: Stage03Config,
) -> None:
    matching = {
        "experiment.name": (
            q_cfg.experiment.name,
            bc_cfg.experiment.name,
            sac_cfg.experiment.name,
        ),
        "train.dataset_path": (
            q_cfg.train.dataset_path,
            bc_cfg.train.dataset_path,
            sac_cfg.train.dataset_path,
        ),
        "paths.basis_path": (
            q_cfg.paths.basis_path,
            bc_cfg.paths.basis_path,
            sac_cfg.paths.basis_path,
        ),
        "latent.flow_dim": (
            q_cfg.latent.flow_dim,
            bc_cfg.latent.flow_dim,
            sac_cfg.latent.flow_dim,
        ),
    }
    mismatched = [name for name, values in matching.items() if len(set(values)) != 1]
    if mismatched:
        raise ValueError(f"Q/BC/SAC configs disagree on: {', '.join(mismatched)}")
    if q_cfg.q.target.type != "finite_horizon_return":
        raise ValueError("Offline-Q config must use q.target.type=finite_horizon_return")
    if sac_cfg.q.target.type != "sac_td":
        raise ValueError("SAC config must use q.target.type=sac_td")


def run_stage(index: int, label: str, command: list[str], project_root: Path) -> None:
    print("=" * 96, flush=True)
    print(f"[PIPELINE {index}/3] START {label}", flush=True)
    print(f"[COMMAND] {shlex.join(command)}", flush=True)
    print("=" * 96, flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=project_root, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        print(f"[PIPELINE {index}/3] FAILED {label} after {elapsed / 60:.1f} min", flush=True)
        raise SystemExit(completed.returncode)
    print(f"[PIPELINE {index}/3] DONE {label} in {elapsed / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Offline-Q, Flow-BC, and online SAC sequentially")
    parser.add_argument("--q-config", required=True)
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--sac-config", required=True)
    parser.add_argument("--sac-set", action="append", default=[], dest="sac_overrides")
    parser.add_argument("--start-stage", choices=("q", "bc", "sac"), default="q")
    parser.add_argument("--sac-resume", default=None)
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    q_config = Path(args.q_config).expanduser().resolve()
    bc_config = Path(args.bc_config).expanduser().resolve()
    sac_config = Path(args.sac_config).expanduser().resolve()
    q_cfg = load_config(q_config)
    bc_cfg = load_config(bc_config)
    sac_cfg = load_config(sac_config, args.sac_overrides)
    validate_configs(q_cfg, bc_cfg, sac_cfg)

    run_date = os.environ.get("SKATE_BFM_RUN_DATE") or date.today().isoformat()
    checkpoint_dir = Path(q_cfg.paths.checkpoint_dir) / run_date / q_cfg.experiment.name
    dataset_path = Path(q_cfg.train.dataset_path)
    project_root = Path(q_cfg.paths.project_root)
    scripts = Path(__file__).resolve().parent
    q_checkpoint = checkpoint_dir / "offline_q.pt"
    bc_checkpoint = checkpoint_dir / "flow_bc.pt"
    sac_checkpoint = checkpoint_dir / "sac_final.pt"
    sac_command = [
        sys.executable,
        str(scripts / "train_online_sac.py"),
        "--config",
        str(sac_config),
        "--policy-checkpoint",
        str(bc_checkpoint),
        "--q-checkpoint",
        str(q_checkpoint),
    ]
    for override in args.sac_overrides:
        sac_command.extend(("--set", override))
    if args.sac_resume:
        sac_command.extend(("--resume", str(Path(args.sac_resume).expanduser().resolve())))
    if args.weights_only:
        sac_command.append("--weights-only")
    stages = (
        (
            1,
            "Offline Twin-Q",
            [sys.executable, str(scripts / "train_offline_q.py"), "--config", str(q_config)],
        ),
        (
            2,
            "Flow Behavior Cloning",
            [sys.executable, str(scripts / "train_flow_bc.py"), "--config", str(bc_config)],
        ),
        (3, "Online Latent SAC", sac_command),
    )
    start_index = {"q": 1, "bc": 2, "sac": 3}[args.start_stage]

    print(f"[PIPELINE] dataset: {dataset_path}", flush=True)
    print(f"[PIPELINE] checkpoints: {checkpoint_dir}", flush=True)
    print(
        f"[PIPELINE] Q steps: {q_cfg.train.steps:,}; BC steps: {bc_cfg.train.steps:,}; "
        f"SAC transitions: {sac_cfg.train.steps:,}; start: {args.start_stage}",
        flush=True,
    )
    if args.dry_run:
        for index, label, command in stages:
            if index >= start_index:
                print(f"[DRY RUN {index}/3] {label}: {shlex.join(command)}", flush=True)
        return
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Merged branch dataset is missing: {dataset_path}")

    required = []
    if start_index > 1:
        required.append(q_checkpoint)
    if start_index > 2:
        required.append(bc_checkpoint)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required warm-start checkpoints are missing: {missing}")

    for index, label, command in stages:
        if index < start_index:
            continue
        run_stage(index, label, command, project_root)
        expected = {1: q_checkpoint, 2: bc_checkpoint, 3: sac_checkpoint}[index]
        if not expected.is_file():
            raise FileNotFoundError(f"{label} completed without expected checkpoint: {expected}")

    print(f"[PIPELINE] COMPLETE: {sac_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
