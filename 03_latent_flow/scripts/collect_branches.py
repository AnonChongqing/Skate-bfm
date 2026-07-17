from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

from skate_bfm_flow.algorithms.collector import BranchCollector
from skate_bfm_flow.bfm.latent_basis import build_mode_basis, configured_mode_files, save_basis
from skate_bfm_flow.config import load_config
from skate_bfm_flow.data.branch_dataset import BranchDataset
from skate_bfm_flow.env.macro_env import LatentFlowMacroEnv
from skate_bfm_flow.utils.seed import seed_everything


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not 1 <= len(gpus) <= 3:
        raise ValueError("--gpus requires one to three comma-separated GPU ids")
    if len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must not contain duplicates")
    return gpus


def launch_and_merge(args, cfg, parser: argparse.ArgumentParser) -> None:
    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output or cfg.train.dataset_path)
    part_dir = output.parent / f".{output.stem}.parts-{os.getpid()}"
    shard_paths = [part_dir / f"part-{index:03d}-of-{len(gpus):03d}.pt" for index in range(len(gpus))]
    commands: list[tuple[list[str], dict[str, str]]] = []
    for shard_index, (gpu, shard_path) in enumerate(zip(gpus, shard_paths, strict=True)):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--config", str(Path(args.config).resolve()),
            "--num-shards", str(len(gpus)),
            "--shard-index", str(shard_index),
            "--output", str(shard_path),
        ]
        for override in args.overrides:
            command.extend(("--set", override))
        if args.progress_interval is not None:
            command.extend(("--progress-interval", str(args.progress_interval)))
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        commands.append((command, environment))

    print(f"[BRANCH] GPUs={','.join(gpus)} shards={len(gpus)} final={output}", flush=True)
    if args.dry_run:
        for command, environment in commands:
            print(f"[DRY RUN GPU {environment['CUDA_VISIBLE_DEVICES']}] {shlex.join(command)}", flush=True)
        return

    part_dir.mkdir(parents=True, exist_ok=False)
    processes = [
        subprocess.Popen(command, cwd=cfg.paths.project_root, env=environment)
        for command, environment in commands
    ]
    return_codes = [process.wait() for process in processes]
    if any(return_codes):
        failed = [index for index, code in enumerate(return_codes) if code]
        raise RuntimeError(f"Branch shard workers failed: {failed}; temporary files kept at {part_dir}")

    print(f"[BRANCH] All workers complete; merging {len(shard_paths)} shards", flush=True)
    dataset = BranchDataset.merge(shard_paths)
    dataset.save(output)
    for shard_path in shard_paths:
        shard_path.unlink()
    part_dir.rmdir()
    print(f"[BRANCH] COMPLETE: {len(dataset)} candidates -> {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--output", default=None)
    parser.add_argument("--gpus", default=None, help="Launch 1-3 GPU workers and merge automatically")
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.gpus is not None:
        launch_and_merge(args, cfg, parser)
        return
    if not 1 <= args.num_shards <= 3 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 1 <= num_shards <= 3 and 0 <= shard_index < num_shards")
    if cfg.branch.disable_interval_push:
        cfg.env.interval_push = False
    cfg.env.quiet = True
    total_anchors = cfg.branch.num_anchors
    base_count, remainder = divmod(total_anchors, args.num_shards)
    shard_count = base_count + int(args.shard_index < remainder)
    anchor_offset = args.shard_index * base_count + min(args.shard_index, remainder)
    shard_seed = cfg.experiment.seed + args.shard_index * 100003
    if not Path(cfg.paths.basis_path).exists():
        files = configured_mode_files(cfg.latent.prototype_paths, cfg.latent.basis_source_paths)
        basis, metadata = build_mode_basis(files, cfg.latent.flow_dim, cfg.experiment.seed)
        save_basis(basis, metadata, cfg.paths.basis_path)
    cfg.experiment.seed = shard_seed
    seed_everything(shard_seed, cfg.experiment.deterministic)
    env = LatentFlowMacroEnv(cfg)
    try:
        dataset = BranchCollector(env, shard_seed).collect(
            shard_count, cfg.branch.candidates_per_anchor, cfg.branch.horizon_low_steps,
            horizon_low_steps_range=cfg.branch.horizon_low_steps_range,
            anchor_offset=anchor_offset,
            log_interval=args.progress_interval or cfg.env.num_envs,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        output = args.output or cfg.train.dataset_path
        if args.num_shards > 1 and args.output is None:
            path = Path(output)
            output = str(path.with_name(f"{path.stem}.part-{args.shard_index:03d}-of-{args.num_shards:03d}{path.suffix}"))
        dataset.save(output)
        print(f"[BRANCH] Saved {len(dataset)} branch candidates to {output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
