from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

from skate_bfm_flow.algorithms.collector import BranchCollector
from skate_bfm_flow.bfm.latent_basis import build_mode_basis, configured_mode_files, save_basis
from skate_bfm_flow.config import load_config
from skate_bfm_flow.data.branch_dataset import BranchDataset
from skate_bfm_flow.env.macro_env import LatentFlowMacroEnv
from skate_bfm_flow.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--merge-glob", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.merge_glob:
        paths = sorted(glob(args.merge_glob))
        dataset = BranchDataset.merge(paths)
        output = args.output or cfg.train.dataset_path
        dataset.save(output)
        print(f"merged {len(paths)} shards, {len(dataset)} candidates -> {output}")
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
