from __future__ import annotations

import argparse

from skate_bfm_flow.algorithms.collector import BranchCollector
from skate_bfm_flow.bfm.latent_basis import build_mode_basis, configured_mode_files, save_basis
from skate_bfm_flow.config import load_config
from skate_bfm_flow.env.macro_env import LatentFlowMacroEnv
from skate_bfm_flow.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
    if not __import__("pathlib").Path(cfg.paths.basis_path).exists():
        files = configured_mode_files(cfg.latent.prototype_paths, cfg.latent.basis_source_paths)
        basis, metadata = build_mode_basis(files, cfg.latent.flow_dim, cfg.experiment.seed)
        save_basis(basis, metadata, cfg.paths.basis_path)
    env = LatentFlowMacroEnv(cfg)
    try:
        dataset = BranchCollector(env, cfg.experiment.seed).collect(
            cfg.branch.num_anchors, cfg.branch.candidates_per_anchor, cfg.branch.horizon_low_steps
        )
        output = args.output or cfg.train.dataset_path
        dataset.save(output)
        print(f"saved {len(dataset)} branch candidates to {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
