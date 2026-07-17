from __future__ import annotations

import argparse

from skate_bfm_flow.bfm.latent_basis import build_mode_basis, configured_mode_files, save_basis
from skate_bfm_flow.config import load_config
from skate_bfm_flow.utils.git_info import git_commit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    mode_files = configured_mode_files(cfg.latent.prototype_paths, cfg.latent.basis_source_paths)
    print(
        f"[BASIS] Building {len(mode_files)} mode bases with flow_dim={cfg.latent.flow_dim}",
        flush=True,
    )
    for mode, paths in mode_files.items():
        print(f"[BASIS] {mode}: {len(paths)} source file(s)", flush=True)
    basis, metadata = build_mode_basis(mode_files, cfg.latent.flow_dim, cfg.experiment.seed)
    metadata.update({"git_commit": git_commit(), "z_dim": cfg.latent.z_dim, "source": "configured latent samples"})
    save_basis(basis, metadata, args.output)
    print(f"[BASIS] Saved basis {tuple(basis.shape)} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
