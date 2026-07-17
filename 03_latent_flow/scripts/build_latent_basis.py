from __future__ import annotations

import argparse

from skate_bfm_flow.bfm.latent_basis import build_mode_basis, save_basis
from skate_bfm_flow.config import load_config
from skate_bfm_flow.enums import MODE_NAMES
from skate_bfm_flow.utils.git_info import git_commit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    mode_files = {mode: [cfg.latent.prototype_paths[mode]] for mode in MODE_NAMES}
    basis, metadata = build_mode_basis(mode_files, cfg.latent.flow_dim, cfg.experiment.seed)
    metadata.update({"git_commit": git_commit(), "z_dim": cfg.latent.z_dim, "source": "configured prototypes"})
    save_basis(basis, metadata, args.output)
    print(f"saved basis {tuple(basis.shape)} to {args.output}")


if __name__ == "__main__":
    main()
