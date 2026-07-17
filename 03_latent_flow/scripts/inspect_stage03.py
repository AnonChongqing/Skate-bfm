from __future__ import annotations

import argparse
from pathlib import Path

import torch

from skate_bfm_flow.bfm.latent_basis import build_mode_basis, save_basis
from skate_bfm_flow.config import load_config, save_resolved_config
from skate_bfm_flow.env.macro_env import LatentFlowMacroEnv
from skate_bfm_flow.enums import MODE_NAMES
from skate_bfm_flow.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
    basis_path = Path(cfg.paths.basis_path)
    if not basis_path.exists():
        files = {mode: [cfg.latent.prototype_paths[mode]] for mode in MODE_NAMES}
        basis, metadata = build_mode_basis(files, cfg.latent.flow_dim, cfg.experiment.seed)
        save_basis(basis, metadata, basis_path)
    env = LatentFlowMacroEnv(cfg)
    try:
        actor_obs = env.reset()
        print("feature_schema", env.features.schema_version)
        print("actor_obs", tuple(actor_obs.shape), "actor_frame", tuple(env.latest_features.actor_frame.shape))
        for name in ("critic_robot", "critic_board", "critic_contact", "critic_goal_mode"):
            value = getattr(env.latest_features, name)
            print(name, tuple(value.shape), "finite", bool(torch.isfinite(value).all()))
        print("bfm_obs", {name: tuple(value.shape) for name, value in env.low_env.observation.items()})
        print("latent", tuple(env.z_current.shape), "action_preview_input", 29, "adapter_output", 23)
        result = env.step(torch.zeros(1, cfg.latent.flow_dim, device=cfg.experiment.device))
        print("macro_step", tuple(result.reward_macro.shape), "terminated", result.terminated.tolist(), "truncated", result.truncated.tolist())
        print("PASS")
    finally:
        env.close()


if __name__ == "__main__":
    main()
