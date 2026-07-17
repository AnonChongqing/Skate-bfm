from __future__ import annotations

import argparse
from pathlib import Path

import torch

from skate_bfm_flow.algorithms.behavior_clone import bc_update, best_flow_targets
from skate_bfm_flow.config import load_config, save_resolved_config
from skate_bfm_flow.data.branch_dataset import BranchDataset
from skate_bfm_flow.models.flow_policy import FlowPolicy
from skate_bfm_flow.utils.checkpoint import dated_checkpoint_dir, make_checkpoint, save_checkpoint, validate_checkpoint
from skate_bfm_flow.utils.logging import RunLogger
from skate_bfm_flow.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--weights-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
    print(
        f"[BC] Loading branch dataset {cfg.train.dataset_path} on {cfg.experiment.device}",
        flush=True,
    )
    dataset = BranchDataset.load(cfg.train.dataset_path, cfg.experiment.device)
    observations, targets = best_flow_targets(dataset, cfg.bc.target_type, cfg.bc.temperature)
    print(
        f"[BC] anchors={len(observations):,} steps={cfg.train.steps:,} "
        f"batch={cfg.train.batch_size} log_every={cfg.train.log_interval:,}",
        flush=True,
    )
    frame_dim = observations.shape[-1] // cfg.policy.frame_stack
    policy = FlowPolicy(frame_dim, cfg.latent.flow_dim, cfg.policy.frame_stack, cfg.policy.hidden_dims, cfg.policy.activation, cfg.policy.log_std_min, cfg.policy.log_std_max).to(cfg.experiment.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.policy.optimizer_lr)
    start_step = 0
    if args.resume:
        resumed = torch.load(args.resume, map_location=cfg.experiment.device, weights_only=False)
        validate_checkpoint(resumed, {"flow_dim": cfg.latent.flow_dim})
        policy.load_state_dict(resumed["policy"])
        if not args.weights_only and "policy_optimizer" in resumed:
            optimizer.load_state_dict(resumed["policy_optimizer"])
            start_step = int(resumed.get("training_step", 0))
    run_dir = Path(cfg.paths.run_dir) / cfg.experiment.name / "bc_flow"
    logger = RunLogger(run_dir)
    save_resolved_config(cfg, run_dir / "resolved_config.yaml")
    for step in range(start_step, cfg.train.steps):
        indices = torch.randint(len(observations), (cfg.train.batch_size,), device=observations.device)
        loss = bc_update(policy, optimizer, observations[indices], targets[indices])
        completed = step + 1
        if completed % cfg.train.log_interval == 0 or completed == cfg.train.steps:
            logger.report("Flow BC", completed, cfg.train.steps, {"train/bc_loss": loss})
    checkpoint = dated_checkpoint_dir(cfg.paths.checkpoint_dir, cfg.experiment.name) / "flow_bc.pt"
    save_checkpoint(make_checkpoint(
        policy=policy.state_dict(), policy_optimizer=optimizer.state_dict(), frame_dim=frame_dim,
        flow_dim=cfg.latent.flow_dim, config=cfg.model_dump(mode="json"), training_step=cfg.train.steps,
    ), checkpoint)
    print(f"[BC] Saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
