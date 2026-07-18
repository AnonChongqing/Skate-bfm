from __future__ import annotations

import argparse
from pathlib import Path

import torch

from skate_bfm_flow.algorithms.offline_q_trainer import OfflineQTrainer
from skate_bfm_flow.bfm.action_preview import FrozenBfmActionPreview
from skate_bfm_flow.config import load_config, save_resolved_config
from skate_bfm_flow.data.branch_dataset import ANCHOR_SPLIT_VERSION, AnchorBatchSampler, BranchDataset
from skate_bfm_flow.env.macro_env import LatentFlowMacroEnv
from skate_bfm_flow.evaluation.q_ranking import evaluate_ranking
from skate_bfm_flow.models.skate_q import TwinSkateQ
from skate_bfm_flow.enums import MODE_NAMES
from skate_bfm_flow.q.input_builder import QInputBuilder
from skate_bfm_flow.utils.checkpoint import dated_checkpoint_dir, make_checkpoint, save_checkpoint, validate_checkpoint
from skate_bfm_flow.utils.logging import RunLogger
from skate_bfm_flow.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--weights-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg.env.num_envs = 1
    cfg.env.domain_randomization = False
    cfg.env.observation_noise = False
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
    print(
        f"[Q] Loading branch dataset {cfg.train.dataset_path} on {cfg.experiment.device}",
        flush=True,
    )
    dataset = BranchDataset.load(cfg.train.dataset_path, cfg.experiment.device)
    mode_counts = dataset.anchor_mode_counts()
    missing_modes = [name for name in cfg.train.required_modes if mode_counts.get(MODE_NAMES.index(name), 0) == 0]
    if missing_modes:
        raise ValueError(f"Branch dataset is missing required modes: {missing_modes}; counts={mode_counts}")
    if cfg.train.required_modes:
        dataset.validate_formal_semantics()
    print(
        f"[Q] candidates={len(dataset):,} steps={cfg.train.steps:,} "
        f"batch={cfg.train.batch_size} log_every={cfg.train.log_interval:,}",
        flush=True,
    )
    env = LatentFlowMacroEnv(cfg)
    run_dir = Path(cfg.paths.run_dir) / cfg.experiment.name / "offline_q"
    logger = RunLogger(run_dir)
    save_resolved_config(cfg, run_dir / "resolved_config.yaml")
    try:
        sample = dataset.batch(torch.zeros(1, dtype=torch.long, device=cfg.experiment.device))
        mapper, preview = env.mapper, FrozenBfmActionPreview(env.bfm, env.adapter, cfg.q.preview.type)
        dataset.validate_basis(env.basis_metadata["sha256"])
        builder = QInputBuilder(cfg.q.input_profile, cfg.q.state_profile)
        input_example = OfflineQTrainer.__new__(OfflineQTrainer)
        input_example.mapper, input_example.preview, input_example.input_builder = mapper, preview, builder
        q_input = input_example.q_input(sample)
        q = TwinSkateQ(builder.branch_dims(q_input), cfg.q.activation, cfg.q.final_hidden_dims).to(cfg.experiment.device)
        optimizer = torch.optim.AdamW(q.parameters(), lr=cfg.q.optimizer.lr, weight_decay=cfg.q.optimizer.weight_decay)
        start_step = 0
        if args.resume:
            resumed = torch.load(args.resume, map_location=cfg.experiment.device, weights_only=False)
            validate_checkpoint(resumed, {
                "flow_dim": cfg.latent.flow_dim,
                "anchor_split_version": ANCHOR_SPLIT_VERSION,
                "basis_sha256": env.basis_metadata["sha256"],
            })
            q.load_state_dict(resumed["q"])
            if not args.weights_only and "q_optimizer" in resumed:
                optimizer.load_state_dict(resumed["q_optimizer"])
                start_step = int(resumed.get("training_step", 0))
        trainer = OfflineQTrainer(
            q, mapper, preview, builder, optimizer,
            cfg.q.loss.type, cfg.q.loss.huber_delta, cfg.q.optimizer.grad_clip,
            cfg.q.loss.ranking_weight, cfg.q.loss.ranking_margin,
            cfg.q.loss.ranking_min_return_gap, cfg.q.loss.failure_weight,
            cfg.q.loss.failure_margin,
        )
        train_indices, val_indices = dataset.anchor_split(cfg.train.validation_fraction, cfg.experiment.seed)
        train_indices, val_indices = train_indices.to(cfg.experiment.device), val_indices.to(cfg.experiment.device)
        candidates_per_anchor = int(dataset.metadata["candidates_per_anchor"])
        if cfg.train.batch_size % candidates_per_anchor:
            raise ValueError(
                f"Q batch size {cfg.train.batch_size} must be divisible by "
                f"candidates_per_anchor={candidates_per_anchor}"
            )
        sampler = AnchorBatchSampler(dataset, train_indices, cfg.q.offline_sampling, cfg.experiment.seed)
        anchors_per_batch = cfg.train.batch_size // candidates_per_anchor
        best_loss = float("inf")
        checkpoint_dir = dated_checkpoint_dir(cfg.paths.checkpoint_dir, cfg.experiment.name)
        best_loss_path = checkpoint_dir / "offline_q_best_loss.pt"
        for step in range(start_step, cfg.train.steps):
            indices = sampler.sample(anchors_per_batch)
            metrics = trainer.update(dataset.batch(indices), candidates_per_anchor)
            completed = step + 1
            if completed % cfg.train.log_interval == 0 or completed == cfg.train.steps:
                logger.report("Offline Twin-Q", completed, cfg.train.steps, {f"train/{key}": value for key, value in metrics.items()})
            total_loss = metrics["total_loss"]
            if total_loss < best_loss:
                best_loss = total_loss
                save_checkpoint(make_checkpoint(
                    q=q.state_dict(), q_optimizer=optimizer.state_dict(), branch_dims=q.q1.branch_dims,
                    config=cfg.model_dump(mode="json"), training_step=step + 1,
                    flow_dim=cfg.latent.flow_dim, q_input_profile=cfg.q.input_profile,
                    anchor_split_version=ANCHOR_SPLIT_VERSION,
                    basis_sha256=env.basis_metadata["sha256"],
                ), best_loss_path)
        validation = BranchDataset({name: value[val_indices] for name, value in dataset.tensors.items()}, dataset.metadata)
        ranking = evaluate_ranking(validation, trainer, q, cfg.q.target.aggregation, cfg.q.target.uncertainty_beta)
        (run_dir / "ranking.json").write_text(__import__("json").dumps(ranking, indent=2))
        checkpoint = args.checkpoint or str(checkpoint_dir / "offline_q.pt")
        final_payload = make_checkpoint(
            q=q.state_dict(), q_optimizer=optimizer.state_dict(), branch_dims=q.q1.branch_dims,
            config=cfg.model_dump(mode="json"), training_step=cfg.train.steps,
            flow_dim=cfg.latent.flow_dim, q_input_profile=cfg.q.input_profile,
            anchor_split_version=ANCHOR_SPLIT_VERSION,
            basis_sha256=env.basis_metadata["sha256"],
        )
        save_checkpoint(final_payload, checkpoint)
        save_checkpoint(final_payload, checkpoint_dir / "offline_q_best_ranking.pt")
        print(
            f"[Q] Saved {checkpoint}; validation candidates={len(val_indices)}; "
            f"ranking={ranking['overall']}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
