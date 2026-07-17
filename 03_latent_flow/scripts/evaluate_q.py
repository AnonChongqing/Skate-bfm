from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from skate_bfm_flow.algorithms.offline_q_trainer import OfflineQTrainer
from skate_bfm_flow.bfm.action_preview import FrozenBfmActionPreview
from skate_bfm_flow.config import load_config
from skate_bfm_flow.data.branch_dataset import BranchDataset
from skate_bfm_flow.env.macro_env import LatentFlowMacroEnv
from skate_bfm_flow.evaluation.q_ranking import evaluate_ranking
from skate_bfm_flow.models.skate_q import TwinSkateQ
from skate_bfm_flow.q.input_builder import QInputBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    dataset = BranchDataset.load(args.dataset or cfg.train.dataset_path, cfg.experiment.device)
    payload = torch.load(args.checkpoint, map_location=cfg.experiment.device, weights_only=False)
    env = LatentFlowMacroEnv(cfg)
    try:
        builder = QInputBuilder(cfg.q.input_profile, cfg.q.state_profile)
        q = TwinSkateQ(payload["branch_dims"], cfg.q.activation, cfg.q.final_hidden_dims).to(cfg.experiment.device)
        q.load_state_dict(payload["q"])
        preview = FrozenBfmActionPreview(env.bfm, env.adapter, cfg.q.preview.type)
        trainer = OfflineQTrainer(q, env.mapper, preview, builder, torch.optim.Adam(q.parameters(), lr=1e-4))
        summary = evaluate_ranking(dataset, trainer, q, cfg.q.target.aggregation, cfg.q.target.uncertainty_beta)
        output = Path(args.output or Path(cfg.paths.run_dir) / cfg.experiment.name / "q_ranking.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
