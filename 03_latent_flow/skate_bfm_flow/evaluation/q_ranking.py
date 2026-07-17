from __future__ import annotations

import torch

from ..algorithms.offline_q_trainer import OfflineQTrainer
from ..data.branch_dataset import BranchDataset
from ..models.skate_q import TwinSkateQ
from ..q.aggregators import aggregate
from .metrics import mean_metrics, ranking_metrics, spearman

QUALITY_FIELDS = {
    "board_progress": 1.0,
    "heading_progress": 1.0,
    "retention": 1.0,
    "contact_loss": -1.0,
    "illegal_contact": -1.0,
    "fall": -1.0,
}


@torch.no_grad()
def evaluate_ranking(dataset: BranchDataset, trainer: OfflineQTrainer, q: TwinSkateQ, aggregation: str = "min", beta: float = 0.5) -> dict:
    by_mode: dict[int, list[dict[str, float]]] = {mode: [] for mode in range(5)}
    all_rows = []
    for anchor in torch.unique(dataset.tensors["anchor_id"]):
        indices = (dataset.tensors["anchor_id"] == anchor).reshape(-1).nonzero().reshape(-1)
        batch = dataset.batch(indices)
        q1, q2 = q(trainer.q_input(batch))
        prediction = aggregate(q1, q2, aggregation, beta).reshape(-1)
        target = batch["finite_horizon_return"].reshape(-1)
        failure = torch.logical_or(
            batch["fall"].reshape(-1).bool(),
            torch.logical_or(batch["contact_loss"].reshape(-1).bool(), batch["illegal_contact"].reshape(-1).bool()),
        )
        row = ranking_metrics(prediction, target, failure)
        row["q_disagreement"] = float((q1 - q2).abs().mean())
        for name, direction in QUALITY_FIELDS.items():
            quality = batch[name].reshape(-1) * direction
            if torch.unique(quality).numel() > 1:
                if torch.unique(target).numel() > 1:
                    row[f"return_vs_{name}"] = spearman(target, quality)
                if torch.unique(prediction).numel() > 1:
                    row[f"q_vs_{name}"] = spearman(prediction, quality)
        if failure.any() and (~failure).any():
            row["return_failure_gap"] = float(target[~failure].mean() - target[failure].mean())
            row["q_failure_gap"] = float(prediction[~failure].mean() - prediction[failure].mean())
        mode = int(batch["mode_id"][0].item())
        by_mode[mode].append(row)
        all_rows.append(row)
    return {"overall": mean_metrics(all_rows), "by_mode": {str(mode): mean_metrics(rows) for mode, rows in by_mode.items()}}
