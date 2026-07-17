from __future__ import annotations

import torch

from ..algorithms.offline_q_trainer import OfflineQTrainer
from ..data.branch_dataset import BranchDataset
from ..models.skate_q import TwinSkateQ
from ..q.aggregators import aggregate
from .metrics import mean_metrics, ranking_metrics


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
        row = ranking_metrics(prediction, target, batch["fall"].reshape(-1).bool())
        row["q_disagreement"] = float((q1 - q2).abs().mean())
        mode = int(batch["mode_id"][0].item())
        by_mode[mode].append(row)
        all_rows.append(row)
    return {"overall": mean_metrics(all_rows), "by_mode": {str(mode): mean_metrics(rows) for mode, rows in by_mode.items()}}
