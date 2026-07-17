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
def evaluate_ranking(
    dataset: BranchDataset,
    trainer: OfflineQTrainer,
    q: TwinSkateQ,
    aggregation: str = "min",
    beta: float = 0.5,
    batch_size: int = 4096,
) -> dict:
    by_mode: dict[int, list[dict[str, float]]] = {mode: [] for mode in range(5)}
    all_rows = []
    groups, _ = dataset.grouped_indices()
    ordered_indices = groups.reshape(-1)
    prediction_chunks, disagreement_chunks = [], []
    for start in range(0, len(ordered_indices), batch_size):
        batch = dataset.batch(ordered_indices[start : start + batch_size])
        q1, q2 = q(trainer.q_input(batch))
        prediction_chunks.append(aggregate(q1, q2, aggregation, beta).reshape(-1).cpu())
        disagreement_chunks.append((q1 - q2).abs().reshape(-1).cpu())
    shape = groups.shape
    predictions = torch.cat(prediction_chunks).reshape(shape)
    disagreements = torch.cat(disagreement_chunks).reshape(shape)
    target_values = dataset.tensors["finite_horizon_return"][ordered_indices].reshape(shape).cpu()
    modes = dataset.tensors["mode_id"][groups[:, 0]].reshape(-1).cpu()
    quality_values = {
        name: dataset.tensors[name][ordered_indices].reshape(shape).cpu() * direction
        for name, direction in QUALITY_FIELDS.items()
    }
    for group_index in range(len(groups)):
        prediction = predictions[group_index]
        target = target_values[group_index]
        failure = torch.logical_or(
            quality_values["fall"][group_index] < 0,
            torch.logical_or(
                quality_values["contact_loss"][group_index] < 0,
                quality_values["illegal_contact"][group_index] < 0,
            ),
        )
        row = ranking_metrics(prediction, target, failure)
        row["q_disagreement"] = float(disagreements[group_index].mean())
        for name in QUALITY_FIELDS:
            quality = quality_values[name][group_index]
            if torch.unique(quality).numel() > 1:
                if torch.unique(target).numel() > 1:
                    row[f"return_vs_{name}"] = spearman(target, quality)
                if torch.unique(prediction).numel() > 1:
                    row[f"q_vs_{name}"] = spearman(prediction, quality)
        if failure.any() and (~failure).any():
            row["return_failure_gap"] = float(target[~failure].mean() - target[failure].mean())
            row["q_failure_gap"] = float(prediction[~failure].mean() - prediction[failure].mean())
        mode = int(modes[group_index].item())
        by_mode[mode].append(row)
        all_rows.append(row)
    return {"overall": mean_metrics(all_rows), "by_mode": {str(mode): mean_metrics(rows) for mode, rows in by_mode.items()}}
