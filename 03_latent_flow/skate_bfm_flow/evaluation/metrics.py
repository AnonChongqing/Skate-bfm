from __future__ import annotations

import torch


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(len(values), device=values.device, dtype=torch.float32)
    return ranks


def spearman(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if len(prediction) < 2:
        return 0.0
    first, second = _ranks(prediction), _ranks(target)
    first, second = first - first.mean(), second - second.mean()
    denominator = first.norm() * second.norm()
    return float((first * second).sum() / denominator) if denominator > 0 else 0.0


def kendall(prediction: torch.Tensor, target: torch.Tensor) -> float:
    concordance = 0.0
    pairs = 0
    for i in range(len(prediction)):
        for j in range(i + 1, len(prediction)):
            concordance += float(torch.sign((prediction[i] - prediction[j]) * (target[i] - target[j])))
            pairs += 1
    return concordance / pairs if pairs else 0.0


def ndcg(prediction: torch.Tensor, target: torch.Tensor) -> float:
    relevance = target - target.min()
    predicted_order = torch.argsort(prediction, descending=True)
    ideal_order = torch.argsort(target, descending=True)
    discounts = torch.log2(torch.arange(len(target), device=target.device, dtype=torch.float32) + 2.0)
    dcg = (relevance[predicted_order] / discounts).sum()
    ideal = (relevance[ideal_order] / discounts).sum()
    return float(dcg / ideal) if ideal > 0 else 1.0


def ranking_metrics(prediction: torch.Tensor, target: torch.Tensor, failure: torch.Tensor | None = None) -> dict[str, float]:
    predicted_order = torch.argsort(prediction, descending=True)
    true_order = torch.argsort(target, descending=True)
    top1_regret = target.max() - target[predicted_order[0]]
    top3 = set(predicted_order[: min(3, len(prediction))].tolist())
    metrics = {
        "spearman": spearman(prediction, target), "kendall": kendall(prediction, target),
        "top1_regret": float(top1_regret), "top3_hit": float(int(true_order[0]) in top3),
        "ndcg": ndcg(prediction, target),
    }
    if failure is not None and failure.any() and (~failure.bool()).any():
        metrics["failure_last"] = float(prediction[failure.bool()].max() < prediction[~failure.bool()].min())
    return metrics


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set.intersection(*(set(row) for row in rows))
    return {key: sum(row[key] for row in rows) / len(rows) for key in sorted(keys)}
