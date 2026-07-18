from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import torch

from ..schemas import FEATURE_SCHEMA_VERSION
from .batch import TensorBatch
from .storage import atomic_torch_save

ANCHOR_SPLIT_VERSION = "anchor-group-ranked-v2"
BRANCH_ACTION_SEMANTICS = "single_macro_then_phase_baseline_v2"
PHASE_SAMPLING_VERSION = "stratified-v1"


class AnchorBatchSampler:
    def __init__(
        self,
        dataset: "BranchDataset",
        row_indices: torch.Tensor,
        sampling: str,
        seed: int,
    ) -> None:
        groups, anchor_ids = dataset.grouped_indices()
        selected_anchor_ids = torch.unique(dataset.tensors["anchor_id"][row_indices].reshape(-1))
        selected = torch.isin(anchor_ids, selected_anchor_ids)
        self.groups = groups[selected]
        first_rows = self.groups[:, 0]
        self.modes = dataset.tensors["mode_id"][first_rows].reshape(-1).long()
        row_modes = dataset.tensors["mode_id"][self.groups].reshape(self.groups.shape).long()
        contact_failure = torch.logical_and(
            dataset.tensors["contact_loss"][self.groups].reshape(self.groups.shape).bool(),
            row_modes != 3,
        )
        failures = torch.logical_or(
            dataset.tensors["fall"][self.groups].reshape(self.groups.shape).bool(),
            torch.logical_or(
                contact_failure,
                dataset.tensors["illegal_contact"][self.groups].reshape(self.groups.shape) > 0,
            ),
        )
        self.mixed_outcome = torch.logical_and(failures.any(dim=1), (~failures).any(dim=1))
        self.sampling = sampling
        self.generator = torch.Generator(device=self.groups.device).manual_seed(seed)
        self.mode_values = torch.unique(self.modes)
        self.mode_pools = {int(mode): (self.modes == mode).nonzero().reshape(-1) for mode in self.mode_values}
        self.mode_mixed_pools = {
            int(mode): torch.logical_and(self.modes == mode, self.mixed_outcome).nonzero().reshape(-1)
            for mode in self.mode_values
        }
        self.all_indices = torch.arange(len(self.groups), device=self.groups.device)
        self.mixed_indices = self.mixed_outcome.nonzero().reshape(-1)

    def _draw(self, pool: torch.Tensor, count: int) -> torch.Tensor:
        if not count:
            return pool[:0]
        return pool[torch.randint(len(pool), (count,), generator=self.generator, device=pool.device)]

    def sample(self, num_anchors: int) -> torch.Tensor:
        if self.sampling == "uniform":
            selected = self._draw(self.all_indices, num_anchors)
        elif self.sampling == "outcome_balanced":
            mixed_count = num_anchors // 2 if len(self.mixed_indices) else 0
            selected = torch.cat((
                self._draw(self.mixed_indices, mixed_count),
                self._draw(self.all_indices, num_anchors - mixed_count),
            ))
        else:
            chosen_modes = self.mode_values[
                torch.randint(
                    len(self.mode_values), (num_anchors,), generator=self.generator,
                    device=self.mode_values.device,
                )
            ]
            parts = []
            balance_outcomes = self.sampling == "mode_outcome_balanced"
            for mode in self.mode_values:
                count = int((chosen_modes == mode).sum())
                mixed_pool = self.mode_mixed_pools[int(mode)]
                mixed_count = count // 2 if balance_outcomes and len(mixed_pool) else 0
                parts.extend((
                    self._draw(mixed_pool, mixed_count),
                    self._draw(self.mode_pools[int(mode)], count - mixed_count),
                ))
            selected = torch.cat(parts)
        permutation = torch.randperm(len(selected), generator=self.generator, device=selected.device)
        return self.groups[selected[permutation]].reshape(-1)


class BranchDataset:
    def __init__(self, tensors: dict[str, torch.Tensor], metadata: dict | None = None) -> None:
        lengths = {value.shape[0] for value in tensors.values()}
        if len(lengths) != 1:
            raise ValueError(f"Branch tensor lengths differ: {lengths}")
        self.tensors = tensors
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return next(iter(self.tensors.values())).shape[0]

    def batch(self, indices: torch.Tensor) -> TensorBatch:
        return TensorBatch({name: value[indices] for name, value in self.tensors.items()})

    def grouped_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        anchors = self.tensors["anchor_id"].reshape(-1).long()
        candidates = self.tensors["candidate_id"].reshape(-1).long()
        candidate_span = int(candidates.max().item()) + 1
        order = torch.argsort(anchors * candidate_span + candidates)
        sorted_anchors = anchors[order]
        unique_anchors, counts = torch.unique_consecutive(sorted_anchors, return_counts=True)
        if not torch.all(counts == counts[0]):
            raise ValueError("Branch anchors have different candidate counts")
        candidates_per_anchor = int(counts[0].item())
        expected = self.metadata.get("candidates_per_anchor")
        if expected is not None and candidates_per_anchor != expected:
            raise ValueError(
                f"Branch candidate count mismatch: metadata={expected}, data={candidates_per_anchor}"
            )
        return order.reshape(len(unique_anchors), candidates_per_anchor), unique_anchors

    def anchor_split(self, validation_fraction: float, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_ids = self.tensors["anchor_id"].reshape(-1).cpu()
        anchors = torch.unique(anchor_ids)
        generator = torch.Generator().manual_seed(seed)
        anchors = anchors[torch.randperm(len(anchors), generator=generator)]
        validation_count = max(1, round(len(anchors) * validation_fraction))
        validation_anchors = anchors[:validation_count]
        validation_mask = torch.isin(anchor_ids, validation_anchors)
        return torch.where(~validation_mask)[0], torch.where(validation_mask)[0]

    def anchor_mode_counts(self) -> dict[int, int]:
        groups, _ = self.grouped_indices()
        modes = self.tensors["mode_id"][groups[:, 0]].reshape(-1).long()
        return {int(mode): int((modes == mode).sum()) for mode in torch.unique(modes)}

    def validate_basis(self, basis_sha256: str) -> None:
        actual = self.metadata.get("basis_sha256")
        if actual != basis_sha256:
            raise ValueError(f"Branch dataset basis mismatch: expected {basis_sha256}, got {actual}")

    def validate_formal_semantics(self) -> None:
        expected = {
            "branch_action_semantics": BRANCH_ACTION_SEMANTICS,
            "phase_sampling": PHASE_SAMPLING_VERSION,
        }
        mismatched = {
            name: (value, self.metadata.get(name))
            for name, value in expected.items()
            if self.metadata.get(name) != value
        }
        if mismatched:
            raise ValueError(f"Branch dataset semantics mismatch: {mismatched}")

    def save(self, path: str | Path) -> None:
        atomic_torch_save({
            "tensors": {name: value.cpu() for name, value in self.tensors.items()},
            "metadata": {**self.metadata, "feature_schema_version": FEATURE_SCHEMA_VERSION},
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "BranchDataset":
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload["metadata"].get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("Branch dataset feature schema mismatch")
        return cls(payload["tensors"], payload["metadata"])

    @classmethod
    def merge(cls, paths: Sequence[str | Path]) -> "BranchDataset":
        if not paths:
            raise ValueError("No branch shards provided")
        shards = [cls.load(path) for path in paths]
        fields = set(shards[0].tensors)
        if any(set(shard.tensors) != fields for shard in shards[1:]):
            raise ValueError("Branch shard fields differ")
        basis_paths = {shard.metadata.get("basis_path") for shard in shards}
        if len(basis_paths) != 1:
            raise ValueError(f"Branch shard basis paths differ: {basis_paths}")
        basis_hashes = {shard.metadata.get("basis_sha256") for shard in shards}
        if None in basis_hashes or len(basis_hashes) != 1:
            raise ValueError(f"Branch shard basis checksums differ or are missing: {basis_hashes}")
        candidates_per_anchor = {shard.metadata.get("candidates_per_anchor") for shard in shards}
        if len(candidates_per_anchor) != 1:
            raise ValueError(f"Branch shard candidate counts differ: {candidates_per_anchor}")
        horizons = {
            tuple(value) if isinstance(value := shard.metadata.get("horizon_low_steps"), list) else value
            for shard in shards
        }
        if len(horizons) != 1:
            raise ValueError(f"Branch shard horizons differ: {horizons}")
        action_semantics = {shard.metadata.get("branch_action_semantics") for shard in shards}
        if None in action_semantics or len(action_semantics) != 1:
            raise ValueError(f"Branch shard action semantics differ or are missing: {action_semantics}")
        hold_steps = {shard.metadata.get("candidate_hold_low_steps") for shard in shards}
        if None in hold_steps or len(hold_steps) != 1:
            raise ValueError(f"Branch shard candidate hold steps differ or are missing: {hold_steps}")
        phase_sampling = {shard.metadata.get("phase_sampling") for shard in shards}
        if None in phase_sampling or len(phase_sampling) != 1:
            raise ValueError(f"Branch shard phase sampling differs or is missing: {phase_sampling}")
        tensors = {name: torch.cat([shard.tensors[name] for shard in shards], dim=0) for name in fields}
        anchors = tensors["anchor_id"].reshape(-1)
        candidates = tensors["candidate_id"].reshape(-1)
        pairs = torch.stack((anchors, candidates), dim=-1)
        if len(torch.unique(pairs, dim=0)) != len(pairs):
            raise ValueError("Branch shards contain duplicate anchor/candidate pairs")
        horizon_value = horizons.pop()
        metadata = {
            "num_anchors": int(torch.unique(anchors).numel()),
            "candidates_per_anchor": candidates_per_anchor.pop(),
            "horizon_low_steps": list(horizon_value) if isinstance(horizon_value, tuple) else horizon_value,
            "candidate_hold_low_steps": hold_steps.pop(),
            "branch_action_semantics": action_semantics.pop(),
            "phase_sampling": phase_sampling.pop(),
            "phase_anchor_counts": {
                name: sum(int(shard.metadata.get("phase_anchor_counts", {}).get(name, 0)) for shard in shards)
                for name in sorted({
                    name
                    for shard in shards
                    for name in shard.metadata.get("phase_anchor_counts", {})
                })
            },
            "basis_path": basis_paths.pop(),
            "basis_sha256": basis_hashes.pop(),
            "merged_shards": len(shards),
            "source_paths": [str(Path(path)) for path in paths],
        }
        return cls(tensors, metadata)
