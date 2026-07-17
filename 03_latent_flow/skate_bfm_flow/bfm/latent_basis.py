from __future__ import annotations

import hashlib
import json
from glob import glob, has_magic
from pathlib import Path

import numpy as np
import torch

from ..enums import MODE_NAMES


def load_latent(path: str | Path) -> torch.Tensor:
    value = np.load(Path(path))
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1, value.shape[-1])


def orthonormal_basis(samples: torch.Tensor, flow_dim: int, seed: int = 42) -> tuple[torch.Tensor, list[float]]:
    if samples.ndim != 2 or samples.shape[1] != 256:
        raise ValueError("latent samples must have shape [N,256]")
    centered = samples - samples.mean(0, keepdim=True)
    rank = min(centered.shape[0], centered.shape[1], flow_dim)
    if rank:
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        vectors = vh[:rank].T
        variance = singular.square()
        explained = (variance[:rank] / variance.sum().clamp_min(1e-12)).tolist()
    else:
        vectors = samples.new_empty(256, 0)
        explained = []
    if rank < flow_dim:
        generator = torch.Generator(device=samples.device).manual_seed(seed)
        extra = torch.randn(256, flow_dim - rank, generator=generator, device=samples.device)
        vectors = torch.linalg.qr(torch.cat((vectors, extra), dim=1), mode="reduced").Q[:, :flow_dim]
        explained.extend([0.0] * (flow_dim - rank))
    return vectors, explained


def build_mode_basis(mode_files: dict[str, list[str]], flow_dim: int, seed: int = 42) -> tuple[torch.Tensor, dict]:
    bases, metadata = [], {"flow_dim": flow_dim, "modes": {}}
    all_samples = [load_latent(path) for files in mode_files.values() for path in files]
    fallback = torch.cat(all_samples) if all_samples else torch.eye(256)
    for index, mode in enumerate(MODE_NAMES):
        files = mode_files.get(mode, [])
        samples = torch.cat([load_latent(path) for path in files]) if files else fallback
        basis, explained = orthonormal_basis(samples, flow_dim, seed + index)
        bases.append(basis)
        metadata["modes"][mode] = {"sources": files, "sample_count": len(samples), "explained_variance": explained}
    return torch.stack(bases), metadata


def configured_mode_files(
    prototype_paths: dict[str, str], basis_source_paths: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Resolve explicit mode samples while retaining each configured prototype."""
    files: dict[str, list[str]] = {}
    for mode in MODE_NAMES:
        sources = list(basis_source_paths.get(mode, []))
        prototype = prototype_paths[mode]
        expanded: list[str] = []
        for source in sources:
            matches = sorted(glob(source)) if has_magic(source) else [source]
            if not matches:
                raise FileNotFoundError(f"Latent basis source matched no files: {source}")
            expanded.extend(matches)
        files[mode] = list(dict.fromkeys([prototype, *expanded]))
    return files


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_basis(basis: torch.Tensor, metadata: dict, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = tensor_sha256(basis)
    torch.save({"basis": basis.cpu(), "metadata": metadata, "sha256": digest}, output)
    output.with_suffix(".json").write_text(json.dumps({**metadata, "sha256": digest}, indent=2))


def load_basis(path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    basis = payload["basis"].to(device)
    if payload.get("sha256") != tensor_sha256(basis):
        raise ValueError("Latent basis checksum mismatch")
    return basis, payload.get("metadata", {})
