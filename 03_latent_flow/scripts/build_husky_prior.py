from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from skate_bfm_flow.bfm.frozen_policy import FrozenBfmPolicy
from skate_bfm_flow.config import load_config
from skate_bfm_flow.data.husky_motion import load_motion_directory
from skate_bfm_flow.env.husky import HuskyEnv, HuskyEnvConfig
from skate_bfm_flow.utils.git_info import git_commit


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode HUSKY push references with frozen BFM0")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    output = Path(args.output or cfg.husky_prior.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    motions = load_motion_directory(cfg.husky_prior.motion_dir)
    low_env = HuskyEnv(HuskyEnvConfig(
        task_id=cfg.env.task_id,
        device=cfg.experiment.device,
        domain_randomization=False,
        preserve_terminal_state=True,
    ))
    bfm = FrozenBfmPolicy(cfg.paths.bfm_model_dir, cfg.experiment.device, mean=True)
    latent_parts: list[np.ndarray] = []
    sources: list[dict[str, object]] = []
    try:
        stride = cfg.husky_prior.frame_stride
        batch_size = cfg.husky_prior.batch_size
        motion_root = Path(cfg.husky_prior.motion_dir)
        for motion in motions:
            joint_pos = torch.from_numpy(motion.joint_pos[::stride]).to(cfg.experiment.device)
            chunks: list[np.ndarray] = []
            for start in range(0, len(joint_pos), batch_size):
                observations = low_env.tracking_observation(joint_pos[start:start + batch_size])
                chunks.append(bfm.tracking(observations).float().cpu().numpy())
            latents = np.concatenate(chunks, axis=0)
            latent_parts.append(latents)
            source_path = motion_root / motion.name
            sources.append({
                "file": motion.name,
                "frames": len(motion.frames),
                "encoded_frames": len(latents),
                "sha256": file_sha256(source_path),
            })
    finally:
        low_env.close()

    prior = np.concatenate(latent_parts, axis=0).astype(np.float32, copy=False)
    np.save(output, prior)
    metadata = {
        "schema": "husky-push-bfm-latent-v1",
        "motion_schema": "root_xyz[3], root_quat_wxyz[4], root_linear_velocity[3], root_angular_velocity[3], mujoco_joint_position[23]",
        "control_hz": 50,
        "frame_stride": cfg.husky_prior.frame_stride,
        "latent_shape": list(prior.shape),
        "source_role": "push latent basis prior only; never low-level policy actions",
        "sources": sources,
        "git_commit": git_commit(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"saved HUSKY-derived frozen-BFM prior {prior.shape} to {output}")


if __name__ == "__main__":
    main()
