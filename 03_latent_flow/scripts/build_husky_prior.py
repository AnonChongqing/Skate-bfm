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

    print(f"[PRIOR] Loading HUSKY motions from {cfg.husky_prior.motion_dir}", flush=True)
    motions = load_motion_directory(cfg.husky_prior.motion_dir)
    print(
        f"[PRIOR] Found {len(motions)} motions; device={cfg.experiment.device}; "
        f"batch_size={cfg.husky_prior.batch_size}",
        flush=True,
    )
    low_env = HuskyEnv(HuskyEnvConfig(
        task_id=cfg.env.task_id,
        device=cfg.experiment.device,
        domain_randomization=False,
        preserve_terminal_state=True,
    ))
    bfm = FrozenBfmPolicy(cfg.paths.bfm_model_dir, cfg.experiment.device, mean=True)
    latent_parts: list[np.ndarray] = []
    sources: list[dict[str, object]] = []
    transition_outputs: dict[str, tuple[Path, np.ndarray]] = {}
    try:
        stride = cfg.husky_prior.frame_stride
        batch_size = cfg.husky_prior.batch_size
        motion_root = Path(cfg.husky_prior.motion_dir)
        for motion_index, motion in enumerate(motions, start=1):
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
            print(
                f"[PRIOR] motion {motion_index}/{len(motions)}: {motion.name} "
                f"frames={len(motion.frames)} encoded={len(latents)}",
                flush=True,
            )

        frames = cfg.husky_prior.transition_frames
        alpha = torch.linspace(0.0, 1.0, frames, device=cfg.experiment.device)
        alpha = (alpha.square() * (3.0 - 2.0 * alpha)).unsqueeze(-1)
        default = low_env.husky_env.robot.data.default_joint_pos[0]
        steer = low_env.husky_env.steer_init_pos[0]
        for name, start, end, path in (
            ("mount", default, steer, Path(cfg.husky_prior.mount_output_path)),
            ("dismount", steer, default, Path(cfg.husky_prior.dismount_output_path)),
        ):
            trajectory = (1.0 - alpha) * start + alpha * end
            latents = bfm.tracking(low_env.tracking_observation(trajectory)).float().cpu().numpy()
            transition_outputs[name] = (path, latents)
    finally:
        low_env.close()

    prior = np.concatenate(latent_parts, axis=0).astype(np.float32, copy=False)
    np.save(output, prior)
    metadata = {
        "schema": "husky-push-bfm-latent-v2",
        "motion_schema": "root_xyz[3], root_quat_wxyz[4], mujoco_joint_position[29]",
        "joint_selection": "29DoF to HUSKY 23DoF: source indices [0:19,22:26]",
        "control_hz": 50,
        "frame_stride": cfg.husky_prior.frame_stride,
        "latent_shape": list(prior.shape),
        "source_role": "push latent basis prior only; never low-level policy actions",
        "sources": sources,
        "git_commit": git_commit(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"[PRIOR] Saved HUSKY-derived frozen-BFM prior {prior.shape} to {output}", flush=True)
    for name, (path, latents) in transition_outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, latents.astype(np.float32, copy=False))
        path.with_suffix(".json").write_text(json.dumps({
            "schema": "husky-transition-bfm-latent-v1",
            "mode": name,
            "latent_shape": list(latents.shape),
            "source_role": "BFM tracking latent basis only; never direct joint actions",
            "git_commit": git_commit(),
        }, indent=2))
        print(f"[PRIOR] Saved {name} tracking prior {latents.shape} to {path}", flush=True)


if __name__ == "__main__":
    main()
