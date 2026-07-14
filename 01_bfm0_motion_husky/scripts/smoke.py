#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = Path(__file__).resolve().parents[1]
HUSKY_SRC = ROOT / "husky_sim" / "src"
for path in (HUSKY_SRC, STAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from skate_bfm01.envs import Bfm0Husky23Env, Bfm0Husky23EnvCfg


def _resolve_output(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items() if k != "extras"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="Mjlab-Skater-Flat-Unitree-G1")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--action-mode", choices=("zero", "random"), default="zero")
    parser.add_argument("--action-mapping", choices=("target_position", "raw_shared"), default="target_position")
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    env = Bfm0Husky23Env(
        Bfm0Husky23EnvCfg(
            task_id=args.task_id,
            device=args.device,
            action_mapping=args.action_mapping,
            action_clip=args.action_clip,
        )
    )
    rows = []
    try:
        obs = env.reset()
        print("[Skate-BFM] BFM0 motion in HUSKY env smoke")
        print(f"state={tuple(obs['state'].shape)} history_actor={tuple(obs['history_actor'].shape)} last_action={tuple(obs['last_action'].shape)}")
        print(f"husky_joints={len(env.mapping_report.husky_joint_names)} shared={len(env.mapping_report.shared_joint_names)} dropped={env.mapping_report.dropped_joint_names}")

        for step in range(args.steps):
            if args.action_mode == "zero":
                action = torch.zeros(29)
            else:
                action = 0.05 * torch.randn(29)
            _, obs, reward, info = env.step(action)
            row = {
                "step": step + 1,
                "reward": reward,
                "root_height": info["root_height"],
                "skateboard_lin_vel_b": _jsonable(info["skateboard_lin_vel_b"]),
                "contact_phase": _jsonable(info["contact_phase"]),
                "terminated": info["terminated"],
                "truncated": info["truncated"],
            }
            rows.append(row)
            print(
                f"step={row['step']} reward={reward:.4f} "
                f"root_h={info['root_height']:.3f} "
                f"board_vx={info['skateboard_lin_vel_b'][0]:.3f}"
            )
    finally:
        env.close()

    if args.output is not None:
        output = _resolve_output(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
