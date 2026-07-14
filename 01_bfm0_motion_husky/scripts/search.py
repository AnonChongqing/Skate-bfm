#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
STAGE = Path(__file__).resolve().parents[1]
ROLLOUT = STAGE / "scripts" / "rollout.py"
for path in (ROOT / "husky_sim" / "src", STAGE, STAGE / "bfm0"):
    sys.path.insert(0, str(path))

from skate_bfm01.policy import BfmPolicy, load_zs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(rows: list[dict], source: str, index: int, steps: int, task: str) -> dict:
    fall_step = next((row["step"] for row in rows if row["root_height"] < 0.3), steps + 1)
    lost_step = next(
        (row["step"] for row in rows if row["skateboard_xy_distance"] > 0.6),
        steps + 1,
    )
    active_steps = min(fall_step - 1, lost_step - 1, len(rows))
    active = rows[: max(1, active_steps)]
    contacts = [float(any(row["feet_board_contact"] or ())) for row in active]
    both_contacts = [float(all(row["feet_board_contact"] or ())) for row in active]
    retained_speeds = [
        max(row["skateboard_lin_vel_b"][0], 0.0) * row["scores"]["board_retention"]
        for row in active
    ]
    if task == "steer":
        initial_error = abs(active[0]["control"]["heading_error"])
        final_error = abs(active[-1]["control"]["heading_error"])
        heading_reduction = initial_error - final_error
        objective = (
            4.0 * active_steps / steps
            + 4.0 * _mean(contacts)
            + 5.0 * _mean(both_contacts)
            + 3.0 * _mean([row["scores"]["board_retention"] for row in active])
            + 2.0 * _mean([row["scores"]["steer_pose"] for row in active])
            + 2.0 * heading_reduction
            - active[-1]["skateboard_xy_distance"]
        )
    else:
        heading_reduction = 0.0
        objective = (
            2.0 * min(fall_step - 1, steps) / steps
            + 5.0 * _mean(contacts)
            + 3.0 * _mean([row["scores"]["board_retention"] for row in active])
            + 2.0 * _mean([row["scores"]["push_task"] for row in active])
            + _mean(retained_speeds)
            + 3.0 * _mean([row["skateboard_lin_vel_b"][0] for row in active])
            - max(active[-1]["skateboard_xy_distance"] - 0.6, 0.0)
        )
    return {
        "index": index,
        "source": source,
        "objective": objective,
        "fall_step": fall_step,
        "lost_step": lost_step,
        "survival": active_steps / steps,
        "mean_height": _mean([row["root_height"] for row in active]),
        "mean_push_task": _mean([row["scores"]["push_task"] for row in active]),
        "mean_board_vx": _mean([row["skateboard_lin_vel_b"][0] for row in active]),
        "mean_retained_speed": _mean(retained_speeds),
        "mean_retention": _mean([row["scores"]["board_retention"] for row in active]),
        "mean_board_distance": _mean([row["skateboard_xy_distance"] for row in active]),
        "final_board_distance": active[-1]["skateboard_xy_distance"],
        "board_contact_rate": _mean(contacts),
        "both_contact_rate": _mean(both_contacts),
        "heading_reduction": heading_reduction,
    }


def _collect_trial(args, candidate: np.ndarray, source: str, index: int, trial: int):
    with tempfile.TemporaryDirectory(prefix=f"skate_bfm_{os.getpid()}_{trial}_") as tmp:
        tmp_path = Path(tmp)
        latent_path = tmp_path / "latent.npy"
        output_path = tmp_path / "rollout.json"
        sample_path = tmp_path / "samples.npz"
        np.save(latent_path, candidate)
        command = [
            sys.executable,
            str(ROLLOUT),
            "--model-path",
            str(args.model_path),
            "--device",
            args.device,
            "--husky-device",
            args.husky_device,
            "--seed",
            str(args.seed),
            "--steps",
            str(args.steps),
            "--mean",
            "--reference-blend",
            str(args.reference_blend),
            "--action-gain",
            str(args.action_gain),
            "--output",
            str(output_path),
            "--samples",
            str(sample_path),
        ]
        if args.task == "steer":
            command.extend(
                [
                    "--z-path", str(args.z_path),
                    "--control", "steer",
                    "--steer-z", str(latent_path),
                    "--turn-index", str(args.turn_index),
                    "--turn-mix", str(args.turn_mix),
                    "--follow-mix", "0",
                    "--steer-initial-speed", str(args.steer_initial_speed),
                    "--sample-reward", "steer",
                ]
            )
        else:
            command.extend(
                [
                    "--z-path", str(latent_path),
                    "--control", "fixed",
                    "--sample-reward", "push",
                ]
            )
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if result.returncode != 0:
            detail = (result.stdout + result.stderr)[-4000:]
            raise RuntimeError(f"Explorer {source}[{index}] failed:\n{detail}")
        rows = json.loads(output_path.read_text(encoding="utf-8"))
        with np.load(sample_path) as samples:
            batch = {name: samples[name].copy() for name in samples.files if name != "reward"}
            reward = samples["reward"].copy()
        return _summarize(rows, source, index, args.steps, args.task), batch, reward


def main() -> None:
    data = Path(os.environ.get("SKATE_BFM_DATA", "/63data1/hwh_data/Skate-bfm"))
    model = data / "models" / "bfm0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=model)
    parser.add_argument("--z-path", type=Path, default=model / "reward_inference/reward_locomotion.pkl")
    parser.add_argument("--z-key", default="move-ego-0-0.7")
    parser.add_argument("--task", choices=("push", "steer"), default="push")
    parser.add_argument("--indices", default=None, help="Comma-separated official indices; default is all.")
    parser.add_argument("--anchor-key", default=None, help="Optional reward latent used as blend anchor.")
    parser.add_argument("--anchor-index", type=int, default=0)
    parser.add_argument("--mixes", default="1.0", help="Comma-separated candidate weights relative to anchor.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--random-latents", type=int, default=0)
    parser.add_argument("--reference-blend", type=float, default=0.0)
    parser.add_argument("--action-gain", type=float, default=1.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--husky-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turn-index", type=int, default=9)
    parser.add_argument("--turn-mix", type=float, default=0.1)
    parser.add_argument("--steer-initial-speed", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=data / "prompts" / "push.npy")
    parser.add_argument("--best-output", type=Path, default=data / "prompts" / "push_best.npy")
    parser.add_argument("--report", type=Path, default=data / "runs" / "search.json")
    args = parser.parse_args()
    if args.task == "steer" and args.z_path.suffix == ".npy":
        parser.error("Steer search needs the BFM reward dictionary for signed rotate latents.")
    report = args.report if args.report.is_absolute() else ROOT / args.report

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = BfmPolicy(args.model_path, args.z_path, args.z_key, args.device, True)
    official = load_zs(args.z_path, args.z_key).to(args.device)
    official_indices = (
        [int(item) for item in args.indices.split(",") if item]
        if args.indices
        else list(range(len(official)))
    )
    mixes = [float(item) for item in args.mixes.split(",") if item]
    anchor = load_zs(args.z_path, args.anchor_key)[args.anchor_index].to(args.device) if args.anchor_key else None
    explorers = []
    for index in official_indices:
        for mix in mixes:
            if not 0.0 <= mix <= 1.0:
                raise ValueError(f"Mix must be in [0, 1], got {mix}")
            candidate = official[index]
            source = f"official:{args.z_key}"
            if anchor is not None:
                candidate = policy.project((1.0 - mix) * anchor + mix * candidate)
                source = f"blend:{args.anchor_key}[{args.anchor_index}]+{mix:g}*{args.z_key}"
            explorers.append((source, index, candidate))
    if args.random_latents:
        random = policy.model.sample_z(args.random_latents, device=args.device)
        explorers.extend(("random", index, latent) for index, latent in enumerate(random))

    rows = []
    observations: dict[str, list[np.ndarray]] = {}
    task_rewards: list[np.ndarray] = []
    for trial, (source, index, candidate) in enumerate(explorers):
        row, batch, reward = _collect_trial(
            args,
            candidate.detach().cpu().numpy(),
            source,
            index,
            trial,
        )
        rows.append(row)
        for name, values in batch.items():
            observations.setdefault(name, []).append(values)
        task_rewards.append(reward)
        print(f"candidate={trial} {json.dumps(row)}", flush=True)

    batch = {name: torch.from_numpy(np.concatenate(values)) for name, values in observations.items()}
    reward = torch.from_numpy(np.concatenate(task_rewards)).float()
    inferred = policy.infer_reward(batch, reward, weighted=True)
    best_trial = max(range(len(rows)), key=lambda index: rows[index]["objective"])
    best = rows[best_trial]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, inferred.detach().cpu().numpy())
    args.best_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.best_output, explorers[best_trial][2].detach().cpu().numpy())
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "method": "bfm0_reward_inference_isolated_rollouts",
                "task": args.task,
                "reward": (
                    "upright * proximity * (0.35 * both_contact + 0.20 * board_contact "
                    "+ 0.20 * steer_pose + 0.10 * steer_feet + 0.15 * heading)"
                    if args.task == "steer"
                    else "upright * (0.45 * board_contact * proximity + 0.55 * push_task)"
                ),
                "samples": int(reward.numel()),
                "reward_mean": float(reward.mean()),
                "reward_max": float(reward.max()),
                "best_explorer": best,
                "best_output": str(args.best_output),
                "explorers": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"inferred from {reward.numel()} isolated HUSKY samples; wrote {args.output}")
    print(f"best explorer={best['source']}[{best['index']}]; wrote {args.best_output}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
