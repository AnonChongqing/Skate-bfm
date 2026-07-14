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
for path in (STAGE, STAGE / "bfm0"):
    sys.path.insert(0, str(path))

from skate_bfm01.policy import BfmPolicy, load_z


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score(rows: list[dict], distance_limit: float) -> dict[str, float]:
    valid = []
    for row in rows:
        if row["root_height"] < 0.3 or row["skateboard_xy_distance"] > distance_limit:
            break
        valid.append(row)
    active = valid or rows[:1]
    any_contact = _mean(
        [float(any(row.get("feet_board_contact") or ())) for row in active]
    )
    both_contact = _mean(
        [float(all(row.get("feet_board_contact") or ())) for row in active]
    )
    continuous_steps = 0
    for row in rows:
        if not any(row.get("feet_board_contact") or ()):
            break
        continuous_steps += 1
    continuous_contact = continuous_steps / len(rows)
    retention = _mean([row["scores"]["board_retention"] for row in active])
    pose = _mean([row["scores"]["steer_pose"] for row in active])
    heading = _mean(
        [float(np.exp(-np.square(row["control"]["heading_error"] / 0.2))) for row in active]
    )
    survival = len(valid) / len(rows)
    final_distance = active[-1]["skateboard_xy_distance"]
    objective = (
        5.0 * any_contact
        + 8.0 * both_contact
        + 8.0 * continuous_contact
        + 4.0 * retention
        + 2.0 * pose
        + 3.0 * heading
        + 4.0 * survival
        - 2.0 * final_distance
    )
    return {
        "objective": objective,
        "survival": survival,
        "active_steps": len(valid),
        "any_contact": any_contact,
        "both_contact": both_contact,
        "continuous_contact": continuous_contact,
        "continuous_steps": continuous_steps,
        "retention": retention,
        "steer_pose": pose,
        "heading": heading,
        "final_distance": final_distance,
        "min_height": min(row["root_height"] for row in active),
    }


def _rollout(args, latent: np.ndarray, iteration: int, candidate: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix=f"skate_bfm_adapt_{os.getpid()}_") as tmp:
        tmp_path = Path(tmp)
        latent_path = tmp_path / "latent.npy"
        output_path = tmp_path / "rollout.json"
        np.save(latent_path, latent)
        command = [
            sys.executable,
            str(ROLLOUT),
            "--model-path", str(args.model_path),
            "--z-path", str(args.reward_path),
            "--device", args.device,
            "--husky-device", args.husky_device,
            "--seed", str(args.seed),
            "--steps", str(args.steps),
            "--control", "steer",
            "--steer-z", str(latent_path),
            "--turn-index", str(args.turn_index),
            "--turn-mix", str(args.turn_mix),
            "--follow-mix", str(args.follow_mix),
            "--steer-initial-speed", str(args.steer_initial_speed),
            "--mean",
            "--output", str(output_path),
        ]
        command.extend(
            [
                "--follow-start", str(args.follow_start),
                "--follow-pos-gain", str(args.follow_pos_gain),
                "--follow-vel-gain", str(args.follow_vel_gain),
                "--follow-tilt-gain", str(args.follow_tilt_gain),
                "--follow-foot-gain", str(args.follow_foot_gain),
            ]
        )
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if result.returncode != 0:
            detail = (result.stdout + result.stderr)[-4000:]
            raise RuntimeError(
                f"Adapt rollout iteration={iteration} candidate={candidate} failed:\n{detail}"
            )
        return _score(json.loads(output_path.read_text(encoding="utf-8")), args.distance_limit)


def main() -> None:
    data = Path(os.environ.get("SKATE_BFM_DATA", "/63data1/hwh_data/Skate-bfm"))
    model = data / "models" / "bfm0"
    reward = model / "reward_inference" / "reward_locomotion.pkl"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=model)
    parser.add_argument("--reward-path", type=Path, default=reward)
    parser.add_argument("--base-key", default="move-ego-0-0")
    parser.add_argument("--base-index", type=int, default=0)
    parser.add_argument("--base-z", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--sigma-decay", type=float, default=0.6)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--distance-limit", type=float, default=0.6)
    parser.add_argument("--turn-index", type=int, default=9)
    parser.add_argument("--turn-mix", type=float, default=0.1)
    parser.add_argument("--steer-initial-speed", type=float, default=0.7)
    parser.add_argument("--follow-mix", type=float, default=0.0)
    parser.add_argument("--follow-start", type=float, default=0.08)
    parser.add_argument("--follow-pos-gain", type=float, default=1.0)
    parser.add_argument("--follow-vel-gain", type=float, default=0.0)
    parser.add_argument("--follow-tilt-gain", type=float, default=0.0)
    parser.add_argument("--follow-foot-gain", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--husky-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=data / "prompts" / "steer_adapt.npy")
    parser.add_argument("--report", type=Path, default=data / "runs" / "steer_adapt.json")
    args = parser.parse_args()
    if args.population < 2 or not 1 <= args.elite < args.population:
        parser.error("Require population >= 2 and 1 <= elite < population.")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    policy = BfmPolicy(
        args.model_path,
        args.reward_path,
        args.base_key,
        args.device,
        True,
        args.base_index,
    )
    mean = (
        torch.as_tensor(np.load(args.base_z), device=args.device, dtype=torch.float32).reshape(-1)
        if args.base_z is not None
        else load_z(args.reward_path, args.base_key, args.base_index).to(args.device)
    )
    mean = policy.project(mean)
    sigma = args.sigma
    best_z = mean.clone()
    best_score = float("-inf")
    history = []

    for iteration in range(args.iterations):
        candidates = [mean]
        for _ in range(1, args.population):
            noise = torch.as_tensor(
                rng.normal(0.0, sigma, size=mean.shape),
                device=args.device,
                dtype=torch.float32,
            )
            candidates.append(policy.project(mean + noise))

        rows = []
        for candidate, latent in enumerate(candidates):
            metrics = _rollout(
                args,
                latent.detach().cpu().numpy(),
                iteration,
                candidate,
            )
            metrics.update({"iteration": iteration, "candidate": candidate})
            rows.append(metrics)
            print(json.dumps(metrics), flush=True)

        elite_indices = np.argsort([row["objective"] for row in rows])[-args.elite:]
        elites = torch.stack([candidates[int(index)] for index in elite_indices])
        mean = policy.project(torch.mean(elites, dim=0))
        iteration_best = int(np.argmax([row["objective"] for row in rows]))
        if rows[iteration_best]["objective"] > best_score:
            best_score = rows[iteration_best]["objective"]
            best_z = candidates[iteration_best].clone()
        history.extend(rows)
        sigma = max(args.sigma_min, sigma * args.sigma_decay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, best_z.detach().cpu().numpy())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "method": "frozen_bfm_latent_cem",
                "best_objective": best_score,
                "output": str(args.output),
                "config": vars(args) | {"model_path": str(args.model_path), "reward_path": str(args.reward_path), "base_z": str(args.base_z) if args.base_z else None, "output": str(args.output), "report": str(args.report)},
                "trials": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
