#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ROLLOUT = Path(__file__).with_name("rollout.py")
STAGE = Path(__file__).resolve().parents[1]
for path in (STAGE, STAGE / "bfm0"):
    sys.path.insert(0, str(path))

from skate_bfm01.policy import load_zs


def main() -> None:
    data = Path(os.environ.get("SKATE_BFM_DATA", "/63data1/hwh_data/Skate-bfm"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--kind", choices=("push", "turn", "direct"), default="turn")
    parser.add_argument("--steps", type=int, default=190)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--husky-device", default="cuda:0")
    parser.add_argument("--steer-key", default="move-ego-0-0")
    parser.add_argument("--steer-z", type=Path, default=None)
    parser.add_argument("--turn-indices", default="0")
    parser.add_argument("--turn-mixes", default="0.1")
    parser.add_argument("--steer-initial-speed", type=float, default=0.7)
    parser.add_argument("--follow-mix", type=float, default=0.0)
    parser.add_argument("--follow-index", type=int, default=0)
    parser.add_argument("--follow-indices", default=None)
    parser.add_argument("--follow-start", type=float, default=0.08)
    parser.add_argument("--follow-pos-gain", type=float, default=1.0)
    parser.add_argument("--follow-vel-gain", type=float, default=0.0)
    parser.add_argument("--follow-tilt-gain", type=float, default=0.0)
    parser.add_argument("--follow-foot-gain", type=float, default=0.0)
    parser.add_argument("--report", type=Path, default=data / "runs" / "sweep.json")
    args = parser.parse_args()
    report = args.report if args.report.is_absolute() else ROOT / args.report
    trial_dir = data / "trials" / args.kind
    trial_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    steer_indices = [int(value) for value in args.indices.split(",")]
    turn_indices = [int(value) for value in args.turn_indices.split(",")]
    turn_mixes = [float(value) for value in args.turn_mixes.split(",")]
    follow_indices = (
        [int(value) for value in args.follow_indices.split(",")]
        if args.follow_indices
        else [args.follow_index]
    )
    candidates = (
        [(steer_index, turn_index, turn_mix, follow_index) for steer_index in steer_indices
         for turn_index in turn_indices for turn_mix in turn_mixes
         for follow_index in follow_indices]
        if args.kind == "direct"
        else [(index, 0, 0.0, args.follow_index) for index in steer_indices]
    )
    for index, turn_index, turn_mix, follow_index in candidates:
        suffix = (
            f"{index}_turn{turn_index}_mix{turn_mix:g}_follow{follow_index}"
            if args.kind == "direct" else str(index)
        )
        output = trial_dir / f"{args.kind}_{suffix}.json"
        command = [
            sys.executable,
            str(ROLLOUT),
            "--device", args.device,
            "--husky-device", args.husky_device,
            "--mean",
        ]
        if args.kind == "push":
            reward_path = data / "models/bfm0/reward_inference/reward_locomotion.pkl"
            command.extend(
                [
                    "--control", "fixed",
                    "--z-path", str(reward_path),
                    "--z-key", "move-ego-0-0.7",
                    "--z-index", str(index),
                ]
            )
        elif args.kind == "turn":
            command.extend(
                [
                    "--control", "phase",
                    "--turn-index", str(index),
                ]
            )
        else:
            command.extend(
                [
                    "--control", "steer",
                    "--steer-key", args.steer_key,
                    "--steer-index", str(index),
                    "--turn-index", str(turn_index),
                    "--turn-mix", str(turn_mix),
                    "--steer-initial-speed", str(args.steer_initial_speed),
                    "--follow-mix", str(args.follow_mix),
                    "--follow-index", str(follow_index),
                    "--follow-start", str(args.follow_start),
                    "--follow-pos-gain", str(args.follow_pos_gain),
                    "--follow-vel-gain", str(args.follow_vel_gain),
                    "--follow-tilt-gain", str(args.follow_tilt_gain),
                    "--follow-foot-gain", str(args.follow_foot_gain),
                ]
            )
            if args.steer_z is not None:
                command.extend(["--steer-z", str(args.steer_z)])
        command.extend(["--steps", str(args.steps), "--output", str(output)])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        rollout = json.loads(output.read_text(encoding="utf-8"))
        fall_step = next((row["step"] for row in rollout if row["root_height"] < 0.3), args.steps + 1)
        lost_step = next(
            (row["step"] for row in rollout if row.get("skateboard_xy_distance", 0.0) > 0.6),
            args.steps + 1,
        )
        steer = [
            row
            for row in rollout
            if row.get("control", {}).get("state") in {"steer", "steer_only"}
            and row["root_height"] >= 0.3
        ]
        initial_error = abs(steer[0]["control"]["heading_error"]) if steer else 0.0
        final_error = abs(steer[-1]["control"]["heading_error"]) if steer else initial_error
        continuous_steps = 0
        for item in rollout:
            if not any(item.get("feet_board_contact") or ()):
                break
            continuous_steps += 1
        row = {
            "index": index,
            "turn_index": turn_index,
            "turn_mix": turn_mix,
            "follow_index": follow_index,
            "fall_step": fall_step,
            "lost_step": lost_step,
            "survived": min(fall_step - 1, lost_step - 1, args.steps),
            "heading_reduction": initial_error - final_error,
            "mean_board_vx": float(np.mean([item["skateboard_lin_vel_b"][0] for item in rollout])),
            "board_displacement": float(
                rollout[-1].get("skateboard_position", [0.0])[0]
                - rollout[0].get("skateboard_position", [0.0])[0]
            ),
            "min_upright_height": float(np.min([item["root_height"] for item in rollout[: max(fall_step - 1, 1)]])),
            "mean_push": float(
                np.mean([item["scores"].get("push_task", item["scores"]["push_quality"]) for item in rollout])
            ),
            "mean_retention": float(
                np.mean([item["scores"].get("board_retention", 0.0) for item in rollout])
            ),
            "any_contact_rate": float(
                np.mean([any(item.get("feet_board_contact") or ()) for item in rollout])
            ),
            "both_contact_rate": float(
                np.mean([all(item.get("feet_board_contact") or ()) for item in rollout])
            ),
            "final_board_distance": float(rollout[-1]["skateboard_xy_distance"]),
            "continuous_contact_steps": continuous_steps,
            "continuous_contact_rate": continuous_steps / len(rollout),
        }
        if args.kind == "direct":
            row["objective"] = (
                row["survived"]
                + 80.0 * row["heading_reduction"]
                + 40.0 * row["any_contact_rate"]
                + 20.0 * row["both_contact_rate"]
                + 40.0 * row["continuous_contact_rate"]
                + 5.0 * max(row["mean_board_vx"], 0.0)
                - 20.0 * row["final_board_distance"]
            )
        else:
            row["objective"] = (
                row["survived"]
                + 100.0 * row["heading_reduction"]
                + 5.0 * row["mean_board_vx"]
                + 20.0 * row["board_displacement"]
                + (10.0 * row["mean_push"] + 5.0 * row["mean_retention"] if args.kind == "push" else 0.0)
            )
        rows.append(row)
        print(json.dumps(row), flush=True)

    best = max(rows, key=lambda row: row["objective"])
    if args.kind == "push":
        reward_path = data / "models/bfm0/reward_inference/reward_locomotion.pkl"
        candidates = load_zs(reward_path, "move-ego-0-0.7")
        push_path = data / "prompts/push.npy"
        push_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(push_path, candidates[best["index"]].numpy())
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"best": best, "candidates": rows}, indent=2), encoding="utf-8")
    print(f"best={json.dumps(best)}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
