#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGE = Path(__file__).resolve().parents[1]
ROLLOUT = STAGE / "scripts" / "rollout.py"


def _values(value: str, cast):
    return [cast(item) for item in value.split(",") if item]


def _summarize(rows: list[dict], cfg: dict, steps: int) -> dict:
    fall_step = next((row["step"] for row in rows if row["root_height"] < 0.3), steps + 1)
    lost_step = next(
        (row["step"] for row in rows if row.get("skateboard_xy_distance", 0.0) > 0.6),
        steps + 1,
    )
    survived = min(fall_step - 1, lost_step - 1, steps)
    active = rows[: max(1, survived)]
    transition = [row for row in active if row["control"]["state"] == "push2steer"]
    late = active[min(len(active) - 1, 104) :] or active[-1:]
    contacts = [
        sum(row["feet_board_contact"]) / len(row["feet_board_contact"])
        for row in transition
        if row["feet_board_contact"] is not None
    ]
    reached_steer = any(row["control"]["state"] == "steer" for row in active)
    mean = lambda values: sum(values) / len(values) if values else 0.0
    displacement = active[-1].get("skateboard_position", [0.0])[0] - active[0].get(
        "skateboard_position", [0.0]
    )[0]
    active_contacts = [
        float(any(row["feet_board_contact"]))
        for row in active
        if row["feet_board_contact"] is not None
    ]
    objective = (
        12.0 * survived / steps
        + 3.0 * float(reached_steer)
        + 2.0 * mean([row["root_height"] for row in late])
        + mean([row["scores"]["transition_goal"] for row in transition])
        + mean(contacts)
        + displacement
        + mean([row["scores"].get("board_retention", 0.0) for row in active])
        + mean(active_contacts)
    )
    return {
        **cfg,
        "objective": objective,
        "fall_step": fall_step,
        "lost_step": lost_step,
        "survived_steps": survived,
        "reached_steer": reached_steer,
        "mean_height": mean([row["root_height"] for row in active]),
        "min_transition_height": min(row["root_height"] for row in late),
        "mean_transition_goal": mean([row["scores"]["transition_goal"] for row in transition]),
        "mean_board_contact": mean(contacts),
        "active_board_contact": mean(active_contacts),
        "mean_board_vx": mean([row["skateboard_lin_vel_b"][0] for row in active]),
        "board_displacement": displacement,
        "max_board_distance": max(row.get("skateboard_xy_distance", 0.0) for row in rows),
    }


def _run_trial(args, cfg: dict, index: int) -> dict:
    output = Path(tempfile.gettempdir()) / f"skate_bfm_tune_{os.getpid()}_{index}.json"
    command = [
        sys.executable,
        str(ROLLOUT),
        "--model-path",
        str(args.model_path),
        "--z-path",
        str(args.reward_path),
        "--push-z",
        str(args.push_z),
        "--device",
        args.device,
        "--husky-device",
        args.husky_device,
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--control",
        "phase",
        "--mean",
        "--start-key",
        cfg["start_key"],
        "--start-index",
        str(cfg["start_index"]),
        "--start-steps",
        str(cfg["start_steps"]),
        "--transition-start",
        str(cfg["transition_start"]),
        "--transition-enter",
        str(cfg["transition_enter"]),
        "--transition-scale",
        str(cfg["transition_scale"]),
        "--transition-mix",
        str(cfg["transition_mix"]),
        "--transition-blend",
        str(cfg["transition_blend"]),
        "--turn-mix",
        str(cfg["turn_mix"]),
        "--reference-blend",
        str(cfg["reference_blend"]),
        "--action-gain",
        str(cfg["action_gain"]),
        "--push-drive-mix",
        str(cfg["push_drive_mix"]),
        "--push-hold-steps",
        str(cfg["push_hold_steps"]),
        "--push-drive-steps",
        str(cfg["push_drive_steps"]),
        "--trigger-speed",
        str(cfg["trigger_speed"]),
        "--trigger-distance",
        str(cfg["trigger_distance"]),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        detail = (result.stdout + result.stderr)[-4000:]
        raise RuntimeError(f"Trial {index} failed:\n{detail}")
    try:
        rows = json.loads(output.read_text(encoding="utf-8"))
    finally:
        output.unlink(missing_ok=True)
    return _summarize(rows, cfg, args.steps)


def main() -> None:
    data = Path(os.environ.get("SKATE_BFM_DATA", "/63data1/hwh_data/Skate-bfm"))
    model = data / "models" / "bfm0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=model)
    parser.add_argument("--reward-path", type=Path, default=model / "reward_inference/reward_locomotion.pkl")
    parser.add_argument("--push-z", type=Path, default=data / "prompts/push_back.npy")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--husky-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-keys", default="move-ego-0-0")
    parser.add_argument("--start-indices", default="0")
    parser.add_argument("--start-steps", default="0")
    parser.add_argument("--transition-starts", default="0.35")
    parser.add_argument("--transition-enters", default="0.4")
    parser.add_argument("--transition-scales", default="1.0")
    parser.add_argument("--transition-mixes", default="0.5")
    parser.add_argument("--transition-blends", default="0.7")
    parser.add_argument("--turn-mixes", default="0.05")
    parser.add_argument("--reference-blends", default="0.0")
    parser.add_argument("--action-gains", default="1.25")
    parser.add_argument("--push-drive-mixes", default="0.2,0.3,0.4")
    parser.add_argument("--push-hold-steps", default="12,15,20")
    parser.add_argument("--push-drive-steps", default="6,8")
    parser.add_argument("--trigger-speeds", default="0.2")
    parser.add_argument("--trigger-distances", default="0.25")
    parser.add_argument("--report", type=Path, default=data / "runs" / "tune.json")
    args = parser.parse_args()
    report = args.report if args.report.is_absolute() else ROOT / args.report

    grid = itertools.product(
        _values(args.start_keys, str),
        _values(args.start_indices, int),
        _values(args.start_steps, int),
        _values(args.transition_starts, float),
        _values(args.transition_enters, float),
        _values(args.transition_scales, float),
        _values(args.transition_mixes, float),
        _values(args.transition_blends, float),
        _values(args.turn_mixes, float),
        _values(args.reference_blends, float),
        _values(args.action_gains, float),
        _values(args.push_drive_mixes, float),
        _values(args.push_hold_steps, int),
        _values(args.push_drive_steps, int),
        _values(args.trigger_speeds, float),
        _values(args.trigger_distances, float),
    )
    rows = []
    for index, values in enumerate(grid):
        keys = (
            "start_key",
            "start_index",
            "start_steps",
            "transition_start",
            "transition_enter",
            "transition_scale",
            "transition_mix",
            "transition_blend",
            "turn_mix",
            "reference_blend",
            "action_gain",
            "push_drive_mix",
            "push_hold_steps",
            "push_drive_steps",
            "trigger_speed",
            "trigger_distance",
        )
        cfg = dict(zip(keys, values, strict=True))
        row = _run_trial(args, cfg, index)
        rows.append(row)
        print(f"candidate={index} {json.dumps(row)}", flush=True)

    best = max(rows, key=lambda row: row["objective"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"best": best, "candidates": rows}, indent=2), encoding="utf-8")
    print(f"best={json.dumps(best)}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
