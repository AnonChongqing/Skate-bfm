#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = Path(__file__).resolve().parents[1]
HUSKY_SRC = ROOT / "husky_sim" / "src"
BFM_LOCAL = STAGE_ROOT / "bfm0"
for path in (HUSKY_SRC, STAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(BFM_LOCAL) not in sys.path:
    sys.path.append(str(BFM_LOCAL))

from skate_bfm01.envs import Bfm0Husky23Env, Bfm0Husky23EnvCfg
from skate_bfm01.control import make_control
from skate_bfm01.policy import BfmPolicy
from skate_bfm01.score import ScoreCfg, compute_scores
from skate_bfm01.viewer import run_viser


def _resolve_output(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise RuntimeError("No frames were rendered; video was not written.")
    from moviepy import ImageSequenceClip

    path.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(path), logger=None)
    clip.close()


def main() -> None:
    data_root = Path(os.environ.get("SKATE_BFM_DATA", "/63data1/hwh_data/Skate-bfm"))
    default_model = data_root / "models" / "bfm0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument("--z-path", type=Path, default=None)
    parser.add_argument("--z-key", default="move-ego-0-0.7")
    parser.add_argument("--z-index", type=int, default=0)
    parser.add_argument("--task-id", default="Mjlab-Skater-Flat-Unitree-G1")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="BFM model device.")
    parser.add_argument("--husky-device", default=None)
    parser.add_argument(
        "--action-mapping",
        choices=("reference", "nominal_aligned", "target_position", "raw_shared"),
        default="reference",
    )
    parser.add_argument("--action-clip", type=float, default=None)
    parser.add_argument("--reference-blend", type=float, default=0.0)
    parser.add_argument("--action-gain", type=float, default=1.25)
    parser.add_argument("--command-speed", type=float, default=0.7)
    parser.add_argument("--command-heading", type=float, default=0.4)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--reset-noise", type=float, default=0.0)
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--output", type=Path, default=data_root / "runs" / "rollout.json")
    parser.add_argument("--samples", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--sample-reward",
        choices=("auto", "push", "steer"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--video",
        nargs="?",
        type=Path,
        const=data_root / "runs" / "run.mp4",
        default=None,
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--train-cfg", action="store_true", help="Use training resets and terminations.")
    parser.add_argument("--viewer", choices=("none", "viser"), default="none")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--control", choices=("fixed", "phase", "steer"), default="phase")
    parser.add_argument("--initial-mode", choices=("push", "steer"), default=None)
    parser.add_argument("--steer-initial-speed", type=float, default=0.7)
    parser.add_argument("--push-z", type=Path, default=data_root / "prompts" / "push_back.npy")
    parser.add_argument("--steer-z", type=Path, default=None)
    parser.add_argument("--steer-key", default="move-ego-0-0")
    parser.add_argument("--steer-index", type=int, default=0)
    parser.add_argument("--follow-mix", type=float, default=0.0)
    parser.add_argument("--follow-index", type=int, default=0)
    parser.add_argument("--follow-start", type=float, default=0.08)
    parser.add_argument("--follow-pos-gain", type=float, default=1.0)
    parser.add_argument("--follow-vel-gain", type=float, default=0.0)
    parser.add_argument("--follow-tilt-gain", type=float, default=0.0)
    parser.add_argument("--follow-foot-gain", type=float, default=0.0)
    parser.add_argument("--start-steps", type=int, default=0)
    parser.add_argument("--start-key", default="move-ego-0-0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--transition-start", type=float, default=0.35)
    parser.add_argument("--transition-enter", type=float, default=0.4)
    parser.add_argument("--transition-scale", type=float, default=1.0)
    parser.add_argument("--transition-mix", type=float, default=0.5)
    parser.add_argument("--transition-blend", type=float, default=0.7)
    parser.add_argument("--transition-steps", type=int, default=18)
    parser.add_argument("--trigger-after", type=float, default=0.12)
    parser.add_argument("--trigger-speed", type=float, default=0.2)
    parser.add_argument("--trigger-distance", type=float, default=0.25)
    parser.add_argument("--turn-mix", type=float, default=0.05)
    parser.add_argument("--turn-index", type=int, default=0)
    parser.add_argument("--recover-height", type=float, default=0.0)
    parser.add_argument("--stable-height", type=float, default=0.01)
    parser.add_argument("--push-feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push-hold-key", default="move-ego-0-0")
    parser.add_argument("--push-hold-index", type=int, default=0)
    parser.add_argument("--push-drive-mix", type=float, default=0.3)
    parser.add_argument("--push-hold-steps", type=int, default=15)
    parser.add_argument("--push-drive-steps", type=int, default=8)
    args = parser.parse_args()
    if args.video is not None and args.viewer != "none":
        parser.error("Choose either --viewer viser or --video, not both.")
    if args.z_path is None:
        args.z_path = args.model_path / "reward_inference" / "reward_locomotion.pkl"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = BfmPolicy(args.model_path, args.z_path, args.z_key, args.device, args.mean, args.z_index)
    env = Bfm0Husky23Env(
        Bfm0Husky23EnvCfg(
            task_id=args.task_id,
            seed=args.seed,
            device=args.husky_device,
            action_mapping=args.action_mapping,
            action_clip=args.action_clip,
            observation_mapping="reference" if args.action_mapping == "reference" else (
                "nominal_aligned" if args.action_mapping == "nominal_aligned" else "bfm_absolute"
            ),
            reference_blend=args.reference_blend,
            action_gain=args.action_gain,
            command_speed=args.command_speed,
            command_heading=args.command_heading,
            domain_randomization=args.domain_randomization,
            reset_noise=args.reset_noise,
            play=not args.train_cfg,
            render_mode="rgb_array" if args.video is not None else None,
            width=args.width,
            height=args.height,
            initial_mode=args.initial_mode or ("steer" if args.control == "steer" else "push"),
            steer_initial_speed=args.steer_initial_speed,
        )
    )

    rows = []
    frames: list[np.ndarray] = []
    sample_obs: dict[str, list[np.ndarray]] = {}
    sample_rewards: list[float] = []
    score_cfg = ScoreCfg()
    try:
        obs = env.reset()
        control = make_control(
            args.control,
            policy,
            env,
            reward_path=args.z_path,
            push_path=args.push_z,
            steer_path=args.steer_z,
            start_steps=args.start_steps,
            start_key=args.start_key,
            start_index=args.start_index,
            transition_start=args.transition_start,
            transition_enter=args.transition_enter,
            transition_scale=args.transition_scale,
            transition_mix=args.transition_mix,
            transition_blend=args.transition_blend,
            transition_steps=args.transition_steps,
            trigger_after=args.trigger_after,
            trigger_speed=args.trigger_speed,
            trigger_distance=args.trigger_distance,
            turn_mix=args.turn_mix,
            turn_index=args.turn_index,
            recover_height=args.recover_height,
            stable_height=args.stable_height,
            push_feedback=args.push_feedback,
            push_hold_key=args.push_hold_key,
            push_hold_index=args.push_hold_index,
            push_drive_mix=args.push_drive_mix,
            push_hold_steps=args.push_hold_steps,
            push_drive_steps=args.push_drive_steps,
            steer_key=args.steer_key,
            steer_index=args.steer_index,
            follow_mix=args.follow_mix,
            follow_index=args.follow_index,
            follow_start=args.follow_start,
            follow_pos_gain=args.follow_pos_gain,
            follow_vel_gain=args.follow_vel_gain,
            follow_tilt_gain=args.follow_tilt_gain,
            follow_foot_gain=args.follow_foot_gain,
        )
        print("[Skate-BFM] rolling BFM0 motion on HUSKY simulation")
        print(f"control={args.control}")
        print(f"z_shape={tuple(policy.z.shape)} dropped={env.mapping_report.dropped_joint_names}")
        if args.viewer == "viser":
            steps = None if args.steps <= 0 else args.steps
            print(f"[Skate-BFM] Viser: http://127.0.0.1:{args.port}")
            run_viser(env, control, port=args.port, steps=steps)
            return
        for step in range(args.steps):
            action = control(obs)
            _, obs, reward, info = env.step(action)
            scores = compute_scores(env.husky_env, score_cfg)
            score_row = {name: float(value[0].detach().cpu()) for name, value in scores.items()}
            if args.samples is not None:
                for name, value in obs.items():
                    sample_obs.setdefault(name, []).append(value.detach().cpu().numpy().copy())
                sample_reward = (
                    "steer" if args.sample_reward == "auto" and args.control == "steer"
                    else "push" if args.sample_reward == "auto"
                    else args.sample_reward
                )
                if sample_reward == "steer":
                    both_contact = min(max(score_row["steer_contact"] / 2.0, 0.0), 1.0)
                    heading_error = float(control.status.get("heading_error", 0.0))
                    heading = float(np.exp(-np.square(heading_error / 0.3)))
                    balance = (
                        0.35 * both_contact
                        + 0.20 * score_row["board_contact"]
                        + 0.20 * score_row["steer_pose"]
                        + 0.10 * score_row["steer_feet"]
                        + 0.15 * heading
                    )
                    task_reward = (
                        score_row["upright"] * score_row["board_proximity"] * balance
                    )
                else:
                    board_contact = score_row["board_contact"]
                    proximity = score_row["board_proximity"]
                    task_reward = score_row["upright"] * (
                        0.45 * board_contact * proximity + 0.55 * score_row["push_task"]
                    )
                sample_rewards.append(task_reward)
            if args.video is not None:
                frame = env.render()
                if isinstance(frame, np.ndarray) and frame.ndim == 4:
                    frame = frame[0]
                if frame is not None:
                    frame = np.asarray(frame)
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
                    frames.append(frame)
            row = {
                "step": step + 1,
                "husky_reward": reward,
                "root_height": info["root_height"],
                "root_position": info["root_position"].tolist(),
                "terminated": info["terminated"],
                "truncated": info["truncated"],
                "skateboard_position": info["skateboard_position"].tolist(),
                "skateboard_relative_position": info["skateboard_relative_position"].tolist(),
                "skateboard_xy_distance": info["skateboard_xy_distance"],
                "skateboard_lin_vel_b": info["skateboard_lin_vel_b"].tolist(),
                "skateboard_heading_w": info["skateboard_heading_w"],
                "contact_phase": info["contact_phase"].tolist(),
                "command": info["command"].tolist() if info["command"] is not None else None,
                "feet_board_contact": (
                    info["feet_board_contact"].tolist()
                    if info["feet_board_contact"] is not None
                    else None
                ),
                "feet_ground_contact": (
                    info["feet_ground_contact"].tolist()
                    if info["feet_ground_contact"] is not None
                    else None
                ),
                "scores": score_row,
                "control": dict(control.status),
            }
            rows.append(row)
            if step % 25 == 0:
                print(
                    f"step={step + 1} husky_reward={reward:.4f} root_h={info['root_height']:.3f} "
                    f"board_vx={info['skateboard_lin_vel_b'][0]:.3f} "
                    f"board_d={info['skateboard_xy_distance']:.3f} "
                    f"push={score_row['push_task']:.3f} steer={score_row['steer']:.3f} "
                    f"transition_goal={score_row['transition_goal']:.3f}"
                )
            if info["terminated"] or info["truncated"]:
                print(f"episode ended at step={step + 1}")
                break
    finally:
        env.close()

    output = _resolve_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    if args.samples is not None:
        samples = _resolve_output(args.samples)
        samples.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            samples,
            reward=np.asarray(sample_rewards, dtype=np.float32),
            **{name: np.stack(values) for name, values in sample_obs.items()},
        )
        print(f"wrote {samples}")
    if args.video is not None:
        video = _resolve_output(args.video)
        fps = 1.0 / env.husky_env.step_dt
        write_video(video, frames, fps)
        print(f"wrote {video}")


if __name__ == "__main__":
    main()
