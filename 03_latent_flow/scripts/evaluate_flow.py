from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import torch

from skate_bfm_flow.config import load_config
from skate_bfm_flow.env.macro_env import LatentFlowMacroEnv
from skate_bfm_flow.env.viewer import run_viser
from skate_bfm_flow.evaluation.metrics import mean_metrics
from skate_bfm_flow.evaluation.rollout import rollout_episode
from skate_bfm_flow.models.flow_policy import FlowPolicy

STANDARD_SCENARIOS = (
    ("straight_slow", 0.4, 0.0),
    ("straight", 0.8, 0.0),
    ("left", 0.8, 0.4),
    ("right", 0.8, -0.4),
)


def set_command(env: LatentFlowMacroEnv, speed: float, heading: float) -> None:
    command = env.low_env.husky_env.command_manager._terms["skate"]
    command.cfg.ranges.lin_vel_x = (speed, speed)
    command.cfg.ranges.heading = (heading, heading)


def delta_metrics(policy: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {name: policy[name] - baseline[name] for name in policy.keys() & baseline.keys()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--viewer", choices=("none", "viser"), default="none")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--viewer-steps", type=int, default=0)
    parser.add_argument("--suite", choices=("none", "standard"), default="standard")
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--compare-zero", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg.env.num_envs = 1
    cfg.env.domain_randomization = False
    cfg.env.observation_noise = False
    cfg.env.interval_push = False
    payload = torch.load(args.checkpoint, map_location=cfg.experiment.device, weights_only=False)
    video = cfg.eval.video or args.video_dir is not None
    env = LatentFlowMacroEnv(cfg, render_mode="rgb_array" if video else None)
    try:
        actor_obs = env.reset(cfg.experiment.seed)
        frame_dim = payload.get("frame_dim", actor_obs.shape[-1] // cfg.policy.frame_stack)
        policy = FlowPolicy(frame_dim, cfg.latent.flow_dim, cfg.policy.frame_stack, cfg.policy.hidden_dims, cfg.policy.activation, cfg.policy.log_std_min, cfg.policy.log_std_max).to(cfg.experiment.device)
        policy.load_state_dict(payload["policy"])
        policy.eval()
        if args.viewer == "viser":
            set_command(env, args.speed, args.heading)
            run_viser(env, policy, args.port, args.viewer_steps or None)
            return
        scenarios = STANDARD_SCENARIOS if args.suite == "standard" else (("custom", args.speed, args.heading),)
        scenario_results = {}
        all_policy_rows = []
        all_zero_rows = []
        for scenario_index, (name, speed, heading) in enumerate(scenarios):
            set_command(env, speed, heading)
            policy_rows, zero_rows = [], []
            for episode in range(args.episodes or cfg.eval.episodes):
                seed = cfg.experiment.seed + scenario_index * 1000 + episode
                metrics, frames = rollout_episode(
                    env, policy, cfg.eval.macro_steps, seed, cfg.eval.deterministic,
                    video and episode == 0,
                )
                policy_rows.append(metrics)
                if frames:
                    directory = Path(args.video_dir or cfg.eval.video_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    imageio.mimsave(directory / f"{name}.mp4", frames, fps=cfg.control.bfm_hz)
                if args.compare_zero:
                    baseline, _ = rollout_episode(env, policy, cfg.eval.macro_steps, seed, True, False, zero_flow=True)
                    zero_rows.append(baseline)
            policy_summary = mean_metrics(policy_rows)
            result = {"command": {"speed": speed, "heading": heading}, "policy": policy_summary, "episodes": policy_rows}
            if zero_rows:
                zero_summary = mean_metrics(zero_rows)
                result.update({"zero_flow": zero_summary, "policy_minus_zero": delta_metrics(policy_summary, zero_summary)})
                all_zero_rows.extend(zero_rows)
            scenario_results[name] = result
            all_policy_rows.extend(policy_rows)
        aggregate = {"policy": mean_metrics(all_policy_rows)}
        if all_zero_rows:
            aggregate["zero_flow"] = mean_metrics(all_zero_rows)
            aggregate["policy_minus_zero"] = delta_metrics(aggregate["policy"], aggregate["zero_flow"])
        summary = {"aggregate": aggregate, "scenarios": scenario_results}
        output = Path(cfg.paths.run_dir) / cfg.experiment.name / "flow_eval.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["aggregate"], indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
