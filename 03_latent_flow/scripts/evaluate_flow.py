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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--viewer", choices=("none", "viser"), default="none")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--viewer-steps", type=int, default=0)
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
            run_viser(env, policy, args.port, args.viewer_steps or None)
            return
        rows = []
        for episode in range(args.episodes or cfg.eval.episodes):
            metrics, frames = rollout_episode(env, policy, cfg.eval.macro_steps, cfg.experiment.seed + episode, cfg.eval.deterministic, video and episode == 0)
            rows.append(metrics)
            if frames:
                directory = Path(args.video_dir or cfg.eval.video_dir)
                directory.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(directory / "episode_000.mp4", frames, fps=cfg.control.flow_hz)
        summary = {"aggregate": mean_metrics(rows), "episodes": rows}
        output = Path(cfg.paths.run_dir) / cfg.experiment.name / "flow_eval.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["aggregate"], indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
