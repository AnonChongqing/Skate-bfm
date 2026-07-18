from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from skate_bfm_flow.algorithms.sac_trainer import SacUpdater
from skate_bfm_flow.bfm.action_preview import FrozenBfmActionPreview
from skate_bfm_flow.config import load_config, save_resolved_config
from skate_bfm_flow.data.branch_dataset import ANCHOR_SPLIT_VERSION
from skate_bfm_flow.data.replay_buffer import TensorReplayBuffer
from skate_bfm_flow.env.macro_env import LatentFlowMacroEnv
from skate_bfm_flow.env.reward_adapter import REWARD_COMPONENTS
from skate_bfm_flow.enums import MODE_NAMES
from skate_bfm_flow.models.flow_policy import FlowPolicy
from skate_bfm_flow.models.skate_q import TwinSkateQ
from skate_bfm_flow.q.input_builder import QInputBuilder
from skate_bfm_flow.utils.checkpoint import (
    dated_checkpoint_dir,
    make_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from skate_bfm_flow.utils.logging import MetricAccumulator, RunLogger
from skate_bfm_flow.utils.seed import seed_everything


def transition(
    env: LatentFlowMacroEnv,
    flow: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    current_features = env.latest_features
    current_actor = env._stacked_actor_obs().clone()
    current_bfm = {name: value.to(flow.device) for name, value in env.low_env.observation.items()}
    current_z = env.z_current.clone()
    current_previous = env.previous_flow.clone()
    mode = current_features.mode_id.clone()
    result = env.step(flow)
    values = {
        "flow_actor_obs": current_actor, "critic_robot": current_features.critic_robot,
        "critic_board": current_features.critic_board, "critic_contact": current_features.critic_contact,
        "critic_goal_mode": current_features.critic_goal_mode, "bfm_state": current_bfm["state"],
        "bfm_history_actor": current_bfm["history_actor"], "bfm_last_action": current_bfm["last_action"],
        "bfm_privileged_state": current_bfm["privileged_state"], "z_current": current_z,
        "mode_id": mode.unsqueeze(-1), "flow": flow, "previous_flow": current_previous,
        "reward_macro": result.reward_macro, "reward_components": result.reward_components,
        "terminated": result.terminated, "truncated": result.truncated,
        "next_flow_actor_obs": result.actor_obs, "next_critic_robot": result.features.critic_robot,
        "next_critic_board": result.features.critic_board, "next_critic_contact": result.features.critic_contact,
        "next_critic_goal_mode": result.features.critic_goal_mode,
        "next_bfm_state": result.bfm_obs["state"], "next_bfm_history_actor": result.bfm_obs["history_actor"],
        "next_bfm_last_action": result.bfm_obs["last_action"], "next_bfm_privileged_state": result.bfm_obs["privileged_state"],
        "next_z_current": result.z_current, "next_mode_id": result.features.mode_id.unsqueeze(-1),
        "next_previous_flow": result.previous_flow,
    }
    return values, result.diagnostics


def run_policy_evaluation(
    config_path: Path,
    checkpoint: Path,
    output_dir: Path,
    suite: str,
    episodes: int,
    project_root: Path,
    cuda_visible_devices: str | None = None,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_flow.py")),
        "--config", str(config_path),
        "--checkpoint", str(checkpoint),
        "--episodes", str(episodes),
        "--suite", suite,
        "--video-dir", str(output_dir),
        "--output", str(output_dir / "metrics.json"),
    ]
    print(f"[INFO] Evaluating checkpoint {checkpoint.name} -> {output_dir}", flush=True)
    process_env = os.environ.copy()
    if cuda_visible_devices is not None:
        process_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    completed = subprocess.run(command, cwd=project_root, env=process_env, check=False)
    if completed.returncode:
        print(f"[WARN] Policy evaluation failed with exit code {completed.returncode}; training continues", flush=True)
    return completed.returncode == 0


def apply_command_curriculum(env: LatentFlowMacroEnv, step: int) -> None:
    cfg = env.cfg.curriculum
    if not cfg.enabled:
        return
    progress = min(1.0, step / cfg.ramp_steps)

    def interpolate(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
        return tuple(a + progress * (b - a) for a, b in zip(start, end, strict=True))  # type: ignore[return-value]

    command = env.low_env.husky_env.command_manager._terms["skate"]
    command.cfg.ranges.lin_vel_x = interpolate(cfg.speed_start, cfg.speed_end)
    command.cfg.ranges.heading = interpolate(cfg.heading_start, cfg.heading_end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--policy-checkpoint", default=None, help="BC policy warm start")
    parser.add_argument("--q-checkpoint", default=None, help="Offline Twin Q warm start")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
    print(
        f"[SAC] Initializing HUSKY on {cfg.experiment.device}: "
        f"parallel_envs={cfg.env.num_envs} transitions={cfg.train.steps:,} "
        f"replay={cfg.sac.replay_capacity:,} log_every={cfg.train.log_interval:,}",
        flush=True,
    )
    env = LatentFlowMacroEnv(cfg)
    run_dir = Path(cfg.paths.run_dir) / cfg.experiment.name / "online_sac"
    checkpoint_dir = dated_checkpoint_dir(cfg.paths.checkpoint_dir, cfg.experiment.name)
    logger = RunLogger(run_dir)
    resolved_config = run_dir / "resolved_config.yaml"
    save_resolved_config(cfg, resolved_config)
    print(f"[SAC] Checkpoints: {checkpoint_dir}", flush=True)
    try:
        actor_obs = env.reset()
        frame_dim = actor_obs.shape[-1] // cfg.policy.frame_stack
        policy = FlowPolicy(frame_dim, cfg.latent.flow_dim, cfg.policy.frame_stack, cfg.policy.hidden_dims, cfg.policy.activation, cfg.policy.log_std_min, cfg.policy.log_std_max).to(cfg.experiment.device)
        preview = FrozenBfmActionPreview(env.bfm, env.adapter, cfg.q.preview.type)
        builder = QInputBuilder(cfg.q.input_profile, cfg.q.state_profile)
        num_envs = env.low_env.husky_env.num_envs
        zero = torch.zeros(num_envs, cfg.latent.flow_dim, device=cfg.experiment.device)
        example, _ = transition(env, zero)
        q_example = builder.build(
            env.latest_features, example["z_current"], example["flow"], example["previous_flow"],
            env.mapper(example["z_current"], example["mode_id"].reshape(-1), example["flow"]).z_candidate,
            torch.zeros(num_envs, 2, device=cfg.experiment.device),
            torch.zeros(num_envs, 23, device=cfg.experiment.device),
        )
        q = TwinSkateQ(builder.branch_dims(q_example), cfg.q.activation, cfg.q.final_hidden_dims).to(cfg.experiment.device)
        target_q = q.make_targets().to(cfg.experiment.device)
        q_optimizer = torch.optim.AdamW(q.parameters(), lr=cfg.q.optimizer.lr, weight_decay=cfg.q.optimizer.weight_decay)
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.policy.optimizer_lr)
        if args.policy_checkpoint:
            policy_payload = torch.load(args.policy_checkpoint, map_location=cfg.experiment.device, weights_only=False)
            validate_checkpoint(policy_payload, {"flow_dim": cfg.latent.flow_dim})
            policy.load_state_dict(policy_payload["policy"])
        if args.q_checkpoint:
            q_payload = torch.load(args.q_checkpoint, map_location=cfg.experiment.device, weights_only=False)
            validate_checkpoint(q_payload, {
                "flow_dim": cfg.latent.flow_dim,
                "anchor_split_version": ANCHOR_SPLIT_VERSION,
            })
            q.load_state_dict(q_payload["q"])
            target_q.load_state_dict(q.state_dict())
        updater = SacUpdater(policy, q, target_q, env.mapper, preview, builder, q_optimizer, policy_optimizer, torch.tensor(cfg.sac.initial_alpha, device=cfg.experiment.device), None, cfg.control.gamma_macro, cfg.q.target.aggregation, cfg.q.target.uncertainty_beta, cfg.q.loss.type, cfg.q.loss.huber_delta, cfg.q.target_tau, cfg.sac.flow_magnitude, cfg.sac.flow_smoothness, cfg.q.optimizer.grad_clip)
        start_step = 0
        if args.resume:
            resumed = torch.load(args.resume, map_location=cfg.experiment.device, weights_only=False)
            validate_checkpoint(resumed, {"flow_dim": cfg.latent.flow_dim})
            policy.load_state_dict(resumed["policy"])
            q.load_state_dict(resumed["q"])
            target_q.load_state_dict(resumed["target_q"])
            updater.log_alpha.data.copy_(resumed["log_alpha"])
            if not args.weights_only:
                q_optimizer.load_state_dict(resumed["q_optimizer"])
                policy_optimizer.load_state_dict(resumed["policy_optimizer"])
                updater.alpha_optimizer.load_state_dict(resumed["alpha_optimizer"])
                start_step = int(resumed.get("training_step", 0))
        replay = TensorReplayBuffer.from_example(cfg.sac.replay_capacity, example, device=cfg.experiment.device)
        replay.add(example)
        env.reset(cfg.experiment.seed)
        collected = start_step
        update_budget = 0.0
        last_log_bucket = -1
        accumulator = MetricAccumulator()
        last_eval_step = -1

        def save_sac(path: Path) -> None:
            save_checkpoint(make_checkpoint(
                policy=policy.state_dict(), q=q.state_dict(), target_q=target_q.state_dict(),
                q_optimizer=q_optimizer.state_dict(), policy_optimizer=policy_optimizer.state_dict(),
                alpha_optimizer=updater.alpha_optimizer.state_dict(), log_alpha=updater.log_alpha.detach(),
                frame_dim=frame_dim, branch_dims=q.q1.branch_dims, flow_dim=cfg.latent.flow_dim,
                q_input_profile=cfg.q.input_profile, preview_type=cfg.q.preview.type,
                training_step=collected, environment_step=env.low_env.husky_env.common_step_counter,
                replay_metadata={"size": replay.size, "capacity": replay.capacity},
                config=cfg.model_dump(mode="json"),
            ), path)

        while collected < cfg.train.steps:
            apply_command_curriculum(env, collected)
            if collected < cfg.sac.random_steps:
                flow = torch.empty(num_envs, cfg.latent.flow_dim, device=cfg.experiment.device).uniform_(-1.0, 1.0)
            else:
                flow = policy.sample(env._stacked_actor_obs()).action.detach()
            item, diagnostics = transition(env, flow)
            replay.add(item)
            rollout_metrics = {
                "rollout/reward": item["reward_macro"].mean(),
                "rollout/terminated": item["terminated"].float().mean(),
                "rollout/truncated": item["truncated"].float().mean(),
            }
            for index, name in enumerate(REWARD_COMPONENTS):
                rollout_metrics[f"reward/{name}"] = item["reward_components"][:, index].mean()
            modes = item["mode_id"].reshape(-1)
            for mode_id, name in enumerate(MODE_NAMES):
                rollout_metrics[f"mode/{name}"] = (modes == mode_id).float().mean()
            command = env.low_env.husky_env.command_manager.get_command("skate")
            rollout_metrics["command/speed"] = command[:, 0].mean()
            rollout_metrics["command/heading_abs"] = command[:, 1].abs().mean()
            for name in (
                "board_forward_progress", "board_forward_speed", "speed_error", "heading_error",
                "board_tilt_abs", "board_distance", "root_height",
            ):
                rollout_metrics[f"physics/{name}"] = diagnostics[name].mean()
            for name, value in diagnostics.items():
                if name.startswith(("husky/", "phase/")):
                    rollout_metrics[name] = value.mean()
            accumulator.update(rollout_metrics, weight=num_envs)
            done_ids = (item["terminated"] | item["truncated"]).reshape(-1).nonzero().reshape(-1)
            if len(done_ids):
                env.reset(cfg.experiment.seed + collected + 1, done_ids)
            collected += num_envs
            if replay.size >= max(cfg.sac.update_after, cfg.sac.batch_size):
                update_budget += num_envs * cfg.sac.updates_per_macro_step
                while update_budget >= 1.0:
                    batch = replay.sample(cfg.sac.batch_size, mode_balanced=cfg.replay.sampling == "mode_balanced")
                    train_metrics = updater.update(batch)
                    accumulator.update({f"train/{key}": value for key, value in train_metrics.items()})
                    update_budget -= 1.0
            else:
                update_budget = 0.0
            log_bucket = collected // cfg.train.log_interval
            if log_bucket != last_log_bucket:
                report = accumulator.mean(reset=True)
                report.update({
                    "system/replay_size": float(replay.size),
                    "system/replay_fraction": replay.size / replay.capacity,
                    "system/parallel_envs": float(num_envs),
                    "system/update_budget": update_budget,
                    "curriculum/progress": min(1.0, collected / cfg.curriculum.ramp_steps) if cfg.curriculum.enabled else 1.0,
                })
                logger.report("Online Latent SAC", collected, cfg.train.steps, report)
                last_log_bucket = log_bucket
            checkpoint_due = collected // cfg.train.checkpoint_interval != (collected - num_envs) // cfg.train.checkpoint_interval
            eval_due = cfg.logging.eval_video and (
                collected // cfg.logging.eval_interval
                != (collected - num_envs) // cfg.logging.eval_interval
            )
            if checkpoint_due or eval_due:
                checkpoint = checkpoint_dir / f"sac_{collected:08d}.pt"
                save_sac(checkpoint)
            if eval_due:
                run_policy_evaluation(
                    resolved_config.resolve(), checkpoint,
                    run_dir / "policy_eval" / f"step_{collected:08d}",
                    cfg.logging.eval_suite, cfg.logging.eval_episodes,
                    Path(cfg.paths.project_root),
                    cfg.logging.eval_cuda_visible_devices,
                )
                last_eval_step = collected
        final_checkpoint = checkpoint_dir / "sac_final.pt"
        save_sac(final_checkpoint)
        if cfg.logging.eval_video and last_eval_step != collected:
            run_policy_evaluation(
                resolved_config.resolve(), final_checkpoint,
                run_dir / "policy_eval" / "final",
                cfg.logging.eval_suite, cfg.logging.eval_episodes,
                Path(cfg.paths.project_root),
                cfg.logging.eval_cuda_visible_devices,
            )
        print(f"online SAC complete; saved {final_checkpoint}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
