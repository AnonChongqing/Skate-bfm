from __future__ import annotations

import argparse
from pathlib import Path

import torch

from skate_bfm_flow.algorithms.sac_trainer import SacUpdater
from skate_bfm_flow.bfm.action_preview import FrozenBfmActionPreview
from skate_bfm_flow.config import load_config, save_resolved_config
from skate_bfm_flow.data.replay_buffer import TensorReplayBuffer
from skate_bfm_flow.env.macro_env import LatentFlowMacroEnv
from skate_bfm_flow.models.flow_policy import FlowPolicy
from skate_bfm_flow.models.skate_q import TwinSkateQ
from skate_bfm_flow.q.input_builder import QInputBuilder
from skate_bfm_flow.utils.checkpoint import make_checkpoint, save_checkpoint
from skate_bfm_flow.utils.logging import RunLogger
from skate_bfm_flow.utils.seed import seed_everything


def transition(env: LatentFlowMacroEnv, flow: torch.Tensor) -> dict[str, torch.Tensor]:
    current_features = env.latest_features
    current_actor = env._stacked_actor_obs().clone()
    current_bfm = {name: value.unsqueeze(0).to(flow.device) for name, value in env.low_env.observation.items()}
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
    return values


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
    env = LatentFlowMacroEnv(cfg)
    run_dir = Path(cfg.paths.run_dir) / cfg.experiment.name / "online_sac"
    logger = RunLogger(run_dir)
    save_resolved_config(cfg, run_dir / "resolved_config.yaml")
    try:
        actor_obs = env.reset()
        frame_dim = actor_obs.shape[-1] // cfg.policy.frame_stack
        policy = FlowPolicy(frame_dim, cfg.latent.flow_dim, cfg.policy.frame_stack, cfg.policy.hidden_dims, cfg.policy.activation, cfg.policy.log_std_min, cfg.policy.log_std_max).to(cfg.experiment.device)
        preview = FrozenBfmActionPreview(env.bfm, env.adapter, cfg.q.preview.type)
        builder = QInputBuilder(cfg.q.input_profile, cfg.q.state_profile)
        zero = torch.zeros(1, cfg.latent.flow_dim, device=cfg.experiment.device)
        example = transition(env, zero)
        q_example = builder.build(
            env.latest_features, example["z_current"], example["flow"], example["previous_flow"],
            env.mapper(example["z_current"], example["mode_id"].reshape(-1), example["flow"]).z_candidate,
            torch.zeros(1, 2, device=cfg.experiment.device), torch.zeros(1, 23, device=cfg.experiment.device),
        )
        q = TwinSkateQ(builder.branch_dims(q_example), cfg.q.activation, cfg.q.final_hidden_dims).to(cfg.experiment.device)
        target_q = q.make_targets().to(cfg.experiment.device)
        q_optimizer = torch.optim.AdamW(q.parameters(), lr=cfg.q.optimizer.lr, weight_decay=cfg.q.optimizer.weight_decay)
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.policy.optimizer_lr)
        if args.policy_checkpoint:
            policy.load_state_dict(torch.load(args.policy_checkpoint, map_location=cfg.experiment.device, weights_only=False)["policy"])
        if args.q_checkpoint:
            q.load_state_dict(torch.load(args.q_checkpoint, map_location=cfg.experiment.device, weights_only=False)["q"])
            target_q.load_state_dict(q.state_dict())
        updater = SacUpdater(policy, q, target_q, env.mapper, preview, builder, q_optimizer, policy_optimizer, torch.tensor(cfg.sac.initial_alpha, device=cfg.experiment.device), None, cfg.control.gamma_macro, cfg.q.target.aggregation, cfg.q.target.uncertainty_beta, cfg.q.loss.type, cfg.q.loss.huber_delta, cfg.q.target_tau, cfg.sac.flow_magnitude, cfg.sac.flow_smoothness, cfg.q.optimizer.grad_clip)
        start_step = 0
        if args.resume:
            resumed = torch.load(args.resume, map_location=cfg.experiment.device, weights_only=False)
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
        for step in range(start_step, cfg.train.steps):
            if step < cfg.sac.random_steps:
                flow = torch.empty(1, cfg.latent.flow_dim, device=cfg.experiment.device).uniform_(-1.0, 1.0)
            else:
                flow = policy.sample(env._stacked_actor_obs()).action.detach()
            item = transition(env, flow)
            if bool(item["terminated"].item() or item["truncated"].item()):
                env.reset(cfg.experiment.seed + step + 1)
            replay.add(item)
            if replay.size >= max(cfg.sac.update_after, cfg.sac.batch_size):
                batch = replay.sample(cfg.sac.batch_size)
                metrics = updater.update(batch)
                if step % cfg.train.log_interval == 0:
                    logger.log(step, metrics)
                    print(step, metrics)
            if step % cfg.train.checkpoint_interval == 0 and step:
                checkpoint = Path(cfg.paths.checkpoint_dir) / cfg.experiment.name / f"sac_{step:08d}.pt"
                save_checkpoint(make_checkpoint(
                    policy=policy.state_dict(), q=q.state_dict(), target_q=target_q.state_dict(),
                    q_optimizer=q_optimizer.state_dict(), policy_optimizer=policy_optimizer.state_dict(),
                    alpha_optimizer=updater.alpha_optimizer.state_dict(), log_alpha=updater.log_alpha.detach(),
                    frame_dim=frame_dim, branch_dims=q.q1.branch_dims, flow_dim=cfg.latent.flow_dim,
                    q_input_profile=cfg.q.input_profile, preview_type=cfg.q.preview.type,
                    training_step=step + 1, environment_step=env.low_env.husky_env.common_step_counter,
                    replay_metadata={"size": replay.size, "capacity": replay.capacity}, config=cfg.model_dump(mode="json"),
                ), checkpoint)
        final_checkpoint = Path(cfg.paths.checkpoint_dir) / cfg.experiment.name / "sac_final.pt"
        save_checkpoint(make_checkpoint(
            policy=policy.state_dict(), q=q.state_dict(), target_q=target_q.state_dict(),
            q_optimizer=q_optimizer.state_dict(), policy_optimizer=policy_optimizer.state_dict(),
            alpha_optimizer=updater.alpha_optimizer.state_dict(), log_alpha=updater.log_alpha.detach(),
            frame_dim=frame_dim, branch_dims=q.q1.branch_dims, flow_dim=cfg.latent.flow_dim,
            q_input_profile=cfg.q.input_profile, preview_type=cfg.q.preview.type,
            training_step=cfg.train.steps, environment_step=env.low_env.husky_env.common_step_counter,
            replay_metadata={"size": replay.size, "capacity": replay.capacity}, config=cfg.model_dump(mode="json"),
        ), final_checkpoint)
        print(f"online SAC complete; saved {final_checkpoint}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
