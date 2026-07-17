from __future__ import annotations

import torch

from ..env.macro_env import LatentFlowMacroEnv
from ..models.flow_policy import FlowPolicy


@torch.no_grad()
def rollout_episode(
    env: LatentFlowMacroEnv,
    policy: FlowPolicy,
    macro_steps: int,
    seed: int,
    deterministic: bool = True,
    capture_frames: bool = False,
    zero_flow: bool = False,
    initial_mode: str | None = None,
    start_phase: float | None = None,
) -> tuple[dict[str, float], list]:
    if initial_mode is not None:
        env.low_env.cfg.initial_mode = initial_mode
    actor_obs = env.reset(seed, phase=start_phase)
    env.capture_low_frames = capture_frames
    env.captured_frames = [env.render()] if capture_frames else []
    total_return = 0.0
    retention, contact, distance = [], [], []
    terminated = truncated = False
    fell_over = contact_loss = illegal_contact = False
    low_steps = 0.0
    board_progress = 0.0
    weighted_diagnostics: dict[str, float] = {}
    husky_rewards: dict[str, float] = {}
    for _ in range(macro_steps):
        flow = torch.zeros(1, env.cfg.latent.flow_dim, device=actor_obs.device) if zero_flow else policy.sample(actor_obs, deterministic=deterministic).action
        result = env.step(flow)
        actor_obs = result.actor_obs
        total_return += float(result.reward_macro.item())
        executed = max(1.0, float(result.diagnostics["executed_low_steps"][0].item()))
        low_steps += executed
        board_progress += float(result.diagnostics["board_forward_progress"][0].item())
        for name in ("speed_error", "heading_error", "board_tilt_abs"):
            weighted_diagnostics[name] = weighted_diagnostics.get(name, 0.0) + float(result.diagnostics[name][0]) * executed
        for name in ("push", "steer", "push2steer", "steer2push"):
            key = f"phase/{name}"
            weighted_diagnostics[key] = weighted_diagnostics.get(key, 0.0) + float(result.diagnostics[key][0]) * executed
        for name, value in result.diagnostics.items():
            if name.startswith("husky/"):
                husky_rewards[name] = husky_rewards.get(name, 0.0) + float(value[0])
        retention.append(float(result.reward_components[0, 8]) / executed)
        contact.append(float(result.reward_components[0, 6]) / executed)
        distance.append(float(result.diagnostics["board_distance"][0].item()))
        terminated = bool(result.terminated.item())
        truncated = bool(result.truncated.item())
        fell_over = fell_over or bool(result.diagnostics["fell_over"][0].item())
        contact_loss = contact_loss or bool(result.diagnostics["feet_off_board"][0].item())
        illegal_contact = illegal_contact or bool(result.diagnostics["illegal_contact"][0].item())
        if terminated or truncated:
            break
    metrics = {
        "episode_return": total_return, "fall": float(fell_over), "contact_loss": float(contact_loss),
        "illegal_contact": float(illegal_contact), "terminated": float(terminated), "timeout": float(truncated),
        "low_steps": low_steps, "duration_s": low_steps / env.cfg.control.bfm_hz,
        "retention": sum(retention) / max(1, len(retention)), "board_contact": sum(contact) / max(1, len(contact)),
        "board_progress": board_progress, "final_board_distance": distance[-1] if distance else 0.0,
        "success": float(board_progress > 0.5 and distance and distance[-1] < 0.5 and not terminated),
        "board_forward_speed": board_progress / max(low_steps / env.cfg.control.bfm_hz, 1e-6),
    }
    metrics.update({name: value / max(low_steps, 1.0) for name, value in weighted_diagnostics.items()})
    metrics.update(husky_rewards)
    frames = list(env.captured_frames)
    env.capture_low_frames = False
    env.captured_frames = []
    return metrics, frames
