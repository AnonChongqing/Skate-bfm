from __future__ import annotations

import torch

from ..env.macro_env import LatentFlowMacroEnv
from ..models.flow_policy import FlowPolicy


@torch.no_grad()
def rollout_episode(env: LatentFlowMacroEnv, policy: FlowPolicy, macro_steps: int, seed: int, deterministic: bool = True, capture_frames: bool = False) -> tuple[dict[str, float], list]:
    actor_obs = env.reset(seed)
    env.capture_low_frames = capture_frames
    env.captured_frames = [env.render()] if capture_frames else []
    total_return = 0.0
    retention, contact, distance = [], [], []
    start_board_x = float(env.low_env.husky_env.skateboard.data.root_link_pos_w[0, 0])
    terminated = truncated = False
    fell_over = contact_loss = illegal_contact = False
    low_steps = 0.0
    for _ in range(macro_steps):
        flow = policy.sample(actor_obs, deterministic=deterministic).action
        result = env.step(flow)
        actor_obs = result.actor_obs
        total_return += float(result.reward_macro.item())
        executed = max(1.0, result.diagnostics["executed_low_steps"])
        low_steps += executed
        retention.append(float(result.reward_components[0, 8]) / executed)
        contact.append(float(result.reward_components[0, 6]) / executed)
        distance.append(result.diagnostics["board_distance"])
        terminated = bool(result.terminated.item())
        truncated = bool(result.truncated.item())
        fell_over = fell_over or bool(result.diagnostics["fell_over"])
        contact_loss = contact_loss or bool(result.diagnostics["feet_off_board"])
        illegal_contact = illegal_contact or bool(result.diagnostics["illegal_contact"])
        if terminated or truncated:
            break
    board_progress = float(env.low_env.husky_env.skateboard.data.root_link_pos_w[0, 0]) - start_board_x
    metrics = {
        "episode_return": total_return, "fall": float(fell_over), "contact_loss": float(contact_loss),
        "illegal_contact": float(illegal_contact), "terminated": float(terminated), "timeout": float(truncated),
        "low_steps": low_steps, "duration_s": low_steps / env.cfg.control.bfm_hz,
        "retention": sum(retention) / max(1, len(retention)), "board_contact": sum(contact) / max(1, len(contact)),
        "board_progress": board_progress, "final_board_distance": distance[-1] if distance else 0.0,
        "success": float(board_progress > 0.5 and distance and distance[-1] < 0.5 and not terminated),
    }
    frames = list(env.captured_frames)
    env.capture_low_frames = False
    env.captured_frames = []
    return metrics, frames
