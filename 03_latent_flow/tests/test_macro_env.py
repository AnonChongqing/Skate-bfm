import torch


def test_live_macro_step(stage03_env):
    stage03_env.reset(43)
    flow = torch.zeros(1, stage03_env.cfg.latent.flow_dim, device=stage03_env.z_current.device)
    result = stage03_env.step(flow)
    assert result.actor_obs.shape == (1, 1965)
    assert result.reward_macro.shape == (1, 1)
    assert result.reward_components.shape == (1, 14)
    assert result.diagnostics["executed_low_steps"] == 5
