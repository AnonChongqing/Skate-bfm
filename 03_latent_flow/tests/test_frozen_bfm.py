import torch


def test_bfm_is_frozen_after_live_action(stage03_env):
    stage03_env.reset(45)
    action = stage03_env.bfm.act(stage03_env.low_env.observation, stage03_env.z_current)
    assert action.shape == (1, 29)
    stage03_env.bfm.assert_frozen()
    assert all(parameter.grad is None for parameter in stage03_env.bfm.parameters())
