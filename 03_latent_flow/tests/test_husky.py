import torch

from skate_bfm_flow.enums import SkateMode


def test_live_feature_shapes(stage03_env):
    actor = stage03_env.reset(42)
    features = stage03_env.latest_features
    assert actor.shape == (4, 1965)
    assert features.critic_robot.shape == (4, 79)
    assert features.critic_board.shape == (4, 19)
    assert features.critic_contact.shape == (4, 26)
    assert features.critic_goal_mode.shape == (4, 16)
    assert torch.isfinite(actor).all()


def test_bfm_is_frozen_after_live_action(stage03_env):
    stage03_env.reset(45)
    action = stage03_env.bfm.act(stage03_env.low_env.observation, stage03_env.z_current)
    assert action.shape == (4, 29)
    stage03_env.bfm.assert_frozen()
    assert all(parameter.grad is None for parameter in stage03_env.bfm.parameters())


def test_live_macro_step(stage03_env):
    stage03_env.reset(43)
    flow = torch.zeros(4, stage03_env.cfg.latent.flow_dim, device=stage03_env.z_current.device)
    result = stage03_env.step(flow)
    assert result.actor_obs.shape == (4, 1965)
    assert result.reward_macro.shape == (4, 1)
    assert result.reward_components.shape == (4, 14)
    assert torch.all(result.diagnostics["executed_low_steps"] == 5)


def test_partial_steer_reset_sets_phase(stage03_env):
    stage03_env.low_env.cfg.initial_mode = "steer"
    env_ids = torch.tensor([0, 2], device=stage03_env.z_current.device)
    stage03_env.reset(46, env_ids)
    modes = stage03_env.scheduler.mode(stage03_env.low_env.husky_env)
    assert torch.all(modes[env_ids] == int(SkateMode.STEER))
    stage03_env.low_env.cfg.initial_mode = "push"
    stage03_env.reset(46)


def test_snapshot_roundtrip(stage03_env):
    stage03_env.reset(44)
    snapshot = stage03_env.snapshot()
    flow = torch.zeros(4, stage03_env.cfg.latent.flow_dim, device=stage03_env.z_current.device)
    first = stage03_env.step(flow)
    robot_first = stage03_env.low_env.husky_env.robot.data.joint_pos.clone()
    board_first = stage03_env.low_env.husky_env.skateboard.data.root_link_pos_w.clone()
    stage03_env.restore(snapshot)
    second = stage03_env.step(flow)
    assert torch.allclose(robot_first, stage03_env.low_env.husky_env.robot.data.joint_pos, atol=1e-4)
    assert torch.allclose(board_first, stage03_env.low_env.husky_env.skateboard.data.root_link_pos_w, atol=1e-4)
    assert torch.allclose(first.reward_macro, second.reward_macro, atol=1e-4)
