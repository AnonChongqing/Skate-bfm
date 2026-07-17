import torch


def test_live_feature_shapes(stage03_env):
    actor = stage03_env.reset(42)
    features = stage03_env.latest_features
    assert actor.shape == (1, 1965)
    assert features.critic_robot.shape == (1, 79)
    assert features.critic_board.shape == (1, 19)
    assert features.critic_contact.shape == (1, 26)
    assert features.critic_goal_mode.shape == (1, 16)
    assert torch.isfinite(actor).all()
