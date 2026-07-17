import torch


def test_snapshot_roundtrip(stage03_env):
    stage03_env.reset(44)
    snapshot = stage03_env.snapshot()
    flow = torch.zeros(1, stage03_env.cfg.latent.flow_dim, device=stage03_env.z_current.device)
    first = stage03_env.step(flow)
    robot_first = stage03_env.low_env.husky_env.robot.data.joint_pos.clone()
    board_first = stage03_env.low_env.husky_env.skateboard.data.root_link_pos_w.clone()
    stage03_env.restore(snapshot)
    second = stage03_env.step(flow)
    assert torch.allclose(robot_first, stage03_env.low_env.husky_env.robot.data.joint_pos, atol=1e-5)
    assert torch.allclose(board_first, stage03_env.low_env.husky_env.skateboard.data.root_link_pos_w, atol=1e-5)
    assert torch.allclose(first.reward_macro, second.reward_macro, atol=1e-5)
