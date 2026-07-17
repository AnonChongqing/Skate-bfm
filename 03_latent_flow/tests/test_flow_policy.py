import torch

from skate_bfm_flow.models.flow_policy import FlowPolicy


def test_tanh_gaussian_shapes_and_log_prob():
    policy = FlowPolicy(20, 8, frame_stack=5, hidden_dims=[64, 32])
    sample = policy.sample(torch.randn(6, 100))
    assert sample.action.shape == (6, 8)
    assert sample.log_prob.shape == (6, 1)
    assert torch.all(sample.action.abs() <= 1.0)
    assert torch.isfinite(sample.log_prob).all()
