import torch

from skate_bfm_flow.bfm.batch_action_adapter import BatchActionAdapter


def test_batch_adapter_matches_reference_formula():
    ids = torch.arange(23)
    default = torch.linspace(-0.2, 0.2, 23)
    scale = torch.linspace(0.1, 0.3, 23)
    adapter = BatchActionAdapter(ids, default, scale, reference_blend=0.0, action_gain=1.25)
    action = torch.randn(4, 29)
    expected_target = adapter.bfm_default[ids] + action[:, ids] * adapter.bfm_scales[ids] * 5.0 * 1.25
    expected = (expected_target - default) / scale
    assert torch.allclose(adapter(action), expected)
