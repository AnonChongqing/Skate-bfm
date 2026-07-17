import torch

from skate_bfm_flow.q.aggregators import aggregate
from skate_bfm_flow.q.targets import td_target


def test_aggregators():
    q1 = torch.tensor([[1.0], [3.0]])
    q2 = torch.tensor([[2.0], [1.0]])
    assert torch.equal(aggregate(q1, q2, "min"), torch.tensor([[1.0], [1.0]]))
    assert torch.equal(aggregate(q1, q2, "mean"), torch.tensor([[1.5], [2.0]]))
    assert torch.all(aggregate(q1, q2, "mean_minus_std") <= aggregate(q1, q2, "mean"))


def test_terminated_stops_but_timeout_bootstraps():
    reward = torch.ones(2, 1)
    output = td_target(
        reward, torch.tensor([[True], [False]]), torch.tensor([[False], [True]]),
        torch.full((2, 1), 2.0), torch.full((2, 1), 3.0), 0.9,
        bootstrap_on_timeout=True,
    )
    assert torch.allclose(output.target, torch.tensor([[1.0], [2.8]]))
    assert not output.target.requires_grad
