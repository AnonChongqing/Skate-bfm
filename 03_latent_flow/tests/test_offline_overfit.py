import torch

from skate_bfm_flow.models.skate_q import TwinSkateQ
from skate_bfm_flow.schemas import QInputBatch


def test_small_q_overfit_reduces_loss():
    torch.manual_seed(7)
    dims = {"robot": 6, "board": 4, "contact": 3, "goal_mode": 2, "z_current": 8, "flow": 2}
    branches = {name: torch.randn(24, dim) for name, dim in dims.items()}
    target = (branches["flow"][:, :1] * 2.0 + branches["board"][:, :1]).detach()
    batch = QInputBatch(branches, 24)
    model = TwinSkateQ(dims, final_hidden_dims=[32, 16])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    with torch.no_grad():
        first = sum(torch.nn.functional.mse_loss(value, target) for value in model(batch)).item()
    for _ in range(120):
        q1, q2 = model(batch)
        loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = sum(torch.nn.functional.mse_loss(value, target) for value in model(batch)).item()
    assert final < first * 0.05
