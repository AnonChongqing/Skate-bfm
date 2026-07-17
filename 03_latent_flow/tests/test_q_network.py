import torch

from skate_bfm_flow.models.skate_q import TwinSkateQ
from skate_bfm_flow.q.input_builder import PROFILE_BRANCHES
from skate_bfm_flow.schemas import QInputBatch


DIMS = {"robot": 79, "board": 19, "contact": 26, "goal_mode": 16, "z_current": 256, "z_candidate": 256, "flow": 34, "preview": 23}


def test_all_q_profiles_and_independence():
    for profile, names in PROFILE_BRANCHES.items():
        dims = {name: DIMS[name] for name in names}
        if name := ("flow" if "flow" in dims else None):
            dims[name] = 34 if profile == "full_preview" else 16
        model = TwinSkateQ(dims, final_hidden_dims=[64, 32])
        batch = QInputBatch({key: torch.randn(5, dim) for key, dim in dims.items()}, 5)
        q1, q2 = model(batch)
        assert q1.shape == q2.shape == (5, 1)
        assert all(first.data_ptr() != second.data_ptr() for first, second in zip(model.q1.parameters(), model.q2.parameters()))
        assert torch.isfinite(q1).all()
