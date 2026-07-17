from pathlib import Path

import numpy as np
import pytest
import torch

from skate_bfm_flow.bfm.batch_action_adapter import BatchActionAdapter
from skate_bfm_flow.bfm.latent_basis import configured_mode_files
from skate_bfm_flow.bfm.latent_mapper import LatentMapper
from skate_bfm_flow.config import load_config
from skate_bfm_flow.data.husky_motion import JOINT_POS, load_motion
from skate_bfm_flow.data.replay_buffer import TensorReplayBuffer
from skate_bfm_flow.models.flow_policy import FlowPolicy
from skate_bfm_flow.models.skate_q import TwinSkateQ
from skate_bfm_flow.q.aggregators import aggregate
from skate_bfm_flow.q.input_builder import PROFILE_BRANCHES
from skate_bfm_flow.q.targets import td_target
from skate_bfm_flow.schemas import QInputBatch

BASE = Path(__file__).resolve().parents[1] / "configs/base.yaml"
Q_DIMS = {
    "robot": 79, "board": 19, "contact": 26, "goal_mode": 16,
    "z_current": 256, "z_candidate": 256, "flow": 34, "preview": 23,
}


def test_config_override_and_rates():
    cfg = load_config(BASE, ["latent.flow_dim=8", "control.flow_hz=25", "control.macro_steps=2"])
    assert cfg.latent.flow_dim == 8
    assert cfg.control.macro_steps == 2
    with pytest.raises(ValueError):
        load_config(BASE, ["control.flow_hz=25", "control.macro_steps=5"])


def test_parallel_training_config():
    cfg = load_config(BASE.parent / "train/large.yaml")
    assert cfg.env.num_envs == 64
    assert cfg.env.domain_randomization and cfg.env.observation_noise
    assert cfg.env.command_speed_range == (0.4, 0.8)
    assert cfg.replay.sampling == "mode_balanced"
    assert cfg.curriculum.enabled
    assert cfg.env.interval_push and cfg.branch.disable_interval_push


def test_configured_basis_keeps_prototype_and_prior():
    prototypes = {mode: f"{mode}_prototype" for mode in ("push", "mount", "steer", "dismount", "recover")}
    files = configured_mode_files(prototypes, {"push": ["prior", "push_prototype"]})
    assert files["push"] == ["push_prototype", "prior"]


def test_husky_motion_schema(tmp_path: Path):
    frames = np.zeros((3, 36), dtype=np.float32)
    frames[:, 3] = 1.0
    frames[:, JOINT_POS] = np.arange(23, dtype=np.float32)
    path = tmp_path / "motion.npy"
    np.save(path, frames)
    motion = load_motion(path)
    assert motion.joint_pos.shape == (3, 23)
    assert np.allclose(motion.phase, [0.0, 0.5, 1.0])


def test_tangent_mapper_radius_and_shapes():
    basis = torch.linalg.qr(torch.randn(5, 256, 16)).Q
    mapper = LatentMapper(basis)
    z = mapper.project(torch.randn(7, 256), 16.0)
    output = mapper(z, torch.arange(7) % 5, torch.randn(7, 16))
    assert output.z_candidate.shape == (7, 256)
    assert torch.allclose(output.z_candidate.norm(dim=-1), torch.full((7,), 16.0), atol=1e-5)
    assert torch.allclose((output.tangent_direction * z).sum(-1), torch.zeros(7), atol=1e-4)


def test_tanh_gaussian_shapes_and_log_prob():
    policy = FlowPolicy(20, 8, frame_stack=5, hidden_dims=[64, 32])
    sample = policy.sample(torch.randn(6, 100))
    assert sample.action.shape == (6, 8)
    assert sample.log_prob.shape == (6, 1)
    assert torch.all(sample.action.abs() <= 1.0)
    assert torch.isfinite(sample.log_prob).all()


def test_all_q_profiles_and_independence():
    for profile, names in PROFILE_BRANCHES.items():
        dims = {name: Q_DIMS[name] for name in names}
        if "flow" in dims:
            dims["flow"] = 34 if profile == "full_preview" else 16
        model = TwinSkateQ(dims, final_hidden_dims=[64, 32])
        batch = QInputBatch({key: torch.randn(5, dim) for key, dim in dims.items()}, 5)
        q1, q2 = model(batch)
        assert q1.shape == q2.shape == (5, 1)
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(model.q1.parameters(), model.q2.parameters()))


def test_q_aggregation_and_timeout_target():
    q1 = torch.tensor([[1.0], [3.0]])
    q2 = torch.tensor([[2.0], [1.0]])
    assert torch.equal(aggregate(q1, q2, "min"), torch.tensor([[1.0], [1.0]]))
    assert torch.equal(aggregate(q1, q2, "mean"), torch.tensor([[1.5], [2.0]]))
    assert torch.all(aggregate(q1, q2, "mean_minus_std") <= aggregate(q1, q2, "mean"))
    output = td_target(
        torch.ones(2, 1), torch.tensor([[True], [False]]), torch.tensor([[False], [True]]),
        torch.full((2, 1), 2.0), torch.full((2, 1), 3.0), 0.9, bootstrap_on_timeout=True,
    )
    assert torch.allclose(output.target, torch.tensor([[1.0], [2.8]]))
    assert not output.target.requires_grad


def test_replay_add_wrap_sample_and_save(tmp_path: Path):
    example = {"obs": torch.zeros(1, 3), "mode_id": torch.zeros(1, 1, dtype=torch.long)}
    replay = TensorReplayBuffer.from_example(5, example)
    replay.add({"obs": torch.arange(12.0).reshape(4, 3), "mode_id": torch.arange(4).reshape(4, 1)})
    replay.add({"obs": torch.ones(3, 3), "mode_id": torch.ones(3, 1, dtype=torch.long)})
    assert replay.size == 5
    assert replay.sample(3)["obs"].shape == (3, 3)
    replay.save(tmp_path / "replay.pt")
    assert (tmp_path / "replay.pt").exists()


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


def test_batch_adapter_matches_reference_formula():
    ids = torch.arange(23)
    default = torch.linspace(-0.2, 0.2, 23)
    scale = torch.linspace(0.1, 0.3, 23)
    adapter = BatchActionAdapter(ids, default, scale, reference_blend=0.0, action_gain=1.25)
    action = torch.randn(4, 29)
    expected_target = adapter.bfm_default[ids] + action[:, ids] * adapter.bfm_scales[ids] * 5.0 * 1.25
    assert torch.allclose(adapter(action), (expected_target - default) / scale)
