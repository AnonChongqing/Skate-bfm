import torch

from skate_bfm_flow.bfm.latent_mapper import LatentMapper


def test_tangent_mapper_radius_and_shapes():
    basis = torch.linalg.qr(torch.randn(5, 256, 16)).Q
    mapper = LatentMapper(basis)
    z = torch.randn(7, 256)
    z = mapper.project(z, 16.0)
    output = mapper(z, torch.arange(7) % 5, torch.randn(7, 16))
    assert output.z_candidate.shape == (7, 256)
    assert torch.allclose(output.z_candidate.norm(dim=-1), torch.full((7,), 16.0), atol=1e-5)
    assert torch.allclose((output.tangent_direction * z).sum(-1), torch.zeros(7), atol=1e-4)
