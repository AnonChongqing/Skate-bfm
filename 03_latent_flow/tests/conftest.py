import sys
from pathlib import Path

import pytest
import torch

STAGE_ROOT = Path(__file__).resolve().parents[1]
for path in (STAGE_ROOT, STAGE_ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def stage03_env():
    model = Path("/63data1/hwh_data/Skate-bfm/models/bfm0/checkpoint/model/model.safetensors")
    basis = Path("/63data1/hwh_data/Skate-bfm/latent_basis/skate_mode_basis_v0.pt")
    if not torch.cuda.is_available() or not model.exists() or not basis.exists():
        pytest.skip("Stage 03 GPU integration assets are unavailable")
    from skate_bfm_flow.config import load_config
    from skate_bfm_flow.env.macro_env import LatentFlowMacroEnv

    cfg = load_config(STAGE_ROOT / "configs/base.yaml", ["env.num_envs=4"])
    env = LatentFlowMacroEnv(cfg)
    yield env
    env.close()
