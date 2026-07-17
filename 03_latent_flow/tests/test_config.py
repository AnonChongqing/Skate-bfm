from pathlib import Path

import pytest

from skate_bfm_flow.config import load_config


BASE = Path(__file__).resolve().parents[1] / "configs/base.yaml"


def test_config_override_and_rates():
    cfg = load_config(BASE, ["latent.flow_dim=8", "control.flow_hz=25", "control.macro_steps=2"])
    assert cfg.latent.flow_dim == 8
    assert cfg.control.macro_steps == 2


def test_invalid_rates_rejected():
    with pytest.raises(ValueError):
        load_config(BASE, ["control.flow_hz=25", "control.macro_steps=5"])
