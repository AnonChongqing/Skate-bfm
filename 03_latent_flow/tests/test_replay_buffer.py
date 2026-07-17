import tempfile
from pathlib import Path

import torch

from skate_bfm_flow.data.replay_buffer import TensorReplayBuffer


def test_replay_add_wrap_sample_and_save():
    example = {"obs": torch.zeros(1, 3), "mode_id": torch.zeros(1, 1, dtype=torch.long)}
    replay = TensorReplayBuffer.from_example(5, example)
    replay.add({"obs": torch.arange(12.0).reshape(4, 3), "mode_id": torch.arange(4).reshape(4, 1)})
    replay.add({"obs": torch.ones(3, 3), "mode_id": torch.ones(3, 1, dtype=torch.long)})
    assert replay.size == 5
    assert replay.sample(3)["obs"].shape == (3, 3)
    with tempfile.TemporaryDirectory() as directory:
        replay.save(Path(directory) / "replay.pt")
        assert (Path(directory) / "replay.pt").exists()
