import pytest
import torch
from pathlib import Path

import sfm_hp_history as HH


def frame(value):
    grid = torch.zeros(3, 16, 12)
    grid[2].fill_(float(value))
    return grid


def test_newest_to_oldest_and_episode_reset_padding():
    grids = torch.stack([frame(0), frame(1), frame(2), frame(100), frame(101)])
    episodes = torch.tensor([7, 7, 7, 8, 8])
    steps = torch.tensor([0, 1, 2, 0, 1])
    hp10 = HH.build_hp10(grids, episodes, steps)
    assert hp10.shape == (5, 10, 16, 12)
    assert hp10[2, :, 0, 0].tolist() == [2, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert hp10[3, :, 0, 0].tolist() == [100] * 10
    assert hp10[4, :, 0, 0].tolist() == [101] + [100] * 9


def test_no_future_leakage_and_input_order_independent():
    grids = torch.stack([frame(2), frame(0), frame(1)])
    result = HH.build_hp10(grids, torch.tensor([4, 4, 4]), torch.tensor([2, 0, 1]))
    # Row for step zero must contain no values from steps one or two.
    assert result[1, :, 0, 0].tolist() == [0] * 10
    with pytest.raises(ValueError, match="missing past step"):
        HH.build_hp10(torch.stack([frame(0), frame(2)]), torch.tensor([4, 4]), torch.tensor([0, 2]))


def test_offline_equals_online_deque():
    grids = torch.stack([frame(i) for i in range(13)])
    offline = HH.build_hp10(grids, torch.zeros(13, dtype=torch.long), torch.arange(13))
    online = HH.online_hp10_sequence(grids)
    torch.testing.assert_close(offline, online, rtol=0, atol=0)


def test_stored_dataset_hp10_equals_online_deque():
    path = Path("/home/dohyun/projects/cfm_mppi/overnight_run_07_12_sfm/dataset_id_v01/sfm_windows_g0.5.pt")
    data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    episode = int(torch.unique(data["episode"], sorted=True)[0])
    indices = torch.nonzero(data["episode"] == episode, as_tuple=False).flatten()
    indices = indices[torch.argsort(data["step"][indices])]
    grids = data["grid"][indices]
    offline = HH.build_hp10(grids, data["episode"][indices], data["step"][indices])
    online = HH.online_hp10_sequence(grids)
    torch.testing.assert_close(offline, online, rtol=0, atol=0)
