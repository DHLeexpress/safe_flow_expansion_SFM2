import torch

import stage3_pretrain_sfm as P


def test_pretraining_mass_gamma_trajectory_window():
    # gamma0: trajectories with 2/4 windows; gamma1: trajectories with 1/3 windows.
    episodes = torch.tensor([1, 1, 2, 2, 2, 2, 7, 8, 8, 8])
    gammas = torch.tensor([0] * 6 + [1] * 4)
    weights = P.hierarchical_sampler_weights(episodes, gammas)
    assert torch.isclose(weights.sum(), torch.tensor(1., dtype=weights.dtype))
    assert torch.isclose(weights[gammas == 0].sum(), torch.tensor(.5, dtype=weights.dtype))
    assert torch.isclose(weights[episodes == 1].sum(), torch.tensor(.25, dtype=weights.dtype))
    assert torch.isclose(weights[episodes == 2].sum(), torch.tensor(.25, dtype=weights.dtype))
    assert torch.unique(weights[episodes == 2]).numel() == 1
